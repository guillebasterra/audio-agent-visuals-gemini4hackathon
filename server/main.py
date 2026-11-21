"""FastAPI server entry point.

Phase C:
- GET /health, GET /api/status, GET /api/tracks, GET /tracks/{filename}
- WS /ws/echo: raw JSON + binary echo (Phase 0 diagnostic)
- WS /ws: Section 9 live contract — sends `init` on connect, streams 30 Hz `physics`
  frames, handles `clock` and `control` JSON messages (pause, seek, reset, condition),
  and buffers + records incoming 16 kHz Int16 mono PCM into `runs/<run>/session.wav`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from server.audio import RollingAudioRecorder
from server.config import PHYSICS_HZ, TRACKS_DIR
from server.physics import PhysicsEngine
from server.shove import apply_shove_actions
from server.state import VALID_CONDITIONS, CrowdState, init_crowd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent_crowd.server")

SHOVE_REFRESH_SECONDS: float = 0.5


class SimulationRuntime:
    """Owns CrowdState, PhysicsEngine, RollingAudioRecorder, WebSocket clients, and 30 Hz loop."""

    def __init__(self, seed: int = 42, condition: str = "shove") -> None:
        self.crowd: CrowdState = init_crowd(seed=seed, condition=condition)
        self.physics: PhysicsEngine = PhysicsEngine()
        self.audio: RollingAudioRecorder = RollingAudioRecorder()
        self.clients: dict[WebSocket, asyncio.Lock] = {}
        self.pcm_bytes_received: int = 0
        self._physics_task: asyncio.Task[None] | None = None
        self._shove_timer: float = SHOVE_REFRESH_SECONDS
        self._shove_dirty: bool = True
        self._prev_loud: bool = False

    def ensure_loop_started(self) -> None:
        """Start the 30 Hz physics broadcast loop if not already running."""
        if self._physics_task is None or self._physics_task.done():
            self._physics_task = asyncio.create_task(self._physics_loop(), name="physics_loop")

    async def stop(self) -> None:
        """Cancel background physics loop and close session WAV cleanly."""
        task = self._physics_task
        self._physics_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.audio.close()
        self.clients.clear()

    def mark_shove_dirty(self) -> None:
        self._shove_dirty = True

    async def _send_to_client(self, ws: WebSocket, payload: dict[str, Any]) -> bool:
        lock = self.clients.get(ws)
        if lock is None:
            return False
        try:
            async with lock:
                await ws.send_json(payload)
            return True
        except Exception:
            return False

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        dead: list[WebSocket] = []
        for ws in list(self.clients.keys()):
            ok = await self._send_to_client(ws, payload)
            if not ok:
                dead.append(ws)
        for ws in dead:
            self.clients.pop(ws, None)

    async def reset_crowd(self, seed: int | None = None, condition: str | None = None) -> None:
        """Re-initialize crowd state, physics engine, and audio recorder; broadcast fresh `init`."""
        next_seed = self.crowd.seed if seed is None else int(seed)
        next_cond = self.crowd.condition if condition is None else str(condition)
        if next_cond not in VALID_CONDITIONS:
            next_cond = self.crowd.condition
        self.crowd = init_crowd(seed=next_seed, condition=next_cond)
        self.physics.reset()
        self.audio.close()
        self.audio.clear_buffer()
        self.pcm_bytes_received = 0
        self._shove_timer = SHOVE_REFRESH_SECONDS
        self._shove_dirty = True
        self._prev_loud = False
        await self.broadcast_json(self.crowd.init_payload())

    def handle_pcm_chunk(self, raw_bytes: bytes) -> None:
        """Buffer and record a raw 16 kHz Int16 mono PCM chunk from the browser."""
        self.pcm_bytes_received += len(raw_bytes)
        self.audio.append_pcm_bytes(
            raw_bytes,
            condition=self.crowd.condition,
            seed=self.crowd.seed,
        )

    async def handle_client_json(self, msg: dict[str, Any]) -> None:
        """Process a Section 9 client -> server JSON message (`clock` or `control`)."""
        msg_type = msg.get("type")

        if msg_type == "clock":
            was_playing = self.crowd.playing
            self.crowd.update_dsp(
                t=msg.get("t"),
                playing=msg.get("playing"),
                dsp=msg.get("dsp"),
                source=msg.get("source"),
            )
            is_loud = self.crowd.music_energy > 0.6
            if (self.crowd.playing and not was_playing) or (is_loud != self._prev_loud):
                self._prev_loud = is_loud
                self.mark_shove_dirty()
            return

        if msg_type == "control":
            cond = msg.get("condition")
            if cond in VALID_CONDITIONS and cond != self.crowd.condition:
                self.crowd.condition = cond
                if cond == "shove":
                    self.mark_shove_dirty()

            if "source" in msg and msg["source"] is not None:
                self.crowd.source = str(msg["source"])
            if "label" in msg and msg["label"] is not None:
                self.crowd.source_label = str(msg["label"])

            action = msg.get("action")
            if action == "play":
                self.crowd.playing = True
                if self.crowd.condition == "shove":
                    self.mark_shove_dirty()
            elif action == "pause":
                self.crowd.playing = False
            elif action == "seek":
                # Section 6: on seek, clear the rolling audio buffer
                self.audio.clear_buffer()
                if "t" in msg and msg["t"] is not None:
                    self.crowd.t = float(msg["t"])
            elif action == "reset":
                await self.reset_crowd(seed=msg.get("seed"), condition=cond)
            return

        logger.debug("Ignoring unrecognized WS message type: %r", msg_type)

    def status_snapshot(self) -> dict[str, Any]:
        agents = self.crowd.agents
        n = max(1, len(agents))
        by_arch: dict[str, dict[str, float]] = {}
        for arch in ("diehard", "dancer", "between"):
            group = [a for a in agents if a.archetype == arch]
            if group:
                by_arch[arch] = {
                    "count": len(group),
                    "mean_v": round(sum(a.v for a in group) / len(group), 4),
                    "mean_y": round(sum(a.y for a in group) / len(group), 4),
                    "mean_rho": round(sum(a.rho for a in group) / len(group), 4),
                    "mean_E": round(sum(a.E for a in group) / len(group), 4),
                    "mean_H": round(sum(a.H for a in group) / len(group), 4),
                    "mean_speed": round(
                        sum((a.vx * a.vx + a.vy * a.vy) ** 0.5 for a in group) / len(group), 4
                    ),
                }
        return {
            "t": round(self.crowd.t, 3),
            "playing": self.crowd.playing,
            "condition": self.crowd.condition,
            "seed": self.crowd.seed,
            "w": round(self.crowd.w, 3),
            "music_energy": round(self.crowd.music_energy, 3),
            "energy_5s": round(self.crowd.energy_5s, 3),
            "buffered_seconds": round(self.audio.buffered_seconds(), 3),
            "recorded_samples": self.audio.recorded_samples,
            "chunks_received": self.audio.chunks_received,
            "pcm_bytes_received": self.pcm_bytes_received,
            "session_wav": str(self.audio.session_wav_path) if self.audio.session_wav_path else None,
            "mean_v": round(sum(a.v for a in agents) / n, 4),
            "mean_E": round(sum(a.E for a in agents) / n, 4),
            "mean_rho": round(sum(a.rho for a in agents) / n, 4),
            "mean_H": round(sum(a.H for a in agents) / n, 4),
            "max_H": round(max((a.H for a in agents), default=0.0), 4),
            "top20_diehard": sum(1 for a in sorted(agents, key=lambda ag: ag.y)[:20] if a.archetype == "diehard"),
            "exhausted_count": sum(1 for a in agents if a.exhausted),
            "resting_or_retreating": sum(1 for a in agents if a.action in ("rest", "retreat")),
            "pushing_count": sum(1 for a in agents if a.action == "push"),
            "by_archetype": by_arch,
        }

    async def _physics_loop(self) -> None:
        dt = 1.0 / float(PHYSICS_HZ)
        next_tick = time.perf_counter()

        while True:
            if self.crowd.playing:
                if self.crowd.condition == "shove":
                    self._shove_timer += dt
                    is_loud = self.crowd.music_energy > 0.6
                    if (
                        self._shove_dirty
                        or self._shove_timer >= SHOVE_REFRESH_SECONDS
                        or is_loud != self._prev_loud
                    ):
                        apply_shove_actions(self.crowd)
                        self._shove_timer = 0.0
                        self._shove_dirty = False
                        self._prev_loud = is_loud
                self.physics.step(self.crowd, dt=dt)

            if self.clients:
                await self.broadcast_json(self.crowd.physics_payload())

            next_tick += dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
                await asyncio.sleep(0)


runtime = SimulationRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = runtime
    runtime.ensure_loop_started()
    logger.info("Simulation runtime started (%d Hz physics)", PHYSICS_HZ)
    try:
        yield
    finally:
        await runtime.stop()
        logger.info("Simulation runtime stopped")


app = FastAPI(title="Agent Crowd", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return runtime.status_snapshot()


@app.get("/api/tracks")
async def api_tracks() -> dict[str, list[str]]:
    if not TRACKS_DIR.exists():
        return {"tracks": []}
    files = sorted(
        p.name for p in TRACKS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg"}
    )
    return {"tracks": files}


@app.get("/tracks/{filename}")
async def serve_track(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = TRACKS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Track not found")
    return FileResponse(path)


@app.websocket("/ws/echo")
async def websocket_echo(ws: WebSocket) -> None:
    """Raw JSON + binary WebSocket echo endpoint (Phase 0 diagnostic)."""
    await ws.accept()
    logger.info("Echo WebSocket client connected")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                raw_text = message["text"]
                try:
                    payload = json.loads(raw_text)
                    await ws.send_json({"type": "echo", "data": payload})
                except json.JSONDecodeError:
                    await ws.send_json({"type": "echo", "text": raw_text})
            elif message.get("bytes") is not None:
                await ws.send_bytes(message["bytes"])
    except WebSocketDisconnect:
        logger.info("Echo WebSocket client disconnected")


@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
    autostart: int = Query(default=0),
) -> None:
    """Primary Section 9 WebSocket endpoint streaming init + 30 Hz physics frames."""
    await ws.accept()
    runtime.ensure_loop_started()

    if autostart:
        runtime.crowd.playing = True
        if runtime.crowd.condition == "shove":
            runtime.mark_shove_dirty()

    # Send `init` before subscribing to broadcasts so it is always the first frame received
    await ws.send_json(runtime.crowd.init_payload())
    runtime.clients[ws] = asyncio.Lock()
    logger.info("WebSocket client connected (clients=%d)", len(runtime.clients))

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON frame ignored")
                    continue
                if isinstance(payload, dict):
                    await runtime.handle_client_json(payload)
            elif message.get("bytes") is not None:
                runtime.handle_pcm_chunk(message["bytes"])
    except WebSocketDisconnect:
        pass
    finally:
        runtime.clients.pop(ws, None)
        logger.info("WebSocket client disconnected (clients=%d)", len(runtime.clients))
