"""FastAPI server entry point.

Phase 0: health endpoint + /ws WebSocket echo (JSON text and raw binary PCM).
Later phases wire physics, audio buffer/recorder, and the Gemini tick loop here.
"""

from __future__ import annotations

import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent_crowd.server")

app = FastAPI(title="Agent Crowd")

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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            message = await ws.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break
            if message.get("text") is not None:
                raw_text = message["text"]
                try:
                    payload = json.loads(raw_text)
                    await ws.send_json({"type": "echo", "data": payload})
                except json.JSONDecodeError:
                    await ws.send_json({"type": "echo", "text": raw_text})
            elif message.get("bytes") is not None:
                raw_bytes = message["bytes"]
                await ws.send_bytes(raw_bytes)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
