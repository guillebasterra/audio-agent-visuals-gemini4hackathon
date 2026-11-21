"""Phase 0 verification script.

Verifies:
1. FastAPI WebSocket echo for JSON text and binary PCM frames.
2. Live Decisions API (`MODEL_DECISIONS`) with an audio block (all 3 question types:
   choice, noul, score) + realistic batch latency measurement.
3. Live Hydrogen (`MODEL_HYDROGEN`) Interactions API structured JSON call with audio
   input and thinking_level="low".

Prints raw response payloads and latencies without writing any downstream parsers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi.testclient import TestClient
from google import genai
from pydantic import BaseModel, Field

from server.config import (
    AUDIO_SR,
    GEMINI_API_KEY,
    HOUSE_CLIP_SECONDS,
    MODEL_DECISIONS,
    MODEL_HYDROGEN,
    RING_CLIP_SECONDS,
    TRACKS_DIR,
)
from server.main import app


class AskTest(BaseModel):
    text: str = Field(description="One sentence, <= 20 words, in the asker's voice.")
    claimed_keenness: float = Field(ge=0.0, le=1.0, description="0..1 how much the asker says it cares.")
    heard_section: Literal["intro", "build", "drop", "groove", "breakdown", "outro", "other"] = Field(
        description="Musical section heard in the audio clip."
    )


def extract_wav_bytes(audio_path: Path, start_sec: float, duration_sec: float) -> bytes:
    """Decode a slice of a dev MP3 into 16 kHz mono 16-bit PCM WAV bytes."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SR),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    return proc.stdout


def test_websocket_echo() -> None:
    print("\n=== 1. FASTAPI WEBSOCKET ECHO CHECK (JSON + BINARY) ===")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}, f"Health failed: {r.text}"
    print("GET /health ->", r.json())

    with client.websocket_connect("/ws") as ws:
        sample_json = {
            "type": "clock",
            "t": 30.0,
            "playing": True,
            "source": "file",
            "dsp": {"bass": 0.72, "rms": 0.55, "energy": 0.65},
        }
        ws.send_json(sample_json)
        echoed_json = ws.receive_json()
        print("JSON sent ->", json.dumps(sample_json))
        print("JSON recv ->", json.dumps(echoed_json))
        assert echoed_json == {"type": "echo", "data": sample_json}

        # 250 ms of 16 kHz Int16 mono PCM = 4000 samples = 8000 bytes
        sample_pcm = bytes(range(256)) * 31 + bytes(range(64))
        assert len(sample_pcm) == 8000
        ws.send_bytes(sample_pcm)
        echoed_bytes = ws.receive_bytes()
        print(f"BIN  sent -> {len(sample_pcm)} bytes")
        print(f"BIN  recv -> {len(echoed_bytes)} bytes, exact_match={echoed_bytes == sample_pcm}")
        assert echoed_bytes == sample_pcm
    print("WebSocket echo PASSED.")


def build_ring_questions(n_agents: int = 1) -> dict:
    questions: dict = {
        "r1_section": {
            "type": "choice",
            "instructions": "Classify the current musical section of this dancefloor audio clip.",
            "criteria": {
                "intro": "Sparse opening groove, drums or textures entering.",
                "build": "Rising tension, snare rolls, risers, filter sweeps leading toward a drop.",
                "drop": "Full-energy peak section with heavy kick and bassline hitting.",
                "groove": "Steady rhythmic dance groove with kick and bass, not peaking or building.",
                "breakdown": "Drums or bass drop out, atmospheric/melodic calm before a build.",
                "outro": "Stripping elements down toward the end of the track.",
                "other": "Silence, speech, transition noise, or unclear.",
            },
        },
        "r1_drop_eta": {
            "type": "choice",
            "instructions": "Based on the audio, how soon is a major beat drop?",
            "criteria": {
                "now": "Drop is hitting right now or within 1-2 beats.",
                "next_8_bars": "Imminent build; drop arrives within ~8 bars.",
                "within_32_bars": "Tension or breakdown developing; drop likely within ~32 bars.",
                "not_soon": "Steady groove, intro, or outro with no drop imminent.",
            },
        },
        "r1_rising": {
            "type": "noul",
            "instructions": "Is the musical energy building right now?",
        },
    }
    for i in range(n_agents):
        aid = 10 + i
        k = 0.9 if i % 3 == 0 else (0.3 if i % 3 == 1 else 0.5)
        s = 0.1 if i % 3 == 0 else (0.9 if i % 3 == 1 else 0.5)
        instr = (
            f"You are A{aid} on a dancefloor, ring 1 of 5 from the DJ (1 = front).\n"
            f"Traits: keen {k:.1f}, space {s:.1f}. Now: energy 0.72, proximity 0.85, crowding 0.62, happiness 0.58.\n"
            f"Neighbors: A7 pushing, A30 dancing. Word from ring 2: drop next_8_bars (0.70).\n"
            f"Given what you hear, what do you do now?"
        )
        questions[f"a{aid}_action"] = {
            "type": "choice",
            "instructions": instr,
            "criteria": {
                "hold": "stay where you are",
                "push": "move toward the DJ; costs energy",
                "yield": "step aside so someone behind can pass",
                "rest": "drift to open space and recover energy",
                "dance": "use the room you have to dance; costs a little energy",
            },
        }
        questions[f"a{aid}_push"] = {
            "type": "score",
            "instructions": f"If A{aid} pushes or holds ground right now, how forcefully?",
            "criteria": ["gently", "firmly", "hard"],
        }
    return questions


async def test_gemini_apis() -> None:
    tracks = sorted(TRACKS_DIR.glob("*.mp3"))
    assert tracks, f"No MP3 files found in {TRACKS_DIR}"
    track = tracks[0]
    print(f"\nUsing track: {track.name}")

    wav_8s = extract_wav_bytes(track, start_sec=45.0, duration_sec=RING_CLIP_SECONDS)
    wav_15s = extract_wav_bytes(track, start_sec=45.0, duration_sec=HOUSE_CLIP_SECONDS)
    b64_8s = base64.b64encode(wav_8s).decode("ascii")
    b64_15s = base64.b64encode(wav_15s).decode("ascii")
    print(f"Extracted 8s WAV: {len(wav_8s)} bytes ({len(b64_8s)} b64 chars)")
    print(f"Extracted 15s WAV: {len(wav_15s)} bytes ({len(b64_15s)} b64 chars)")

    features_text = "t=0:53 energy=0.62 trend=+0.08 bass_gap=0s kick_rate=2.1/s"

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})

    # --- 2. Decisions API (small: 8s audio + 5 questions covering choice, noul, score) ---
    print(f"\n=== 2. DECISIONS API ({MODEL_DECISIONS}) WITH 8s AUDIO BLOCK (5 QUESTIONS) ===")
    try:
        t0 = time.perf_counter()
        raw_resp = await client.aio.decisions.with_raw_response.create(
            model=MODEL_DECISIONS,
            input=[
                {"type": "audio", "data": b64_8s, "mime_type": "audio/wav"},
                {"type": "text", "text": features_text},
            ],
            questions=build_ring_questions(n_agents=1),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"HTTP status: {raw_resp.http_response.status_code} | latency: {dt_ms:.1f} ms")
        print("RAW JSON RESPONSE:")
        print(json.dumps(raw_resp.http_response.json(), indent=2))
        parsed = await raw_resp.parse()
        print("PARSED SDK TYPE:", type(parsed).__name__)
    except Exception as e:
        print(f"Decisions API (8s audio) FAILED: {type(e).__name__}: {e}")
        raise

    # --- 2b. Decisions API realistic full-tick load test: 1 house (15s) + 5 rings (8s x 20 agents = 100 agents) ---
    print(f"\n=== 2b. DECISIONS API FULL-TICK LOAD TEST (15s HOUSE + 5x 8s RINGS = 100 AGENTS PARALLEL) ===")
    house_questions = {
        "house_section": build_ring_questions(0)["r1_section"],
        "house_drop_eta": build_ring_questions(0)["r1_drop_eta"],
        "house_rising": build_ring_questions(0)["r1_rising"],
    }
    ring_20_questions = build_ring_questions(n_agents=20)
    print(f"House questions: {len(house_questions)} | Per-ring questions: {len(ring_20_questions)} (x5 rings)")

    async def timed_decision(label: str, b64_audio: str, qs: dict):
        ts = time.perf_counter()
        res = await client.aio.decisions.create(
            model=MODEL_DECISIONS,
            input=[
                {"type": "audio", "data": b64_audio, "mime_type": "audio/wav"},
                {"type": "text", "text": features_text},
            ],
            questions=qs,
        )
        ms = (time.perf_counter() - ts) * 1000.0
        print(
            f"  [{label}] latency={ms:.1f} ms | answers={len(res.answers or {})} | usage={res.usage.model_dump() if res.usage else None}"
        )
        return label, ms, res

    t_par = time.perf_counter()
    results = await asyncio.gather(
        timed_decision("house_15s_3q", b64_15s, house_questions),
        timed_decision("ring1_8s_43q", b64_8s, ring_20_questions),
        timed_decision("ring2_8s_43q", b64_8s, ring_20_questions),
        timed_decision("ring3_8s_43q", b64_8s, ring_20_questions),
        timed_decision("ring4_8s_43q", b64_8s, ring_20_questions),
        timed_decision("ring5_8s_43q", b64_8s, ring_20_questions),
    )
    print(f"Full 6-call parallel wall time (100 agents + house ear): {(time.perf_counter() - t_par) * 1000.0:.1f} ms")
    # Print a quick sample from ring 1 to verify diversity/calibration
    ring_res = results[1][2]
    if ring_res.answers:
        for k in ["r1_section", "r1_drop_eta", "r1_rising", "a10_action", "a11_action", "a12_action", "a10_push"]:
            ans = ring_res.answers.get(k)
            print(f"  sample[{k}] = {ans.model_dump() if ans else None}")

    # --- 3. Hydrogen Interactions API structured call WITH audio input ---
    print(f"\n=== 3. HYDROGEN ({MODEL_HYDROGEN}) STRUCTURED CALL WITH 8s AUDIO INPUT (thinking_level=low) ===")
    prompt = (
        "You are A41 on a crowded dancefloor (keen=0.9, space=0.1, energy=0.72, happiness=0.58, ring 2). "
        "You want to move closer to the DJ, and A12 (energy medium, holding) is directly ahead of you. "
        "You have never yielded to each other. Listen to the audio clip and generate one short spoken ask "
        "to get past A12, plus your claimed keenness (0..1) and the musical section you hear."
    )
    t0 = time.perf_counter()
    raw_h = await client.aio.interactions.with_raw_response.create(
        model=MODEL_HYDROGEN,
        system_instruction="You are an agent on a dancefloor trying to get closer to the DJ.",
        input=[
            {"type": "audio", "data": b64_8s, "mime_type": "audio/wav"},
            {"type": "text", "text": prompt},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AskTest.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
        store=False,
    )
    dt_h = (time.perf_counter() - t0) * 1000.0
    print(f"HTTP status: {raw_h.http_response.status_code} | latency: {dt_h:.1f} ms")
    print("RAW JSON RESPONSE:")
    print(json.dumps(raw_h.http_response.json(), indent=2))
    parsed_h = await raw_h.parse()
    print("PARSED SDK TYPE:", type(parsed_h).__name__)
    print("interaction.output_text:", parsed_h.output_text)
    validated = AskTest.model_validate_json(parsed_h.output_text)
    print("PYDANTIC VALIDATED:", validated)

    # --- 3b. Hydrogen latency comparison: text-only ask (low) and probe minimal thinking ---
    print(f"\n=== 3b. HYDROGEN LATENCY PROBES (text-only low, audio minimal) ===")
    for label, inp, lvl in [
        ("hydrogen_text_low", [{"type": "text", "text": prompt}], "low"),
        (
            "hydrogen_audio_minimal",
            [
                {"type": "audio", "data": b64_8s, "mime_type": "audio/wav"},
                {"type": "text", "text": prompt},
            ],
            "minimal",
        ),
    ]:
        try:
            ts = time.perf_counter()
            res_p = await client.aio.interactions.create(
                model=MODEL_HYDROGEN,
                system_instruction="You are an agent on a dancefloor trying to get closer to the DJ.",
                input=inp,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": AskTest.model_json_schema(),
                },
                generation_config={"thinking_level": lvl},
                store=False,
            )
            ms = (time.perf_counter() - ts) * 1000.0
            usage = res_p.usage.model_dump() if res_p.usage else None
            print(f"  [{label}] OK latency={ms:.1f} ms | output={res_p.output_text} | usage={usage}")
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")


def main() -> None:
    test_websocket_echo()
    asyncio.run(test_gemini_apis())


if __name__ == "__main__":
    main()
