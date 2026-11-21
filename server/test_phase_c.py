"""Phase C verification suite: RollingAudioRecorder, ring filters, /ws PCM + seek + session.wav.

Run with:
    .venv/bin/python -m server.test_phase_c
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from server.audio import RollingAudioRecorder, apply_ring_filter, pcm_float_to_wav_bytes
from server.config import AUDIO_SR, PCM_CHUNK_MS
from server.main import app, runtime


def _sine_int16(freq_hz: float, seconds: float, amp: float = 0.5, sr: int = AUDIO_SR) -> np.ndarray:
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t) * 32767.0).astype("<i2")


def test_rolling_buffer_and_session_wav() -> None:
    print("\n=== 1. ROLLING 30s PCM BUFFER, WRAP-AROUND, CLEAR & SESSION.WAV ===")
    with tempfile.TemporaryDirectory() as tmp:
        rec = RollingAudioRecorder(sr=AUDIO_SR, buffer_seconds=4)  # 4 s capacity for fast wrap test
        run_dir = rec.start_run(condition="shove", seed=42, run_dir=Path(tmp) / "run_test")

        # Write 6 s of 250 ms chunks (24 chunks = 6 s -> wraps 4 s ring buffer)
        chunk_samples = int(AUDIO_SR * PCM_CHUNK_MS / 1000)
        for i in range(24):
            freq = 220.0 if i < 12 else 880.0
            pcm = _sine_int16(freq, PCM_CHUNK_MS / 1000.0)
            assert pcm.size == chunk_samples
            rec.append_pcm_bytes(pcm.tobytes())

        assert abs(rec.buffered_seconds() - 4.0) < 1e-6, rec.buffered_seconds()
        assert rec.recorded_samples == 24 * chunk_samples
        clip2 = rec.clip(2.0)
        assert clip2.size == 2 * AUDIO_SR
        assert float(np.sqrt(np.mean(clip2 ** 2))) > 0.2

        # Seek clears rolling buffer without destroying recorded session.wav
        rec.clear_buffer()
        assert rec.buffered_seconds() == 0.0
        assert rec.clip(2.0).size == 0

        rec.close()
        wav_path = run_dir / "session.wav"
        assert wav_path.is_file()
        info = sf.info(str(wav_path))
        assert info.samplerate == AUDIO_SR and info.channels == 1 and info.subtype == "PCM_16"
        assert abs(info.duration - 6.0) < 0.01, info.duration
    print("PASS: rolling wrap-around, clear_buffer(), and 16 kHz PCM_16 session.wav verified")


def test_ring_filters() -> None:
    print("\n=== 2. RING FILTERS (1..5) & WAV BASE64 ENCODING ===")
    # Composite test tone: 80 Hz bass + 4 kHz high tone
    t = np.arange(AUDIO_SR * 2, dtype=np.float64) / float(AUDIO_SR)
    sig = (0.35 * np.sin(2 * np.pi * 80.0 * t) + 0.35 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)

    def band_rms(y: np.ndarray, lo: float, hi: float) -> float:
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(y.size, 1.0 / AUDIO_SR)
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.sqrt(np.mean(spec[m] ** 2)))

    r1 = apply_ring_filter(sig, ring=1)
    r2 = apply_ring_filter(sig, ring=2)
    r3 = apply_ring_filter(sig, ring=3)
    r4 = apply_ring_filter(sig, ring=4)
    r5 = apply_ring_filter(sig, ring=5)

    # Ring 1 boosts bass over Ring 2
    assert band_rms(r1, 60, 120) > band_rms(r2, 60, 120) * 1.6
    # Ring 3 runs cleanly below Nyquist
    assert r3.shape == sig.shape and np.isfinite(r3).all()
    # Ring 4 & 5 progressively attenuate 4 kHz highs and overall level
    assert band_rms(r4, 3500, 4500) < band_rms(r2, 3500, 4500) * 0.25
    assert band_rms(r5, 3500, 4500) < band_rms(r4, 3500, 4500) * 0.25
    assert float(np.sqrt(np.mean(r5 ** 2))) < float(np.sqrt(np.mean(r2 ** 2))) * 0.4

    wav_bytes = pcm_float_to_wav_bytes(r1)
    assert wav_bytes[:4] == b"RIFF"
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == AUDIO_SR and data.size == sig.size
    print("PASS: all 5 ring filters & WAV encoding verified")


def test_ws_audio_and_seek_integration() -> None:
    print("\n=== 3. FASTAPI /ws BINARY PCM + SEEK BUFFER CLEAR + /api/status ===")
    with tempfile.TemporaryDirectory() as tmp:
        with TestClient(app) as client:
            runtime.audio.start_run(condition="shove", seed=42, run_dir=Path(tmp) / "ws_run")
            with client.websocket_connect("/ws") as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                chunk = _sine_int16(440.0, 0.25).tobytes()
                assert len(chunk) == 8000
                for _ in range(4):
                    ws.send_bytes(chunk)

                # Send a clock message to flush receive loop ordering
                ws.send_json({"type": "clock", "t": 1.0, "playing": True, "dsp": {"energy": 0.85, "energy_5s": 0.8}})
                _ = ws.receive_json()

                st = client.get("/api/status").json()
                assert st["chunks_received"] == 4, st
                assert abs(st["buffered_seconds"] - 1.0) < 0.05, st

                # Seek must clear rolling buffer
                ws.send_json({"type": "control", "action": "seek", "t": 30.0})
                ws.send_json({"type": "control", "action": "pause"})
                _ = ws.receive_json()

                st2 = client.get("/api/status").json()
                assert st2["buffered_seconds"] == 0.0, st2
                assert st2["playing"] is False, st2

            runtime.audio.close()
            wav_path = Path(tmp) / "ws_run" / "session.wav"
            assert wav_path.is_file()
            info = sf.info(str(wav_path))
            assert abs(info.duration - 1.0) < 0.01, info.duration
    print("PASS: /ws PCM buffering, seek clear, pause, and session.wav recording verified")


def main() -> None:
    test_rolling_buffer_and_session_wav()
    test_ring_filters()
    test_ws_audio_and_seek_integration()
    print("\nALL PHASE C UNIT/INTEGRATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
