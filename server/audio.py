"""Rolling 30 s 16 kHz mono PCM buffer, session WAV recorder, and ring acoustic filters.

Plan.md section 6:
- Rolling 30 s buffer filled from browser binary WebSocket PCM chunks (Int16, 16 kHz mono).
- Recorder appends every chunk to `runs/<run>/session.wav` so live/dev sessions can be replayed.
- `clear()` clears the rolling buffer on browser seek.
- `clip(seconds)` returns the latest N seconds as float32 samples in [-1, 1].
- Ring filters (1 front .. 5 back) + WAV bytes / base64 encoding for Phase D Decisions calls.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal

from server.config import AUDIO_SR, BUFFER_SECONDS, RUNS_DIR

logger = logging.getLogger("agent_crowd.audio")


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _low_shelf_biquad(sr: int, cutoff_hz: float, gain_db: float, q: float = 0.707) -> tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook low-shelf biquad coefficients (RBJ)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff_hz / float(sr)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def apply_ring_filter(samples: np.ndarray, ring: int, sr: int = AUDIO_SR) -> np.ndarray:
    """Apply acoustic ring filter (plan.md section 6) to float32 mono PCM in [-1, 1].

    Ring 1 front: +3 dB, low shelf +4 dB below 120 Hz
    Ring 2 near:  unchanged
    Ring 3 mid:   low-pass 8 kHz (clamped safely below Nyquist at 16 kHz SR)
    Ring 4 far:   low-pass 3 kHz, -6 dB
    Ring 5 back:  low-pass 1.5 kHz, -12 dB
    """
    if samples.size == 0 or ring == 2:
        return samples.astype(np.float32, copy=True)

    y = samples.astype(np.float64, copy=True)
    nyq = 0.5 * float(sr)

    if ring == 1:
        b, a = _low_shelf_biquad(sr, cutoff_hz=120.0, gain_db=4.0)
        y = signal.lfilter(b, a, y)
        y *= _db_to_gain(3.0)
    elif ring == 3:
        cutoff = min(7200.0, 0.90 * nyq)
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
        y = signal.sosfilt(sos, y)
    elif ring == 4:
        cutoff = min(3000.0, 0.90 * nyq)
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
        y = signal.sosfilt(sos, y) * _db_to_gain(-6.0)
    elif ring == 5:
        cutoff = min(1500.0, 0.90 * nyq)
        sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
        y = signal.sosfilt(sos, y) * _db_to_gain(-12.0)

    return np.clip(y, -1.0, 1.0).astype(np.float32)


def pcm_float_to_wav_bytes(samples: np.ndarray, sr: int = AUDIO_SR) -> bytes:
    """Encode float32 mono samples in [-1, 1] as 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32, copy=False), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm_float_to_wav_b64(samples: np.ndarray, sr: int = AUDIO_SR) -> str:
    return base64.b64encode(pcm_float_to_wav_bytes(samples, sr=sr)).decode("ascii")


class RollingAudioRecorder:
    """Thread-safe 30 s rolling PCM buffer + streaming `session.wav` recorder."""

    def __init__(self, sr: int = AUDIO_SR, buffer_seconds: int = BUFFER_SECONDS) -> None:
        self.sr = int(sr)
        self.capacity = int(sr * buffer_seconds)
        self._buf = np.zeros(self.capacity, dtype=np.int16)
        self._write_pos: int = 0
        self._filled: int = 0
        self._lock = threading.Lock()

        self.run_dir: Optional[Path] = None
        self.session_wav_path: Optional[Path] = None
        self._wav_file: Optional[sf.SoundFile] = None
        self.recorded_samples: int = 0
        self.chunks_received: int = 0

    def start_run(self, condition: str, seed: int, run_dir: Optional[Path] = None) -> Path:
        """Open a fresh `runs/<timestamp>_<condition>_<seed>/session.wav` recorder."""
        with self._lock:
            self._close_wav_locked()
            if run_dir is None:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_dir = RUNS_DIR / f"{ts}_{condition}_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            wav_path = run_dir / "session.wav"
            self._wav_file = sf.SoundFile(
                str(wav_path),
                mode="w",
                samplerate=self.sr,
                channels=1,
                subtype="PCM_16",
                format="WAV",
            )
            self.run_dir = run_dir
            self.session_wav_path = wav_path
            self.recorded_samples = 0
            self.chunks_received = 0
            logger.info("Recording session audio to %s", wav_path)
            return run_dir

    def ensure_recording(self, condition: str, seed: int) -> Path:
        if self._wav_file is not None and self.run_dir is not None:
            return self.run_dir
        return self.start_run(condition=condition, seed=seed)

    def append_pcm_bytes(self, raw_bytes: bytes, *, condition: str = "shove", seed: int = 42) -> int:
        """Append raw little-endian Int16 mono PCM bytes to rolling buffer and session.wav."""
        n_bytes = len(raw_bytes) - (len(raw_bytes) % 2)
        if n_bytes <= 0:
            return 0
        samples = np.frombuffer(raw_bytes[:n_bytes], dtype="<i2")
        n = int(samples.size)
        if n == 0:
            return 0

        with self._lock:
            # Lazily open session.wav on first arriving audio chunk if not already recording
            if self._wav_file is None:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_dir = RUNS_DIR / f"{ts}_{condition}_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                wav_path = run_dir / "session.wav"
                self._wav_file = sf.SoundFile(
                    str(wav_path),
                    mode="w",
                    samplerate=self.sr,
                    channels=1,
                    subtype="PCM_16",
                    format="WAV",
                )
                self.run_dir = run_dir
                self.session_wav_path = wav_path
                self.recorded_samples = 0
                logger.info("Recording session audio to %s", wav_path)

            # Write into rolling circular buffer
            if n >= self.capacity:
                self._buf[:] = samples[-self.capacity :]
                self._write_pos = 0
                self._filled = self.capacity
            else:
                end = self._write_pos + n
                if end <= self.capacity:
                    self._buf[self._write_pos : end] = samples
                else:
                    first = self.capacity - self._write_pos
                    self._buf[self._write_pos :] = samples[:first]
                    self._buf[: n - first] = samples[first:]
                self._write_pos = end % self.capacity
                self._filled = min(self.capacity, self._filled + n)

            # Stream to session.wav and flush so partial recordings remain valid
            self._wav_file.write(samples)
            self._wav_file.flush()
            self.recorded_samples += n
            self.chunks_received += 1
            return n

    def clear_buffer(self) -> None:
        """Clear the rolling PCM buffer (called on browser seek)."""
        with self._lock:
            self._buf.fill(0)
            self._write_pos = 0
            self._filled = 0
            logger.info("Cleared rolling audio buffer")

    def buffered_seconds(self) -> float:
        with self._lock:
            return self._filled / float(self.sr)

    def clip_int16(self, seconds: float) -> np.ndarray:
        """Return up to the last `seconds` of buffered Int16 mono PCM."""
        want = max(1, min(self.capacity, int(round(seconds * self.sr))))
        with self._lock:
            avail = min(want, self._filled)
            if avail == 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write_pos - avail) % self.capacity
            if start + avail <= self.capacity:
                return self._buf[start : start + avail].copy()
            first = self.capacity - start
            out = np.empty(avail, dtype=np.int16)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: avail - first]
            return out

    def clip(self, seconds: float) -> np.ndarray:
        """Return the last `seconds` of buffered audio as float32 in [-1, 1]."""
        raw = self.clip_int16(seconds)
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)
        return (raw.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    def ring_clip_b64(self, ring: int, seconds: float) -> str:
        """Extract the latest `seconds` clip, apply ring filter, return base64 WAV."""
        raw = self.clip(seconds)
        if raw.size == 0:
            # 0.25 s silence fallback if called before first PCM chunk arrives
            raw = np.zeros(self.sr // 4, dtype=np.float32)
        filtered = apply_ring_filter(raw, ring=ring, sr=self.sr)
        return pcm_float_to_wav_b64(filtered, sr=self.sr)

    def house_clip_b64(self, seconds: float) -> str:
        """Extract the latest unfiltered `seconds` clip as base64 WAV."""
        raw = self.clip(seconds)
        if raw.size == 0:
            raw = np.zeros(self.sr // 4, dtype=np.float32)
        return pcm_float_to_wav_b64(raw, sr=self.sr)

    def _close_wav_locked(self) -> None:
        if self._wav_file is not None:
            try:
                self._wav_file.flush()
                self._wav_file.close()
            except Exception as exc:
                logger.warning("Error closing session.wav: %s", exc)
            finally:
                self._wav_file = None

    def close(self) -> None:
        with self._lock:
            self._close_wav_locked()
