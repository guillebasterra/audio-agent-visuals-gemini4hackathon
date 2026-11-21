"""Reproducible full-MP3 experiment runner for Phase F (shove, silent, talk on same seed).

Quota-safe features:
- Probes Gemini API with a single lightweight call before each run; backs off on 429s.
- Runs conditions sequentially (never overlapping) with gentle tick pacing.
- Records exact API call / failure / 429 counts per run into `<run_dir>/run_summary.json`.
- Automatically reruns any condition whose API failure rate exceeds 5% (configurable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.audio import RollingAudioRecorder
from server.config import AUDIO_SR, FLOOR_Y_MAX, FLOOR_Y_MIN, TICK_SECONDS
from server.gemini_client import GeminiClient
from server.log import RunLogger
from server.physics import PhysicsEngine
from server.shove import apply_shove_actions
from server.state import init_crowd
from server.tick import TickLoop

# Phase E3 floor thirds (post-hoc telemetry only — never shown to models)
_Y_FRONT_EDGE = FLOOR_Y_MIN + (FLOOR_Y_MAX - FLOOR_Y_MIN) * 0.36
_Y_MID_EDGE = FLOOR_Y_MIN + (FLOOR_Y_MAX - FLOOR_Y_MIN) * 0.68

DEFAULT_PACE: dict[str, float] = {
    "shove": 0.5,
    "silent": 1.5,
    "talk": 3.0,  # real-time cadence so async Hydrogen asks land cleanly on the next tick
}


def decode_mp3_16k(mp3_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode audio file to 16 kHz mono (float32, int16)."""
    data, sr = sf.read(str(mp3_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    g = math.gcd(sr, AUDIO_SR)
    mono_16k = signal.resample_poly(mono, AUDIO_SR // g, sr // g).astype(np.float32)
    pcm_i16 = np.clip(mono_16k * 32767.0, -32768, 32767).astype("<i2")
    return mono_16k, pcm_i16


def compute_dsp_timeline(mono_16k: np.ndarray, sr: int = AUDIO_SR) -> list[dict[str, Any]]:
    """Compute 10 Hz browser-matching DSP features (energy, bass, mid, high, trend, bass_gap, kick)."""
    hop = sr // 10  # 0.1 s = 1600 samples
    nfft = 2048
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    bass_mask = (freqs >= 20) & (freqs <= 150)
    mid_mask = (freqs > 150) & (freqs <= 2000)
    high_mask = (freqs > 2000) & (freqs <= 7800)

    max_b = max_m = max_h = 1e-4
    sm_b = sm_m = sm_h = 0.0
    last_bass_t = 0.0
    bass_hist: list[float] = []
    energy_hist: list[tuple[float, float]] = []
    last_kick_t = -10.0

    timeline: list[dict[str, Any]] = []
    n_frames = len(mono_16k) // hop
    win = np.hanning(nfft).astype(np.float32)

    for i in range(n_frames):
        t = (i + 1) * 0.1
        end = (i + 1) * hop
        start = max(0, end - nfft)
        chunk = mono_16k[start:end]
        if len(chunk) < nfft:
            chunk = np.pad(chunk, (nfft - len(chunk), 0))
        spec = np.abs(np.fft.rfft(chunk * win)) / nfft

        rb = float(spec[bass_mask].mean()) if bass_mask.any() else 0.0
        rm = float(spec[mid_mask].mean()) if mid_mask.any() else 0.0
        rh = float(spec[high_mask].mean()) if high_mask.any() else 0.0

        alpha = 1.0 - math.exp(-0.1 / 0.25)
        sm_b += alpha * (rb - sm_b)
        sm_m += alpha * (rm - sm_m)
        sm_h += alpha * (rh - sm_h)

        decay = math.exp(-0.1 / 180.0)
        max_b = max(1e-4, max(sm_b, max_b * decay))
        max_m = max(1e-4, max(sm_m, max_m * decay))
        max_h = max(1e-4, max(sm_h, max_h * decay))

        nb = min(1.0, sm_b / max_b)
        nm = min(1.0, sm_m / max_m)
        nh = min(1.0, sm_h / max_h)
        energy = min(1.0, 0.50 * nb + 0.35 * nm + 0.15 * nh)

        if nb > 0.24:
            last_bass_t = t
        bass_gap = max(0.0, t - last_bass_t)

        bass_hist.append(rb)
        if len(bass_hist) > 10:
            bass_hist.pop(0)
        med_b = float(np.median(bass_hist))
        kick = False
        if (
            len(bass_hist) >= 5
            and rb > med_b * 1.45
            and rb > max_b * 0.12
            and (t - last_kick_t) >= 0.28
        ):
            kick = True
            last_kick_t = t

        energy_hist.append((t, energy))
        while energy_hist and energy_hist[0][0] < t - 8.5:
            energy_hist.pop(0)
        prev_e = energy_hist[0][1]
        trend = max(-1.0, min(1.0, energy - prev_e))

        timeline.append(
            {
                "t": round(t, 2),
                "energy": round(energy, 4),
                "bass": round(nb, 4),
                "mid": round(nm, 4),
                "high": round(nh, 4),
                "trend": round(trend, 4),
                "bass_gap": round(bass_gap, 2),
                "kick": kick,
            }
        )
    return timeline


async def wait_for_quota(client: GeminiClient, max_attempts: int = 8) -> bool:
    """Probe Gemini quota with a single lightweight call; wait with backoff on 429s."""
    delays = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 60.0, 60.0]
    for i in range(max_attempts):
        res = await client.probe_quota(timeout=12.0)
        if res.ok:
            print(
                f"[quota] OK ({res.latency_ms:.0f} ms, rate_limited_before={res.rate_limited})",
                flush=True,
            )
            return True
        delay = delays[min(i, len(delays) - 1)]
        print(
            f"[quota] probe failed (attempt {i + 1}/{max_attempts}, rl={res.rate_limited}): "
            f"{res.error} -> sleeping {delay:.0f}s",
            flush=True,
        )
        await asyncio.sleep(delay)
    return False


async def run_single_condition(
    *,
    condition: str,
    seed: int,
    mp3_path: Path,
    pcm_i16: np.ndarray,
    dsp_timeline: list[dict[str, Any]],
    duration: float,
    tick_seconds: float,
    pace: float,
) -> dict[str, Any]:
    """Run one full-MP3 condition with identical DSP, physics, and Gemini TickLoop."""
    total_dur = len(pcm_i16) / float(AUDIO_SR)
    run_dur = total_dur if duration <= 0 else min(total_dur, float(duration))
    n_ticks = int(run_dur // tick_seconds)

    client = GeminiClient()
    quota_ok = await wait_for_quota(client)
    if not quota_ok:
        await client.aclose()
        raise RuntimeError(f"Gemini quota probe failed repeatedly before {condition} run")

    crowd = init_crowd(seed=seed, condition=condition)
    crowd.playing = True
    crowd.source = "file"
    crowd.source_label = mp3_path.name

    audio = RollingAudioRecorder()
    audio.start_run(condition=condition, seed=seed)

    physics = PhysicsEngine()
    run_logger = RunLogger()

    async def _noop_bcast(_msg: dict[str, Any]) -> None:
        return None

    tick_loop = TickLoop(
        client=client,
        audio=audio,
        physics=physics,
        run_logger=run_logger,
        broadcast_fn=_noop_bcast,
        tick_seconds=tick_seconds,
    )

    print(
        f"\n=== START {condition.upper()} | seed={seed} | {run_dur:.1f}s ({n_ticks} ticks) "
        f"| pace={pace:.1f}s | dir={audio.run_dir} ===",
        flush=True,
    )

    samples_per_step = int(AUDIO_SR * 0.1)
    physics_substeps = 3  # 30 Hz per 0.1 s step
    dt_phys = 1.0 / 30.0
    shove_timer = 1.0  # trigger immediate shove assignment on step 0 (mirrors server/main.py)
    prev_loud = False

    prev_exhausted = {a.id: False for a in crowd.agents}
    exhaustion_events = 0
    sample_cursor = 0
    dsp_cursor = 0
    t_run_start = time.perf_counter()

    for tick_idx in range(1, n_ticks + 1):
        t_tick_wall = time.perf_counter()
        steps_01s = int(round(tick_seconds * 10))

        for _ in range(steps_01s):
            next_sample = min(len(pcm_i16), sample_cursor + samples_per_step)
            if next_sample > sample_cursor:
                audio.append_pcm_bytes(
                    pcm_i16[sample_cursor:next_sample].tobytes(),
                    condition=condition,
                    seed=seed,
                )
                sample_cursor = next_sample

            if dsp_cursor < len(dsp_timeline):
                d = dsp_timeline[dsp_cursor]
                dsp_cursor += 1
                lo_i = max(0, dsp_cursor - 50)
                d_with_5s = dict(d)
                d_with_5s["energy_5s"] = float(
                    np.mean([dsp_timeline[k]["energy"] for k in range(lo_i, dsp_cursor)])
                )
                crowd.update_dsp(t=d["t"], playing=True, dsp=d_with_5s)

            for _ in range(physics_substeps):
                if condition == "shove":
                    shove_timer += dt_phys
                    is_loud = crowd.music_energy > 0.6
                    if shove_timer >= 1.0 or is_loud != prev_loud:
                        apply_shove_actions(crowd)
                        shove_timer = 0.0
                        prev_loud = is_loud
                physics.step(crowd, dt=dt_phys)
                for a in crowd.agents:
                    if a.exhausted and not prev_exhausted[a.id]:
                        exhaustion_events += 1
                    prev_exhausted[a.id] = a.exhausted

        frame = await tick_loop.run_once(crowd)

        if tick_idx % 10 == 0 or tick_idx == 1 or tick_idx == n_ticks:
            st = tick_loop.stats_snapshot()
            mean_v = float(np.mean([a.v for a in crowd.agents]))
            mean_H = float(np.mean([a.H for a in crowd.agents]))
            mean_E = float(np.mean([a.E for a in crowd.agents]))
            top20 = sorted(crowd.agents, key=lambda a: a.v, reverse=True)[:20]
            dh20 = sum(1 for a in top20 if a.archetype == "diehard") / 20.0
            print(
                f"  [{condition:6s} {tick_idx:3d}/{n_ticks} t={crowd.t:5.1f}s] "
                f"house={frame['house']['section']:9s} w={crowd.w:.2f} "
                f"v={mean_v:.3f} H={mean_H:.3f} E={mean_E:.3f} dh20={dh20:.2f} "
                f"p50={st['tick_latency_p50_ms']:.0f}ms p95={st['tick_latency_p95_ms']:.0f}ms "
                f"calls={st['api_calls_total']} fail={st['api_failures']}({st['api_failure_rate']*100:.1f}%) "
                f"rl={st['rate_limit_errors']} asks={st['asks_resolved']}({st['asks_yes']}Y)",
                flush=True,
            )

        if pace > 0 and tick_idx < n_ticks:
            elapsed = time.perf_counter() - t_tick_wall
            if elapsed < pace:
                await asyncio.sleep(pace - elapsed)

    # Allow any final-tick in-flight asks a brief grace window before stats snapshot
    await asyncio.sleep(0.15)
    stats = tick_loop.stats_snapshot()
    tick_loop.negotiate.reset()
    await asyncio.sleep(0.05)

    run_dir = audio.run_dir
    run_logger.close()
    audio.close()
    await client.aclose()

    wall_total_s = time.perf_counter() - t_run_start
    front_agents = [a for a in crowd.agents if a.y <= _Y_FRONT_EDGE]
    back_agents = [a for a in crowd.agents if a.y > _Y_MID_EDGE]

    rel_run_dir = str(run_dir)
    if run_dir is not None:
        try:
            rel_run_dir = str(run_dir.resolve().relative_to(ROOT))
        except ValueError:
            rel_run_dir = str(run_dir)
    summary = {
        "condition": condition,
        "seed": seed,
        "track": mp3_path.name,
        "duration_s": round(run_dur, 1),
        "ticks": n_ticks,
        "wall_seconds": round(wall_total_s, 1),
        "run_dir": rel_run_dir,
        "exhaustion_events": exhaustion_events,
        "final_front_H": round(float(np.mean([a.H for a in front_agents])), 4) if front_agents else 0.0,
        "final_back_H": round(float(np.mean([a.H for a in back_agents])), 4) if back_agents else 0.0,
        "stats": stats,
    }
    if run_dir is not None:
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"=== DONE {condition.upper()} in {wall_total_s:.1f}s | calls={stats['api_calls_total']} "
        f"failures={stats['api_failures']} ({stats['api_failure_rate']*100:.2f}%) "
        f"rl={stats['rate_limit_errors']} p50={stats['tick_latency_p50_ms']:.0f}ms "
        f"p95={stats['tick_latency_p95_ms']:.0f}ms asks={stats['asks_resolved']}({stats['asks_yes']}Y) ===",
        flush=True,
    )
    return summary


async def run_all(args: argparse.Namespace) -> None:
    mp3_path = Path(args.track)
    if not mp3_path.is_absolute():
        mp3_path = ROOT / mp3_path
    print(f"Decoding {mp3_path.name} (16 kHz mono + 10 Hz DSP)...", flush=True)
    mono_16k, pcm_i16 = decode_mp3_16k(mp3_path)
    dsp_timeline = compute_dsp_timeline(mono_16k)
    print(
        f"Decoded {len(pcm_i16) / AUDIO_SR:.1f}s audio, {len(dsp_timeline)} DSP frames.",
        flush=True,
    )

    manifest: dict[str, Any] = {
        "track": mp3_path.name,
        "seed": args.seed,
        "max_failure_rate": args.max_failure_rate,
        "runs": {},
        "failed_attempts": [],
    }

    pace_map = {
        "shove": args.pace_shove,
        "silent": args.pace_silent,
        "talk": args.pace_talk,
    }

    for cond in args.conditions:
        pace = pace_map.get(cond, DEFAULT_PACE[cond])
        best_summary: dict[str, Any] | None = None
        for attempt in range(1, args.max_retries + 2):
            summary = await run_single_condition(
                condition=cond,
                seed=args.seed,
                mp3_path=mp3_path,
                pcm_i16=pcm_i16,
                dsp_timeline=dsp_timeline,
                duration=args.duration,
                tick_seconds=args.tick_seconds,
                pace=pace,
            )
            fail_rate = float(summary["stats"]["api_failure_rate"])
            if fail_rate <= args.max_failure_rate:
                best_summary = summary
                break
            manifest["failed_attempts"].append(
                {
                    "condition": cond,
                    "attempt": attempt,
                    "run_dir": summary["run_dir"],
                    "api_failure_rate": fail_rate,
                    "api_failures": summary["stats"]["api_failures"],
                    "api_calls_total": summary["stats"]["api_calls_total"],
                }
            )
            print(
                f"[WARN] {cond} attempt {attempt} exceeded failure budget "
                f"({fail_rate*100:.2f}% > {args.max_failure_rate*100:.1f}%). "
                f"Waiting 30s before rerun...",
                flush=True,
            )
            if attempt <= args.max_retries:
                await asyncio.sleep(30.0)
            else:
                best_summary = summary

        if best_summary is not None:
            manifest["runs"][cond] = best_summary

        # Brief breather between conditions for shared quota
        if cond != args.conditions[-1]:
            print("[quota] Cooling down 5s before next condition...", flush=True)
            await asyncio.sleep(5.0)

    out_dir = Path(args.manifest).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved experiment manifest to {args.manifest}", flush=True)
    for cond, s in manifest["runs"].items():
        st = s["stats"]
        print(
            f"  {cond:6s}: {s['run_dir']} | fail={st['api_failures']}/{st['api_calls_total']} "
            f"({st['api_failure_rate']*100:.2f}%) | p50={st['tick_latency_p50_ms']:.0f}ms "
            f"p95={st['tick_latency_p95_ms']:.0f}ms"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Run full-MP3 Agent Crowd experiments sequentially.")
    p.add_argument("--track", default="tracks/getuphereletsdance.mp3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", nargs="+", default=["shove", "silent", "talk"], choices=["shove", "silent", "talk"])
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = full track)")
    p.add_argument("--tick-seconds", type=float, default=TICK_SECONDS)
    p.add_argument("--pace-shove", type=float, default=DEFAULT_PACE["shove"])
    p.add_argument("--pace-silent", type=float, default=DEFAULT_PACE["silent"])
    p.add_argument("--pace-talk", type=float, default=DEFAULT_PACE["talk"])
    p.add_argument("--max-failure-rate", type=float, default=0.05)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--manifest", default="out/experiment_manifest.json")
    args = p.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
