"""Headless shove simulation runner for Phase B verification and tuning."""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any

import numpy as np

from server.config import AGENT_RADIUS, DJ_BOOTH_Y, PHYSICS_HZ
from server.physics import PhysicsEngine
from server.shove import apply_shove_actions
from server.state import CrowdState, init_crowd

# Default synthetic 4-phase envelope: quiet intro -> loud drop -> quiet breakdown -> loud drop
DEFAULT_PHASES: tuple[tuple[str, float, float], ...] = (
    ("intro", 25.0, 0.15),
    ("drop1", 45.0, 0.88),
    ("breakdown", 45.0, 0.12),
    ("drop2", 45.0, 0.90),
)


def envelope_for_time(
    t: float,
    phases: tuple[tuple[str, float, float], ...] = DEFAULT_PHASES,
) -> tuple[str, float]:
    """Return (phase_name, instantaneous_music_energy) for timestamp t."""
    elapsed = 0.0
    for name, dur, energy in phases:
        if t < elapsed + dur:
            return name, energy
        elapsed += dur
    last_name, _, last_energy = phases[-1]
    return last_name, last_energy


def _pairwise_max_overlap(crowd: CrowdState) -> float:
    """Compute maximum pairwise overlap fraction ((r_i+r_j - d) / (r_i+r_j)) across all agent pairs."""
    n = len(crowd.agents)
    if n < 2:
        return 0.0
    pos = np.array([[a.x, a.y] for a in crowd.agents], dtype=np.float64)
    rad = np.array([a.radius for a in crowd.agents], dtype=np.float64)
    dx = pos[:, 0:1] - pos[None, :, 0]
    dy = pos[:, 1:2] - pos[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(n, k=1)
    min_dist = (rad[:, None] + rad[None, :])[iu]
    overlaps = np.maximum(0.0, min_dist - dist[iu]) / np.maximum(min_dist, 1e-6)
    return float(overlaps.max()) if overlaps.size > 0 else 0.0


def _max_wall_violation(crowd: CrowdState) -> float:
    """Return maximum distance any agent sits outside its legal per-agent-radius floor bounds."""
    worst = 0.0
    for a in crowd.agents:
        r = a.radius
        worst = max(
            worst,
            r - a.x,
            a.x - (1.0 - r),
            (DJ_BOOTH_Y + r) - a.y,
            a.y - (1.0 - r),
        )
    return max(0.0, float(worst))


def run_headless_shove_demo(
    seed: int = 42,
    phases: tuple[tuple[str, float, float], ...] = DEFAULT_PHASES,
    shove_interval: float = 1.0,
    sample_interval: float = 1.0,
) -> dict[str, Any]:
    """Run the 30 Hz physics + scripted shove crowd headlessly through a synthetic music envelope.

    Returns detailed per-second samples, per-phase summary metrics, step timings,
    exhaustion/recovery event counts, and bounds/overlap diagnostics.
    """
    crowd = init_crowd(seed=seed, condition="shove")
    crowd.playing = True
    crowd.use_dsp_proxy = True
    engine = PhysicsEngine()

    dt = 1.0 / float(PHYSICS_HZ)
    total_duration = sum(dur for _, dur, _ in phases)
    total_steps = int(round(total_duration * PHYSICS_HZ))
    shove_every_steps = max(1, int(round(shove_interval * PHYSICS_HZ)))
    sample_every_steps = max(1, int(round(sample_interval * PHYSICS_HZ)))

    window_len = int(round(5.0 * PHYSICS_HZ))
    energy_window: deque[float] = deque([phases[0][2]] * window_len, maxlen=window_len)

    step_times_ms: list[float] = []
    samples: list[dict[str, Any]] = []
    phase_buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in phases}

    prev_exhausted = [a.exhausted for a in crowd.agents]
    exhaustion_events = 0
    recovery_events = 0
    max_wall_violation = 0.0
    prev_loud = False

    for step_idx in range(total_steps):
        sim_t = step_idx * dt
        phase_name, energy = envelope_for_time(sim_t, phases)
        energy_window.append(energy)
        energy_5s = sum(energy_window) / len(energy_window)

        crowd.music_energy = energy
        crowd.energy_5s = energy_5s
        crowd.dsp = {"energy": energy, "energy_5s": energy_5s}

        is_loud = energy > 0.6
        if step_idx % shove_every_steps == 0 or is_loud != prev_loud:
            apply_shove_actions(crowd)
            prev_loud = is_loud

        t0 = time.perf_counter()
        engine.step(crowd, dt=dt)
        step_times_ms.append((time.perf_counter() - t0) * 1000.0)

        for i, a in enumerate(crowd.agents):
            if a.exhausted and not prev_exhausted[i]:
                exhaustion_events += 1
            elif not a.exhausted and prev_exhausted[i]:
                recovery_events += 1
            prev_exhausted[i] = a.exhausted

        if step_idx % sample_every_steps == 0 or step_idx == total_steps - 1:
            wall_viol = _max_wall_violation(crowd)
            if wall_viol > max_wall_violation:
                max_wall_violation = wall_viol
            max_ov = _pairwise_max_overlap(crowd)
            n = len(crowd.agents)
            snap = {
                "t": round(crowd.t, 2),
                "phase": phase_name,
                "music_energy": round(crowd.music_energy, 3),
                "energy_5s": round(crowd.energy_5s, 3),
                "w": round(crowd.w, 3),
                "mean_v": sum(a.v for a in crowd.agents) / n,
                "mean_E": sum(a.E for a in crowd.agents) / n,
                "mean_rho": sum(a.rho for a in crowd.agents) / n,
                "mean_H": sum(a.H for a in crowd.agents) / n,
                "mean_pressure": sum(a.pressure for a in crowd.agents) / n,
                "n_exhausted": sum(1 for a in crowd.agents if a.exhausted),
                "max_overlap": max_ov,
                "actions": dict(Counter(a.action for a in crowd.agents)),
            }
            samples.append(snap)
            phase_buckets[phase_name].append(snap)

    phase_metrics: dict[str, dict[str, float]] = {}
    for name, _, _ in phases:
        bucket = phase_buckets[name]
        if not bucket:
            continue
        phase_metrics[name] = {
            "start_v": bucket[0]["mean_v"],
            "end_v": bucket[-1]["mean_v"],
            "mean_v": float(np.mean([s["mean_v"] for s in bucket])),
            "max_v": float(np.max([s["mean_v"] for s in bucket])),
            "start_E": bucket[0]["mean_E"],
            "end_E": bucket[-1]["mean_E"],
            "min_E": float(np.min([s["mean_E"] for s in bucket])),
            "max_E": float(np.max([s["mean_E"] for s in bucket])),
            "mean_rho": float(np.mean([s["mean_rho"] for s in bucket])),
            "end_rho": bucket[-1]["mean_rho"],
            "mean_H": float(np.mean([s["mean_H"] for s in bucket])),
            "end_H": bucket[-1]["mean_H"],
            "max_exhausted": float(np.max([s["n_exhausted"] for s in bucket])),
            "mean_overlap": float(np.mean([s["max_overlap"] for s in bucket])),
        }

    step_arr = np.array(step_times_ms, dtype=np.float64)
    return {
        "crowd": crowd,
        "samples": samples,
        "phase_metrics": phase_metrics,
        "exhaustion_events": exhaustion_events,
        "recovery_events": recovery_events,
        "total_steps": total_steps,
        "avg_step_ms": float(step_arr.mean()),
        "p95_step_ms": float(np.percentile(step_arr, 95)),
        "max_step_ms": float(step_arr.max()),
        "max_wall_violation": max_wall_violation,
        "final_max_overlap": _pairwise_max_overlap(crowd),
    }


def _cli() -> None:
    result = run_headless_shove_demo()
    print(
        f"Headless shove demo: {result['total_steps']} steps | "
        f"avg={result['avg_step_ms']:.3f} ms p95={result['p95_step_ms']:.3f} ms "
        f"max={result['max_step_ms']:.3f} ms"
    )
    print(
        f"Exhaustion events={result['exhaustion_events']} "
        f"recovery events={result['recovery_events']} "
        f"max_wall_violation={result['max_wall_violation']:.6f} "
        f"final_max_overlap={result['final_max_overlap']:.5f}"
    )
    print("\nPhase summary:")
    for phase, m in result["phase_metrics"].items():
        print(
            f"  {phase:10s} | v: {m['start_v']:.3f}->{m['end_v']:.3f} (max {m['max_v']:.3f}) | "
            f"E: {m['start_E']:.3f}->{m['end_E']:.3f} (min {m['min_E']:.3f}) | "
            f"rho_mean={m['mean_rho']:.3f} | H_end={m['end_H']:.3f} | "
            f"max_exh={int(m['max_exhausted'])}"
        )


if __name__ == "__main__":
    _cli()
