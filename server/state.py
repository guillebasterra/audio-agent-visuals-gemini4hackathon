"""Agent state, crowd state, seeded initialization, and WS payload serialization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from server.config import (
    AGENT_RADIUS,
    DENSITY_CAP,
    DENSITY_RADIUS_MULT,
    DJ_BOOTH_Y,
    DJ_POS,
    MAX_DIST_TO_DJ,
    N_AGENTS,
    N_LANES,
    RING_EDGES,
)

LANE_LABELS: list[str] = ["B", "C", "E", "G", "H", "L", "M", "P", "S", "Ś"]

VALID_ACTIONS: tuple[str, ...] = ("hold", "push", "yield", "rest", "dance", "retreat")
VALID_CONDITIONS: tuple[str, ...] = ("shove", "silent", "talk")

ARCHETYPE_BASE_TRAITS: dict[str, tuple[float, float]] = {
    "diehard": (0.9, 0.1),
    "dancer": (0.3, 0.9),
    "between": (0.5, 0.5),
}

X_MIN: float = AGENT_RADIUS
X_MAX: float = 1.0 - AGENT_RADIUS
Y_MIN: float = DJ_BOOTH_Y + AGENT_RADIUS
Y_MAX: float = 1.0 - AGENT_RADIUS


@dataclass
class Agent:
    """Single agent on the dancefloor (plan.md section 4)."""

    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    E: float = 0.8
    k: float = 0.5
    s: float = 0.5
    archetype: str = "between"
    lane: int = 0
    action: str = "hold"
    pushiness: float = 0.3
    target: tuple[float, float] = (0.5, 0.9)
    H: float = 0.5
    u: float = 0.0
    rho: float = 0.0
    v: float = 0.0
    pressure: float = 0.0
    exhausted: bool = False
    yield_until: float = 0.0
    last_heard: dict[str, Any] = field(default_factory=dict)
    action_probs: dict[str, float] = field(default_factory=dict)


def dist_to_dj(x: float, y: float) -> float:
    """Euclidean distance from floor coordinate (x, y) to the DJ booth."""
    return math.hypot(x - DJ_POS[0], y - DJ_POS[1])


def proximity_for_pos(x: float, y: float) -> float:
    """Normalized proximity v in [0, 1], where 1 is right at the DJ."""
    return max(0.0, min(1.0, 1.0 - dist_to_dj(x, y) / MAX_DIST_TO_DJ))


def ring_for_distance(dist: float) -> int:
    """Map distance to DJ into acoustic ring 1 (front) .. 5 (back)."""
    for idx, edge in enumerate(RING_EDGES, start=1):
        if dist < edge:
            return idx
    return len(RING_EDGES) + 1


def ring_for_agent(agent: Agent) -> int:
    """Return ring index 1..5 for an agent based on its current position."""
    return ring_for_distance(dist_to_dj(agent.x, agent.y))


@dataclass
class CrowdState:
    """Full mutable world state for the crowd simulation."""

    LANE_LABELS = LANE_LABELS

    agents: list[Agent]
    yielded_to: dict[int, dict[int, int]] = field(default_factory=dict)
    seed: int = 42
    condition: str = "shove"
    t: float = 0.0
    playing: bool = False
    w: float = 1.0
    music_energy: float = 0.0
    energy_5s: float = 0.0
    dsp: dict[str, Any] = field(default_factory=dict)
    use_dsp_proxy: bool = True
    lane_labels: list[str] = field(default_factory=lambda: list(LANE_LABELS))
    tick: int = 0
    source: str = "file"
    source_label: str = ""
    kick_times: list[float] = field(default_factory=list)
    house_latest: dict[str, Any] = field(default_factory=dict)
    rings_latest: dict[int, dict[str, Any]] = field(default_factory=dict)

    def record_yield(self, yielder_id: int, asker_id: int) -> None:
        """Increment yield counter yielded_to[yielder][asker]."""
        bucket = self.yielded_to.setdefault(yielder_id, {})
        bucket[asker_id] = bucket.get(asker_id, 0) + 1

    def yield_count(self, yielder_id: int, asker_id: int) -> int:
        return int(self.yielded_to.get(yielder_id, {}).get(asker_id, 0))

    def kick_rate(self, window_sec: float = 5.0) -> float:
        cutoff = self.t - window_sec
        self.kick_times = [kt for kt in self.kick_times if kt >= cutoff]
        return len(self.kick_times) / max(window_sec, 1e-3)

    def update_dsp(
        self,
        t: float | None = None,
        playing: bool | None = None,
        dsp: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        """Update playback clock and latest browser DSP features."""
        if t is not None:
            new_t = float(t)
            if new_t + 1.0 < self.t:
                self.kick_times.clear()
            self.t = new_t
        if playing is not None:
            self.playing = bool(playing)
        if source is not None:
            self.source = str(source)
        if dsp is not None:
            self.dsp = dict(dsp)
            if dsp.get("kick"):
                if not self.kick_times or (self.t - self.kick_times[-1]) >= 0.12:
                    self.kick_times.append(self.t)
            if "energy" in dsp and dsp["energy"] is not None:
                self.music_energy = max(0.0, min(1.0, float(dsp["energy"])))
            if "energy_5s" in dsp and dsp["energy_5s"] is not None:
                self.energy_5s = max(0.0, min(1.0, float(dsp["energy_5s"])))
            elif "energy" in dsp and dsp["energy"] is not None:
                self.energy_5s = self.music_energy

    def init_payload(self) -> dict[str, Any]:
        """Section 9 WebSocket init payload sent on client connect."""
        return {
            "type": "init",
            "agents": [
                {
                    "id": int(a.id),
                    "archetype": a.archetype,
                    "lane": int(a.lane),
                    "k": round(float(a.k), 4),
                    "s": round(float(a.s), 4),
                }
                for a in self.agents
            ],
            "n_lanes": len(self.lane_labels),
            "lane_labels": list(self.lane_labels),
        }

    def physics_payload(self) -> dict[str, Any]:
        """Section 9 WebSocket 30 Hz physics frame."""
        return {
            "type": "physics",
            "t": round(float(self.t), 3),
            "agents": [
                [
                    int(a.id),
                    round(float(a.x), 4),
                    round(float(a.y), 4),
                    round(float(a.vx), 4),
                    round(float(a.vy), 4),
                    round(float(a.E), 4),
                    round(float(a.rho), 4),
                    round(float(a.H), 4),
                    a.action,
                    round(float(a.pressure), 4),
                ]
                for a in self.agents
            ],
        }


WorldState = CrowdState


def _populate_initial_derived(agents: list[Agent]) -> None:
    """Compute initial v and rho so state is consistent before step 1."""
    n = len(agents)
    if n == 0:
        return
    pos = np.array([[a.x, a.y] for a in agents], dtype=np.float64)
    dx = pos[:, 0:1] - pos[None, :, 0]
    dy = pos[:, 1:2] - pos[None, :, 1]
    dist_sq = dx * dx + dy * dy
    density_r_sq = (DENSITY_RADIUS_MULT * AGENT_RADIUS) ** 2
    neighbors = dist_sq <= density_r_sq
    np.fill_diagonal(neighbors, False)
    counts = neighbors.sum(axis=1)
    rhos = np.minimum(counts, DENSITY_CAP) / float(DENSITY_CAP)

    for i, agent in enumerate(agents):
        agent.v = proximity_for_pos(agent.x, agent.y)
        agent.rho = float(rhos[i])


def init_crowd(
    seed: int = 42,
    condition: str = "shove",
    n_agents: int = N_AGENTS,
    n_lanes: int = N_LANES,
) -> CrowdState:
    """Create a deterministically seeded crowd on the TLC origin line.

    - 30% diehard (k=0.9, s=0.1), 30% dancer (k=0.3, s=0.9), 40% between (k=0.5, s=0.5)
    - Uniform noise +/-0.1 added to k and s, clipped to [0, 1]
    - Arranged in 10 lanes across the back (y in [0.85, 0.97]), 10 per lane, with gentle jitter
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {VALID_CONDITIONS}")

    rng = np.random.default_rng(seed)

    n_diehard = int(round(0.30 * n_agents))
    n_dancer = int(round(0.30 * n_agents))
    n_between = n_agents - n_diehard - n_dancer

    archetypes = (
        ["diehard"] * n_diehard
        + ["dancer"] * n_dancer
        + ["between"] * n_between
    )
    rng.shuffle(archetypes)

    usable_w = X_MAX - X_MIN
    lane_width = usable_w / max(n_lanes, 1)
    y_back_lo = 0.85
    y_back_hi = min(0.97, Y_MAX)

    agents: list[Agent] = []
    per_lane = max(1, (n_agents + n_lanes - 1) // n_lanes)

    for agent_id in range(n_agents):
        lane = agent_id // per_lane
        if lane >= n_lanes:
            lane = agent_id % n_lanes
        slot = agent_id % per_lane

        archetype = archetypes[agent_id]
        base_k, base_s = ARCHETYPE_BASE_TRAITS[archetype]
        k = float(np.clip(base_k + rng.uniform(-0.1, 0.1), 0.0, 1.0))
        s = float(np.clip(base_s + rng.uniform(-0.1, 0.1), 0.0, 1.0))

        # Stagger slots across 2 sub-columns inside each lane with gentle jitter
        lane_center_x = X_MIN + (lane + 0.5) * lane_width
        sub_col = (slot % 2) - 0.5
        sub_row = (slot // 2) / max((per_lane - 1) // 2, 1)
        base_x = lane_center_x + sub_col * (lane_width * 0.36)
        base_y = y_back_lo + sub_row * (y_back_hi - y_back_lo)

        x = float(np.clip(base_x + rng.uniform(-0.009, 0.009), X_MIN, X_MAX))
        y = float(np.clip(base_y + rng.uniform(-0.008, 0.008), y_back_lo, y_back_hi))

        agents.append(
            Agent(
                id=agent_id,
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                E=0.8,
                k=k,
                s=s,
                archetype=archetype,
                lane=lane,
                action="hold",
                pushiness=0.3,
                target=(x, y),
                H=0.5,
                u=0.0,
                rho=0.0,
                v=0.0,
                pressure=0.0,
                exhausted=False,
                last_heard={},
            )
        )

    _populate_initial_derived(agents)

    labels = list(LANE_LABELS[:n_lanes]) if n_lanes <= len(LANE_LABELS) else [
        LANE_LABELS[i % len(LANE_LABELS)] for i in range(n_lanes)
    ]

    return CrowdState(
        agents=agents,
        yielded_to={a.id: {} for a in agents},
        seed=seed,
        condition=condition,
        t=0.0,
        playing=False,
        w=1.0,
        music_energy=0.0,
        energy_5s=0.0,
        dsp={},
        use_dsp_proxy=True,
        lane_labels=labels,
    )
