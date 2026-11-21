"""Word-of-mouth propagation between adjacent rings & nearest-neighbor action summaries (no LLM).

Plan.md section 8.2:
- After each tick, ring n's message = {drop_eta top option + probability, rising probability}.
- Next tick, agents in ring n+1 receive "Word from ring n: ...". One ring per tick.
- Agents also see their 3 nearest neighbors' last actions.
- `silent` and `talk` get word of mouth; `shove` gets none.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from server.state import Agent, CrowdState, ring_for_agent

ACTION_GERUND: dict[str, str] = {
    "hold": "holding",
    "push": "pushing",
    "yield": "yielding",
    "rest": "resting",
    "dance": "dancing",
    "retreat": "retreating",
}


class WordOfMouth:
    """Stores previous tick's ring summaries and formats neighbor/word-of-mouth lines."""

    def __init__(self) -> None:
        # ring_n -> {"drop_eta": str, "drop_eta_p": float, "rising": float, "section": str}
        self._prev_ring_msg: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._prev_ring_msg.clear()

    def message_for_ring(self, ring: int) -> dict[str, Any]:
        """Return word heard from ring `ring - 1` during the previous tick (empty for ring 1)."""
        if ring <= 1:
            return {}
        return dict(self._prev_ring_msg.get(ring - 1, {}))

    def format_word_for_ring(self, ring: int) -> str:
        msg = self.message_for_ring(ring)
        if not msg:
            return ""
        eta = msg.get("drop_eta", "not_soon")
        eta_p = float(msg.get("drop_eta_p", 0.0))
        rising = float(msg.get("rising", 0.5))
        return f"Word from ring {ring - 1}: drop {eta} ({eta_p:.2f}), rising ({rising:.2f})."

    def update_from_tick(self, crowd: CrowdState, ring_results: dict[int, dict[str, Any]]) -> None:
        """Record this tick's ring perceptions and populate `agent.last_heard` for next tick."""
        next_msgs: dict[int, dict[str, Any]] = {}
        for r, info in ring_results.items():
            if not info:
                continue
            next_msgs[int(r)] = {
                "from_ring": int(r),
                "section": info.get("section", "groove"),
                "drop_eta": info.get("drop_eta", "not_soon"),
                "drop_eta_p": float(info.get("drop_eta_p", 0.0)),
                "rising": float(info.get("rising", 0.5)),
            }
        self._prev_ring_msg = next_msgs

        if crowd.condition == "shove":
            for a in crowd.agents:
                a.last_heard = {}
            return

        for a in crowd.agents:
            r = ring_for_agent(a)
            a.last_heard = self.message_for_ring(r)


def nearest_neighbors_text(agents: list[Agent], k: int = 3) -> dict[int, str]:
    """Compute concise 'A7 pushing, A30 resting, A12 dancing' strings for each agent."""
    n = len(agents)
    if n <= 1:
        return {a.id: "none" for a in agents}

    pos = np.array([[a.x, a.y] for a in agents], dtype=np.float64)
    dx = pos[:, 0:1] - pos[None, :, 0]
    dy = pos[:, 1:2] - pos[None, :, 1]
    dist_sq = dx * dx + dy * dy
    np.fill_diagonal(dist_sq, np.inf)

    k_eff = min(k, n - 1)
    out: dict[int, str] = {}
    for i, a in enumerate(agents):
        idxs = np.argpartition(dist_sq[i], k_eff - 1)[:k_eff]
        idxs = idxs[np.argsort(dist_sq[i, idxs])]
        parts = [f"A{agents[j].id} {ACTION_GERUND.get(agents[j].action, agents[j].action)}" for j in idxs]
        out[a.id] = ", ".join(parts)
    return out
