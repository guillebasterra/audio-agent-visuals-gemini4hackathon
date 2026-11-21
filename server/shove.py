"""Scripted shove baseline actions (plan.md section 8.4, pushiness tuned in Phase C)."""

from __future__ import annotations

from server.state import CrowdState


def apply_shove_actions(crowd: CrowdState) -> None:
    """Assign scripted shove actions and pushiness based on archetype, energy, density, and music."""
    music_energy = crowd.music_energy

    for agent in crowd.agents:
        if agent.exhausted:
            agent.action = "retreat"
            agent.pushiness = 0.05
            continue

        if agent.E < 0.2:
            agent.action = "rest"
            agent.pushiness = 0.06
        elif agent.archetype == "diehard":
            agent.action = "push"
            agent.pushiness = 1.0
        elif agent.archetype == "dancer":
            if agent.rho > 0.4:
                agent.action = "rest"
                agent.pushiness = 0.06
            else:
                agent.action = "dance"
                agent.pushiness = 0.12
        else:  # "between"
            if music_energy > 0.6:
                agent.action = "push"
                agent.pushiness = 0.35
            else:
                agent.action = "hold"
                agent.pushiness = 0.12
