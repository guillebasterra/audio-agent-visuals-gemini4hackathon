"""Scripted shove baseline actions (plan.md section 8.4, Phase E2 local_music aware)."""

from __future__ import annotations

from server.state import CrowdState, ring_gain_for_agent


def apply_shove_actions(crowd: CrowdState) -> None:
    """Assign scripted shove actions and pushiness based on archetype, energy, density, and local_music."""
    for agent in crowd.agents:
        if agent.exhausted:
            agent.action = "retreat"
            agent.pushiness = 0.05
            continue

        local_music = (
            agent.local_music
            if agent.local_music > 0.0 or crowd.music_energy <= 0.0
            else crowd.music_energy * ring_gain_for_agent(agent)
        )

        if agent.E < 0.18:
            agent.action = "rest"
            agent.pushiness = 0.06
        elif agent.archetype == "diehard":
            if local_music < 0.24 and crowd.music_energy < 0.32:
                agent.action = "hold"
                agent.pushiness = 0.25
            else:
                agent.action = "push"
                agent.pushiness = 1.0
        elif agent.archetype == "dancer":
            if agent.rho > 0.55 and local_music < 0.45:
                agent.action = "rest"
                agent.pushiness = 0.06
            elif local_music > 0.58 and agent.v < 0.45 and agent.rho < 0.45:
                # Dancers also love loud drops and drift closer when there's room
                agent.action = "push"
                agent.pushiness = 0.28
            else:
                agent.action = "dance"
                agent.pushiness = 0.12
        else:  # "between"
            if local_music > 0.42 or crowd.music_energy > 0.55:
                agent.action = "push"
                agent.pushiness = 0.42
            else:
                agent.action = "hold"
                agent.pushiness = 0.14
