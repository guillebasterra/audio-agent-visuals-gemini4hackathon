"""Ring-level Decisions calls: audio perception + per-agent actions, pushiness & ask answers.

Plan.md section 8.1 + Phase D/E notes:
- Buckets agents into rings 1..5 by distance to DJ.
- One Decisions call per non-empty ring, run in parallel with the house ear.
- Every ring call carries that ring's 8 s filtered WAV audio clip + DSP features text.
- Includes each agent's state and traits in BOTH `a{id}_action` AND `a{id}_push` (isolated questions).
- Injects `a{blocker}_ans_{asker}` (noul) questions for pending asks from previous tick (talk condition).
- Exhausted agents are skipped (forced to `retreat`).
- On failure, agents in that ring keep their previous actions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from server.audio import RollingAudioRecorder
from server.config import RING_CLIP_SECONDS
from server.gemini_client import ApiResult, GeminiClient
from server.negotiate import PendingAsk
from server.sections import build_music_questions, format_features_text, parse_music_answers
from server.state import Agent, CrowdState, ring_for_agent
from server.wordofmouth import WordOfMouth, nearest_neighbors_text

logger = logging.getLogger("agent_crowd.decide")

ACTION_CRITERIA: dict[str, str] = {
    "push": "push forward toward the DJ for a front spot (best when keen is high, energy is high, and music is building or dropping)",
    "hold": "hold your current spot on the floor without pushing forward or drifting back",
    "dance": "dance in place with room around you (best when space desire is high and crowding is low)",
    "rest": "drift back to open space to recover energy (best when energy is low or music calms down)",
    "yield": "step aside so someone behind you can pass",
}

PUSH_RUBRIC: list[str] = ["gently", "firmly", "hard"]

# Pushiness mapping mode:
# - "remapped": maps score in [0, 2] -> [0.4, 1.0] (gently=0.4, firmly=0.7, hard=1.0) for active pushers
#   so silent/talk pushers remain comparable to shove (1.0) under the squared mass formula.
# - "raw": plan default score / 2.0 in [0.0, 1.0].
PUSHINESS_REMAP: bool = True


def map_push_score(score: float, action: str, remap: bool = PUSHINESS_REMAP) -> float:
    """Map Decisions `a{id}_push` expected score in [0, 2] to effective physical pushiness."""
    s = max(0.0, min(2.0, float(score)))
    if action == "push":
        if remap:
            # gently (0) -> 0.40, firmly (1) -> 0.70, hard (2) -> 1.00
            return round(0.40 + 0.30 * s, 4)
        return round(s / 2.0, 4)
    if action == "hold":
        # Mild ground-holding resistance, weaker than active pushing so heavy pushers can pass
        return round(0.10 + 0.08 * s, 4)
    if action == "dance":
        return 0.12
    if action in ("yield", "rest", "retreat"):
        return 0.06
    return round(s / 2.0, 4)


def bucket_agents_by_ring(agents: list[Agent]) -> dict[int, list[Agent]]:
    buckets: dict[int, list[Agent]] = {r: [] for r in range(1, 6)}
    for a in agents:
        buckets[ring_for_agent(a)].append(a)
    return buckets


def _agent_action_instructions(agent: Agent, ring: int, neighbors_str: str, wom_str: str) -> str:
    wom_part = f" {wom_str}" if wom_str else ""
    return (
        f"You are A{agent.id} on a dancefloor, ring {ring} of 5 from the DJ (1 = front).\n"
        f"Traits: keen {agent.k:.1f} (want front row), space {agent.s:.1f} (want room to dance). "
        f"Now: energy {agent.E:.2f}, proximity {agent.v:.2f}, crowding {agent.rho:.2f}, happiness {agent.H:.2f}.\n"
        f"Neighbors: {neighbors_str}.{wom_part}\n"
        f"Given what you hear in the music and your traits, what do you do now?"
    )


def _agent_push_instructions(agent: Agent, ring: int, neighbors_str: str, wom_str: str) -> str:
    # Include state + traits because Decisions questions are isolated (no shared memory across keys).
    wom_part = f" {wom_str}" if wom_str else ""
    return (
        f"You are A{agent.id} on a dancefloor, ring {ring} of 5 from the DJ (1 = front).\n"
        f"Traits: keen {agent.k:.1f} (want front row), space {agent.s:.1f} (want room to dance). "
        f"Now: energy {agent.E:.2f}, proximity {agent.v:.2f}, crowding {agent.rho:.2f}, happiness {agent.H:.2f}.\n"
        f"Neighbors: {neighbors_str}.{wom_part}\n"
        f"Given what you hear and your keenness for the front, how forcefully do you push toward the DJ right now?"
    )


def _ask_answer_instructions(blocker: Agent, ask: PendingAsk, crowd: CrowdState) -> str:
    r = ring_for_agent(blocker)
    y_ba = crowd.yield_count(blocker.id, ask.asker_id)
    y_ab = crowd.yield_count(ask.asker_id, blocker.id)
    return (
        f"You are A{blocker.id} on a dancefloor, ring {r} of 5 from the DJ (1 = front). "
        f"Your traits: keen {blocker.k:.1f}, space {blocker.s:.1f}. "
        f"Your state: energy {blocker.E:.2f}, proximity {blocker.v:.2f}, crowding {blocker.rho:.2f}, happiness {blocker.H:.2f}.\n"
        f"History: you yielded to A{ask.asker_id} {y_ba} times; A{ask.asker_id} yielded to you {y_ab} times.\n"
        f'A{ask.asker_id} behind you asks: "{ask.text}" (claims keenness {ask.claimed_keenness:.2f}).\n'
        f"Given what you hear in the music and your state, do you step aside and yield to A{ask.asker_id}?"
    )


@dataclass
class RingDecisionOutput:
    ring: int
    ok: bool
    music: dict[str, Any] = field(default_factory=dict)
    agent_decisions: dict[int, dict[str, Any]] = field(default_factory=dict)
    ask_answers: list[dict[str, Any]] = field(default_factory=list)
    api: ApiResult[Any] | None = None


async def run_ring_decisions(
    *,
    client: GeminiClient,
    ring: int,
    agents: list[Agent],
    b64_wav: str,
    features_text: str,
    neighbors_map: dict[int, str],
    wom_str: str,
    pending_asks_for_ring: list[PendingAsk],
    crowd: CrowdState,
    remap_pushiness: bool = PUSHINESS_REMAP,
) -> RingDecisionOutput:
    """Execute one Decisions call for a single non-empty ring."""
    questions: dict[str, dict[str, Any]] = build_music_questions(f"r{ring}")

    active_agents = [a for a in agents if not a.exhausted]
    for a in active_agents:
        nb_str = neighbors_map.get(a.id, "none")
        questions[f"a{a.id}_action"] = {
            "type": "choice",
            "instructions": _agent_action_instructions(a, ring, nb_str, wom_str),
            "criteria": ACTION_CRITERIA,
        }
        questions[f"a{a.id}_push"] = {
            "type": "score",
            "instructions": _agent_push_instructions(a, ring, nb_str, wom_str),
            "criteria": PUSH_RUBRIC,
        }

    # Inject pending ask answers targeting blockers in this ring
    ask_key_map: dict[str, PendingAsk] = {}
    agent_by_id = {a.id: a for a in agents}
    for ask in pending_asks_for_ring:
        blocker = agent_by_id.get(ask.blocker_id)
        if blocker is None:
            continue
        qkey = f"a{ask.blocker_id}_ans_{ask.asker_id}"
        if qkey in ask_key_map:
            continue
        ask_key_map[qkey] = ask
        questions[qkey] = {
            "type": "noul",
            "instructions": _ask_answer_instructions(blocker, ask, crowd),
        }

    res = await client.decide(
        input_blocks=[
            {"type": "audio", "data": b64_wav, "mime_type": "audio/wav"},
            {"type": "text", "text": features_text},
        ],
        questions=questions,
        label=f"ring_{ring}",
    )
    if not res.ok or res.value is None:
        logger.warning("Ring %d Decisions failed (%s); keeping prior actions", ring, res.error)
        return RingDecisionOutput(ring=ring, ok=False, api=res)

    answers = getattr(res.value, "answers", None) or {}
    music = parse_music_answers(answers, f"r{ring}")

    agent_decisions: dict[int, dict[str, Any]] = {}
    for a in active_agents:
        act_ans = answers.get(f"a{a.id}_action")
        push_ans = answers.get(f"a{a.id}_push")
        if act_ans is None:
            continue
        chosen = str(getattr(act_ans, "choice", a.action))
        probs = {k: round(float(v), 4) for k, v in (getattr(act_ans, "probabilities", {}) or {}).items()}
        raw_score = float(getattr(push_ans, "score", 1.0)) if push_ans is not None else 1.0
        mapped_push = map_push_score(raw_score, chosen, remap=remap_pushiness)
        agent_decisions[a.id] = {
            "action": chosen,
            "action_probs": probs,
            "raw_push_score": round(raw_score, 4),
            "pushiness": mapped_push,
        }

    resolved_asks: list[dict[str, Any]] = []
    for qkey, ask in ask_key_map.items():
        ans_obj = answers.get(qkey)
        if ans_obj is None:
            continue
        p_yes = float(getattr(ans_obj, "noul", 0.5))
        ans_str = "yes" if p_yes >= 0.5 else "no"
        resolved_asks.append(
            {
                "ask": ask,
                "id": ask.id,
                "from": ask.asker_id,
                "to": ask.blocker_id,
                "answer": ans_str,
                "p": round(p_yes, 4),
            }
        )

    return RingDecisionOutput(
        ring=ring,
        ok=True,
        music=music,
        agent_decisions=agent_decisions,
        ask_answers=resolved_asks,
        api=res,
    )


async def run_all_rings_parallel(
    *,
    client: GeminiClient,
    crowd: CrowdState,
    audio: RollingAudioRecorder,
    wom: WordOfMouth,
    pending_asks: list[PendingAsk],
    remap_pushiness: bool = PUSHINESS_REMAP,
) -> dict[int, RingDecisionOutput]:
    """Build ring clips and execute all non-empty ring Decisions calls concurrently."""
    buckets = bucket_agents_by_ring(crowd.agents)
    neighbors_map = nearest_neighbors_text(crowd.agents, k=3)
    kick_rate = crowd.kick_rate()

    # Map each pending ask to the blocker's current ring
    agent_ring = {a.id: ring_for_agent(a) for a in crowd.agents}
    asks_by_ring: dict[int, list[PendingAsk]] = {r: [] for r in range(1, 6)}
    for ask in pending_asks:
        br = agent_ring.get(ask.blocker_id)
        if br is not None:
            asks_by_ring[br].append(ask)

    tasks = []
    active_rings: list[int] = []
    for r in range(1, 6):
        ring_agents = buckets[r]
        if not ring_agents:
            continue
        active_rings.append(r)
        b64_wav = audio.ring_clip_b64(ring=r, seconds=RING_CLIP_SECONDS)
        feat_txt = format_features_text(crowd.t, crowd.dsp, kick_rate, coarse=(r >= 4))
        wom_str = wom.format_word_for_ring(r)
        tasks.append(
            run_ring_decisions(
                client=client,
                ring=r,
                agents=ring_agents,
                b64_wav=b64_wav,
                features_text=feat_txt,
                neighbors_map=neighbors_map,
                wom_str=wom_str,
                pending_asks_for_ring=asks_by_ring[r],
                crowd=crowd,
                remap_pushiness=remap_pushiness,
            )
        )

    if not tasks:
        return {}

    outputs = await asyncio.gather(*tasks)
    return {out.ring: out for out in outputs}
