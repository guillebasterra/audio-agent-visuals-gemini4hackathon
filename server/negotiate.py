"""Contest detection and Gemini 4 Hydrogen natural-language asks (plan.md section 8.3).

- Uses exact plan Ask schema: `text: str`, `claimed_keenness: float` (no extra fields).
- System prompt says the agent wants to get closer to the DJ — does NOT instruct it
  to lie or to be honest.
- Asks run asynchronously; completed asks are queued into the next tick's blocker ring
  Decisions call (`a{blocker}_ans_{asker}`) so answers land cleanly on the next tick.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from server.config import AGENT_RADIUS, MAX_ASKS_FALLBACK, MAX_ASKS_PER_TICK, MODEL_HYDROGEN
from server.gemini_client import GeminiClient
from server.state import Agent, CrowdState, dj_closest_point, energy_word, ring_for_agent

logger = logging.getLogger("agent_crowd.negotiate")

CONTEST_DIST_MULT: float = 2.2
FORWARD_CONE_COS: float = math.cos(math.radians(35.0))

HYDROGEN_SYSTEM_PROMPT: str = (
    "You are a person on a packed house-music dancefloor who wants to move closer to the DJ. "
    "React to the music moment, your own energy, and the person blocking you in a natural, varied voice "
    "(one short spoken line, <= 20 words — avoid generic clichés), and state how keen you claim to be (0.0 to 1.0)."
)


class Ask(BaseModel):
    """Exact plan.md section 8.3 schema — no extra fields."""

    text: str = Field(description="One sentence, <= 20 words, in the asker's voice.")
    claimed_keenness: float = Field(ge=0.0, le=1.0, description="0..1, how much the asker says it cares.")


@dataclass
class Contest:
    asker: Agent
    blocker: Agent
    score: float


@dataclass
class PendingAsk:
    id: str
    tick_created: int
    t_created: float
    asker_id: int
    blocker_id: int
    asker_archetype: str
    blocker_archetype: str
    true_k: float
    text: str
    claimed_keenness: float
    prompt_summary: str
    asker_yielded_before: int
    blocker_yielded_before: int
    hydrogen_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    announced: bool = False


def _apparent_energy_bucket(E: float) -> str:
    if E < 0.33:
        return "low"
    if E < 0.67:
        return "med"
    return "high"


def detect_contests(
    crowd: CrowdState,
    *,
    max_asks: int = MAX_ASKS_PER_TICK,
    busy_pairs: set[tuple[int, int]] | None = None,
) -> list[Contest]:
    """Find blocked pushing agents with a non-yielding neighbor within contact reach in ±35° toward DJ segment."""
    agents = crowd.agents
    base_max_dist = CONTEST_DIST_MULT * AGENT_RADIUS
    busy = busy_pairs or set()

    candidates: list[Contest] = []
    for asker in agents:
        if asker.exhausted or asker.action != "push":
            continue
        dj_cx, dj_cy = dj_closest_point(asker.x, asker.y)
        to_dj_x = dj_cx - asker.x
        to_dj_y = dj_cy - asker.y
        dj_dist = math.hypot(to_dj_x, to_dj_y)
        if dj_dist < 1e-6:
            continue
        ux, uy = to_dj_x / dj_dist, to_dj_y / dj_dist

        best_blocker: Agent | None = None
        best_score = -1.0
        for other in agents:
            if other.id == asker.id or other.action == "yield" or other.exhausted:
                continue
            if (asker.id, other.id) in busy:
                continue
            pair_max = max(base_max_dist, 1.15 * (asker.radius + other.radius))
            dx = other.x - asker.x
            dy = other.y - asker.y
            d2 = dx * dx + dy * dy
            if d2 > pair_max * pair_max or d2 < 1e-10:
                continue
            d = math.sqrt(d2)
            cos_ang = (dx * ux + dy * uy) / d
            if cos_ang < FORWARD_CONE_COS:
                continue
            score = asker.pushiness * (other.v - asker.v + 0.1)
            if score > best_score:
                best_score = score
                best_blocker = other

        if best_blocker is not None:
            candidates.append(Contest(asker=asker, blocker=best_blocker, score=best_score))

    candidates.sort(key=lambda c: c.score, reverse=True)

    # Keep at most one ask per blocker and one per asker per tick
    selected: list[Contest] = []
    used_askers: set[int] = set()
    used_blockers: set[int] = set()
    for c in candidates:
        if c.asker.id in used_askers or c.blocker.id in used_blockers:
            continue
        used_askers.add(c.asker.id)
        used_blockers.add(c.blocker.id)
        selected.append(c)
        if len(selected) >= max_asks:
            break
    return selected


def build_ask_prompt(
    asker: Agent,
    blocker: Agent,
    crowd: CrowdState,
    ring_heard: dict[str, Any] | None,
) -> tuple[str, str]:
    """Build Hydrogen ask prompt + short log summary (plain-word energy, never revealing hidden info)."""
    r_asker = ring_for_agent(asker)
    r_blocker = ring_for_agent(blocker)
    heard_sec = (ring_heard or {}).get("section", "groove")
    heard_eta = (ring_heard or {}).get("drop_eta", "not_soon")
    heard_rise = float((ring_heard or {}).get("rising", 0.5))

    y_ab = crowd.yield_count(asker.id, blocker.id)
    y_ba = crowd.yield_count(blocker.id, asker.id)
    a_energy = energy_word(asker.E)
    b_energy = energy_word(blocker.E)

    prompt = (
        f"You are A{asker.id} in ring {r_asker} of 5 from the DJ (1=front). "
        f"Your traits: keen={asker.k:.2f}, space={asker.s:.2f}. "
        f"Your state: energy={a_energy}, happiness={asker.H:.2f}. "
        f"Music you hear: section={heard_sec}, drop_eta={heard_eta}, rising={heard_rise:.2f}.\n"
        f"Directly ahead of you is A{blocker.id} (ring {r_blocker}, apparent energy {b_energy}, action={blocker.action}).\n"
        f"History: you have yielded to A{blocker.id} {y_ab} times; A{blocker.id} has yielded to you {y_ba} times.\n"
        f"Ask A{blocker.id} to let you pass in <= 20 words, and give your claimed_keenness (0..1)."
    )
    summary = (
        f"A{asker.id}(k={asker.k:.2f},E={a_energy},r{r_asker},heard={heard_sec}/{heard_eta})->"
        f"A{blocker.id}(r{r_blocker},E={b_energy},{blocker.action},y={y_ab}/{y_ba})"
    )
    return prompt, summary


class NegotiationManager:
    """Manages async Hydrogen ask tasks and queues completed asks for next tick's answers."""

    def __init__(self, client: GeminiClient) -> None:
        self._client = client
        self._seq: int = 0
        self._ready_for_answer: list[PendingAsk] = []
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._inflight_pairs: set[tuple[int, int]] = set()
        self._latencies_ms: list[float] = []
        self.failures: int = 0
        self.rate_limits: int = 0
        self.tokens_total: int = 0
        self.input_tokens_total: int = 0
        self.output_tokens_total: int = 0
        self.thought_tokens_total: int = 0

    def reset(self) -> None:
        for task in list(self._inflight_tasks):
            if not task.done():
                task.cancel()
        self._inflight_tasks.clear()
        self._inflight_pairs.clear()
        self._ready_for_answer.clear()
        self._latencies_ms.clear()
        self._seq = 0
        self.failures = 0
        self.rate_limits = 0
        self.tokens_total = 0
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.thought_tokens_total = 0

    def _effective_max_asks(self) -> int:
        # Plan.md section 14 fallback: if Hydrogen p95 > 6 s, cap at 4 asks
        if len(self._latencies_ms) >= 6:
            recent = sorted(self._latencies_ms[-20:])
            p95 = recent[min(len(recent) - 1, int(0.95 * len(recent)))]
            if p95 > 6000.0:
                return MAX_ASKS_FALLBACK
        return MAX_ASKS_PER_TICK

    def drain_ready_asks(self) -> list[PendingAsk]:
        """Pop all completed asks ready to be answered in the upcoming ring Decisions calls."""
        ready = list(self._ready_for_answer)
        self._ready_for_answer.clear()
        return ready

    def consume_metrics_delta(self) -> dict[str, Any]:
        recent_ms = list(self._latencies_ms[-16:])
        return {
            "hydrogen_p50_ms": round(sorted(recent_ms)[len(recent_ms) // 2], 1) if recent_ms else 0.0,
            "hydrogen_calls": len(self._latencies_ms),
            "hydrogen_failures": self.failures,
            "hydrogen_rate_limits": self.rate_limits,
        }

    def schedule_contests(
        self,
        crowd: CrowdState,
        ring_heard: dict[int, dict[str, Any]],
        tick_num: int,
    ) -> int:
        """Detect contests and fire async Hydrogen ask tasks (non-blocking)."""
        if crowd.condition != "talk":
            return 0

        busy = set(self._inflight_pairs)
        for p in self._ready_for_answer:
            busy.add((p.asker_id, p.blocker_id))

        contests = detect_contests(
            crowd,
            max_asks=self._effective_max_asks(),
            busy_pairs=busy,
        )
        launched = 0
        for c in contests:
            self._seq += 1
            qid = f"q{self._seq}"
            r_asker = ring_for_agent(c.asker)
            prompt, summary = build_ask_prompt(c.asker, c.blocker, crowd, ring_heard.get(r_asker))
            pair = (c.asker.id, c.blocker.id)
            self._inflight_pairs.add(pair)

            task = asyncio.create_task(
                self._run_single_ask(
                    qid=qid,
                    tick_created=tick_num,
                    t_created=crowd.t,
                    asker=c.asker,
                    blocker=c.blocker,
                    prompt=prompt,
                    summary=summary,
                    y_ab=crowd.yield_count(c.asker.id, c.blocker.id),
                    y_ba=crowd.yield_count(c.blocker.id, c.asker.id),
                ),
                name=f"hydrogen_ask_{qid}",
            )
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)
            launched += 1
        return launched

    async def _run_single_ask(
        self,
        *,
        qid: str,
        tick_created: int,
        t_created: float,
        asker: Agent,
        blocker: Agent,
        prompt: str,
        summary: str,
        y_ab: int,
        y_ba: int,
    ) -> None:
        pair = (asker.id, blocker.id)
        try:
            res = await self._client.interact_structured(
                model=MODEL_HYDROGEN,
                system_instruction=HYDROGEN_SYSTEM_PROMPT,
                input_blocks=[{"type": "text", "text": prompt}],
                schema_model=Ask,
                thinking_level="low",
                store=False,
                label=f"ask_{qid}",
            )
            if res.rate_limited:
                self.rate_limits += 1
            if not res.ok or res.value is None:
                self.failures += 1
                return

            ask_obj: Ask = res.value
            clean_text = " ".join(ask_obj.text.strip().split())
            words = clean_text.split()
            if len(words) > 22:
                clean_text = " ".join(words[:20]) + "…"
            claimed = max(0.0, min(1.0, float(ask_obj.claimed_keenness)))

            self._latencies_ms.append(res.latency_ms)
            self.tokens_total += res.total_tokens
            self.input_tokens_total += res.input_tokens
            self.output_tokens_total += res.output_tokens
            self.thought_tokens_total += res.thought_tokens

            self._ready_for_answer.append(
                PendingAsk(
                    id=qid,
                    tick_created=tick_created,
                    t_created=t_created,
                    asker_id=asker.id,
                    blocker_id=blocker.id,
                    asker_archetype=asker.archetype,
                    blocker_archetype=blocker.archetype,
                    true_k=asker.k,
                    text=clean_text,
                    claimed_keenness=claimed,
                    prompt_summary=summary,
                    asker_yielded_before=y_ab,
                    blocker_yielded_before=y_ba,
                    hydrogen_latency_ms=res.latency_ms,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                    thought_tokens=res.thought_tokens,
                )
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.failures += 1
            logger.warning("Hydrogen ask %s unexpected error: %s", qid, exc)
        finally:
            self._inflight_pairs.discard(pair)
