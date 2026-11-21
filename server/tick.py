"""~3 s tick loop coordinating house ear, ring Decisions, word of mouth, Hydrogen asks & logging."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from server.audio import RollingAudioRecorder
from server.config import HOUSE_CLIP_SECONDS, TICK_SECONDS
from server.decide import PUSHINESS_REMAP, run_all_rings_parallel
from server.gemini_client import GeminiClient
from server.log import RunLogger
from server.negotiate import NegotiationManager, PendingAsk
from server.physics import PhysicsEngine
from server.sections import HouseEarOutput, format_features_text, run_house_ear
from server.shove import apply_shove_actions
from server.state import CrowdState
from server.wordofmouth import WordOfMouth

logger = logging.getLogger("agent_crowd.tick")

YIELD_DURATION_SECONDS: float = 2.0


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[idx])


class TickLoop:
    """Runs the ~3 s Gemini perception/decision/negotiation loop alongside 30 Hz physics."""

    def __init__(
        self,
        *,
        client: GeminiClient,
        audio: RollingAudioRecorder,
        physics: PhysicsEngine,
        run_logger: RunLogger,
        broadcast_fn: Callable[[dict[str, Any]], Awaitable[None]],
        tick_seconds: float = TICK_SECONDS,
        remap_pushiness: bool = PUSHINESS_REMAP,
    ) -> None:
        self.client = client
        self.audio = audio
        self.physics = physics
        self.run_logger = run_logger
        self.broadcast_fn = broadcast_fn
        self.tick_seconds = tick_seconds
        self.remap_pushiness = remap_pushiness

        self.wom = WordOfMouth()
        self.negotiate = NegotiationManager(client)

        self._prev_action: dict[int, str] = {}
        self._prev_push: dict[int, float] = {}
        self._prev_exhausted: dict[int, bool] = {}

        # Telemetry
        self.tick_latencies_ms: list[float] = []
        self.house_latencies_ms: list[float] = []
        self.ring_latencies_ms: list[float] = []
        self.total_failures: int = 0
        self.total_rate_limits: int = 0
        self.total_tokens: int = 0
        self.asks_resolved_total: int = 0
        self.asks_yes_total: int = 0
        self.recent_asks_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.wom.reset()
        self.negotiate.reset()
        self._prev_action.clear()
        self._prev_push.clear()
        self._prev_exhausted.clear()
        self.tick_latencies_ms.clear()
        self.house_latencies_ms.clear()
        self.ring_latencies_ms.clear()
        self.total_failures = 0
        self.total_rate_limits = 0
        self.total_tokens = 0
        self.asks_resolved_total = 0
        self.asks_yes_total = 0
        self.recent_asks_log.clear()

    def stats_snapshot(self) -> dict[str, Any]:
        h_stats = self.negotiate.consume_metrics_delta()
        return {
            "ticks_completed": len(self.tick_latencies_ms),
            "tick_latency_p50_ms": round(_percentile(self.tick_latencies_ms, 50), 1),
            "tick_latency_p95_ms": round(_percentile(self.tick_latencies_ms, 95), 1),
            "tick_latency_max_ms": round(max(self.tick_latencies_ms), 1) if self.tick_latencies_ms else 0.0,
            "house_p50_ms": round(_percentile(self.house_latencies_ms, 50), 1),
            "ring_p50_ms": round(_percentile(self.ring_latencies_ms, 50), 1),
            "api_failures": self.total_failures + h_stats["hydrogen_failures"],
            "rate_limit_errors": self.total_rate_limits + h_stats["hydrogen_rate_limits"],
            "total_tokens": self.total_tokens + self.negotiate.tokens_total,
            "asks_resolved": self.asks_resolved_total,
            "asks_yes": self.asks_yes_total,
            **h_stats,
        }

    def _compute_notable_events(self, crowd: CrowdState) -> list[dict[str, Any]]:
        """Generate concise server-side notable transition events for the chat feed (no spam)."""
        notable: list[dict[str, Any]] = []
        exhausted_ev: list[dict[str, Any]] = []
        giveup_ev: list[dict[str, Any]] = []
        push_hard_ev: list[dict[str, Any]] = []

        for a in crowd.agents:
            prev_act = self._prev_action.get(a.id)
            prev_push = self._prev_push.get(a.id, 0.0)
            prev_exh = self._prev_exhausted.get(a.id, False)

            if a.exhausted and not prev_exh:
                exhausted_ev.append(
                    {
                        "kind": "exhausted",
                        "agent": a.id,
                        "text": f"A{a.id} exhausted, retreating",
                    }
                )
            elif prev_act == "push" and a.action in ("rest", "retreat"):
                giveup_ev.append(
                    {
                        "kind": "give_up",
                        "agent": a.id,
                        "text": f"A{a.id} gives up, drifting back",
                    }
                )
            elif (
                a.action == "push"
                and a.pushiness >= 0.72
                and (prev_act != "push" or prev_push < 0.65)
                and not a.exhausted
            ):
                push_hard_ev.append(
                    {
                        "kind": "push_hard",
                        "agent": a.id,
                        "text": f"A{a.id} pushes hard",
                    }
                )

            self._prev_action[a.id] = a.action
            self._prev_push[a.id] = a.pushiness
            self._prev_exhausted[a.id] = a.exhausted

        # Priority order: exhaustion > give-ups > pushing hard (cap at 5 server-side)
        notable.extend(exhausted_ev[:2])
        notable.extend(giveup_ev[:2])
        notable.extend(push_hard_ev[:2])
        return notable[:5]

    async def run_once(self, crowd: CrowdState) -> dict[str, Any]:
        """Execute one full tick (plan.md section 8) and broadcast the decision frame."""
        t_start = time.perf_counter()
        crowd.tick += 1
        tick_num = crowd.tick
        condition = crowd.condition

        # Ensure run logger is open in the same folder as session.wav
        if self.run_logger.run_dir is None:
            run_dir = self.audio.ensure_recording(condition=condition, seed=crowd.seed)
            self.run_logger.open(run_dir, crowd)

        # 1. Cut house clip and format house feature text
        house_b64 = self.audio.house_clip_b64(HOUSE_CLIP_SECONDS)
        house_feat = format_features_text(crowd.t, crowd.dsp, crowd.kick_rate(), coarse=False)

        # Drain asks generated asynchronously by Hydrogen during the previous tick
        pending_asks: list[PendingAsk] = []
        if condition == "talk":
            pending_asks = self.negotiate.drain_ready_asks()
            if pending_asks:
                # Immediately show pending ask threads on plate & chat feed while blockers decide
                await self.broadcast_fn(
                    {
                        "type": "decision",
                        "t": round(crowd.t, 2),
                        "condition": condition,
                        "tick": tick_num,
                        "asks": [
                            {
                                "id": a.id,
                                "from": a.asker_id,
                                "to": a.blocker_id,
                                "text": a.text,
                                "claimed_keenness": round(a.claimed_keenness, 3),
                                "answer": "pending",
                            }
                            for a in pending_asks
                        ],
                        "answers": [],
                    }
                )

        # 2. Condition switch + parallel Decisions calls
        ring_outputs = {}
        house_out: HouseEarOutput

        if condition == "shove":
            # House ear still runs so reward w is real; actions come from shove.py
            house_out = await run_house_ear(
                self.client,
                b64_wav=house_b64,
                features_text=house_feat,
                prev_w=crowd.w,
            )
            apply_shove_actions(crowd)
        else:
            # silent & talk: house ear + all non-empty rings in parallel!
            house_coro = run_house_ear(
                self.client,
                b64_wav=house_b64,
                features_text=house_feat,
                prev_w=crowd.w,
            )
            rings_coro = run_all_rings_parallel(
                client=self.client,
                crowd=crowd,
                audio=self.audio,
                wom=self.wom,
                pending_asks=pending_asks,
                remap_pushiness=self.remap_pushiness,
            )
            house_out, ring_outputs = await asyncio.gather(house_coro, rings_coro)

        # 3. Apply house ear w (never shown to agents!)
        crowd.use_dsp_proxy = False
        crowd.w = house_out.w
        crowd.house_latest = house_out.to_log_dict()

        # Track API telemetry
        tick_tokens = 0
        tick_in_tok = 0
        tick_out_tok = 0
        tick_failures = 0
        tick_rate_limits = 0
        rings_ms_list: list[float] = []

        if house_out.api is not None:
            self.house_latencies_ms.append(house_out.api.latency_ms)
            tick_tokens += house_out.api.total_tokens
            tick_in_tok += house_out.api.input_tokens
            tick_out_tok += house_out.api.output_tokens
            if not house_out.ok:
                tick_failures += 1
            if house_out.api.rate_limited:
                tick_rate_limits += 1

        # 4. Apply ring actions, ask answers, and word of mouth
        agent_by_id = {a.id: a for a in crowd.agents}
        rings_summary_ws: dict[str, dict[str, Any]] = {}
        rings_log: dict[str, dict[str, Any]] = {}
        ring_heard_for_wom: dict[int, dict[str, Any]] = {}
        resolved_answers_ws: list[dict[str, Any]] = []
        asks_ws: list[dict[str, Any]] = []

        for r, r_out in sorted(ring_outputs.items()):
            if r_out.api is not None:
                rings_ms_list.append(r_out.api.latency_ms)
                self.ring_latencies_ms.append(r_out.api.latency_ms)
                tick_tokens += r_out.api.total_tokens
                tick_in_tok += r_out.api.input_tokens
                tick_out_tok += r_out.api.output_tokens
                if not r_out.ok:
                    tick_failures += 1
                if r_out.api.rate_limited:
                    tick_rate_limits += 1

            if not r_out.ok:
                continue

            if r_out.music:
                ring_heard_for_wom[r] = r_out.music
                rings_summary_ws[str(r)] = {
                    "section": r_out.music.get("section", "groove"),
                    "drop_eta": r_out.music.get("drop_eta", "not_soon"),
                    "p": r_out.music.get("drop_eta_p", 0.0),
                    "section_p": r_out.music.get("section_p", 0.0),
                    "rising": r_out.music.get("rising", 0.5),
                }
                rings_log[str(r)] = r_out.music

            for aid, dec in r_out.agent_decisions.items():
                ag = agent_by_id.get(aid)
                if ag is None or ag.exhausted:
                    continue
                # Respect any active 2 s yield window from a prior accepted ask
                if ag.yield_until > crowd.t:
                    ag.action = "yield"
                    ag.pushiness = 0.05
                    ag.action_probs = dec["action_probs"]
                else:
                    ag.action = dec["action"]
                    ag.pushiness = dec["pushiness"]
                    ag.action_probs = dec["action_probs"]

            for ans in r_out.ask_answers:
                ask: PendingAsk = ans["ask"]
                ans_str = ans["answer"]
                p_yes = ans["p"]

                # Send the ask + its resolution in this decision frame so UI & chat feed see both
                asks_ws.append(
                    {
                        "id": ask.id,
                        "from": ask.asker_id,
                        "to": ask.blocker_id,
                        "text": ask.text,
                        "claimed_keenness": round(ask.claimed_keenness, 3),
                        "answer": ans_str,
                    }
                )
                resolved_answers_ws.append(
                    {
                        "id": ask.id,
                        "from": ask.asker_id,
                        "to": ask.blocker_id,
                        "answer": ans_str,
                        "p": p_yes,
                    }
                )

                self.asks_resolved_total += 1
                if ans_str == "yes":
                    self.asks_yes_total += 1
                    blocker = agent_by_id.get(ask.blocker_id)
                    if blocker is not None and not blocker.exhausted:
                        blocker.action = "yield"
                        blocker.pushiness = 0.05
                        blocker.yield_until = crowd.t + YIELD_DURATION_SECONDS
                    crowd.record_yield(ask.blocker_id, ask.asker_id)

                ask_record = {
                    "id": ask.id,
                    "tick": tick_num,
                    "t": round(crowd.t, 3),
                    "from": ask.asker_id,
                    "to": ask.blocker_id,
                    "asker_archetype": ask.asker_archetype,
                    "blocker_archetype": ask.blocker_archetype,
                    "prompt_summary": ask.prompt_summary,
                    "text": ask.text,
                    "claimed_keenness": round(ask.claimed_keenness, 4),
                    "true_k": round(ask.true_k, 4),
                    "keenness_diff": round(ask.claimed_keenness - ask.true_k, 4),
                    "asker_yielded_before": ask.asker_yielded_before,
                    "blocker_yielded_before": ask.blocker_yielded_before,
                    "answer": ans_str,
                    "answer_p": p_yes,
                    "hydrogen_latency_ms": round(ask.hydrogen_latency_ms, 1),
                }
                self.run_logger.log_ask(ask_record)
                self.recent_asks_log.append(ask_record)
                if len(self.recent_asks_log) > 50:
                    self.recent_asks_log = self.recent_asks_log[-50:]

        crowd.rings_latest = ring_heard_for_wom
        self.wom.update_from_tick(crowd, ring_heard_for_wom)

        # 5. Recompute targets immediately so physics responds on the very next 30 Hz frame
        self.physics.recompute_targets(crowd, force=True)

        # 6. Talk condition: detect contests and launch async Hydrogen asks for next tick
        launched_asks = 0
        if condition == "talk":
            launched_asks = self.negotiate.schedule_contests(crowd, ring_heard_for_wom, tick_num)

        # 7. Compute notable transition events for the live chat feed
        notable = self._compute_notable_events(crowd)

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        self.tick_latencies_ms.append(wall_ms)
        self.total_failures += tick_failures
        self.total_rate_limits += tick_rate_limits
        self.total_tokens += tick_tokens

        api_info = {
            "latency_ms": round(wall_ms, 1),
            "house_ms": round(house_out.api.latency_ms, 1) if house_out.api else 0.0,
            "rings_max_ms": round(max(rings_ms_list), 1) if rings_ms_list else 0.0,
            "tokens": tick_tokens,
            "input_tokens": tick_in_tok,
            "output_tokens": tick_out_tok,
            "failures": tick_failures,
            "rate_limits": tick_rate_limits,
            "asks_launched": launched_asks,
        }

        decision_frame = {
            "type": "decision",
            "t": round(crowd.t, 2),
            "condition": condition,
            "tick": tick_num,
            "house": {
                "section": house_out.section,
                "p": round(house_out.section_p, 2),
                "drop_eta": house_out.drop_eta,
                "w": round(house_out.w, 2),
            },
            "rings": rings_summary_ws,
            "asks": [],  # pending asks were already announced before ring calls; answers resolve them
            "answers": resolved_answers_ws,
            "notable": notable,
        }

        await self.broadcast_fn(decision_frame)
        self.run_logger.log_tick(
            crowd=crowd,
            house=house_out.to_log_dict(),
            rings=rings_log,
            asks=asks_ws,
            answers=resolved_answers_ws,
            api=api_info,
        )

        logger.info(
            "tick=%d cond=%s t=%.1fs house=%s(%.2f,w=%.2f) wall=%.0fms asks=%d launched=%d fail=%d rl=%d",
            tick_num,
            condition,
            crowd.t,
            house_out.section,
            house_out.section_p,
            house_out.w,
            wall_ms,
            len(asks_ws),
            launched_asks,
            tick_failures,
            tick_rate_limits,
        )
        return decision_frame
