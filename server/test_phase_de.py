"""Phase D & E unit and mocked end-to-end tests (offline, deterministic)."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Type

import numpy as np

from server.audio import RollingAudioRecorder
from server.config import AGENT_RADIUS, SECTION_WEIGHTS
from server.decide import map_push_score, run_all_rings_parallel
from server.gemini_client import ApiResult
from server.log import RunLogger
from server.negotiate import Ask, PendingAsk, build_ask_prompt, detect_contests
from server.physics import PhysicsEngine
from server.sections import expected_section_weight, format_features_text, parse_music_answers
from server.state import CrowdState, init_crowd, ring_for_agent
from server.tick import TickLoop
from server.wordofmouth import WordOfMouth, nearest_neighbors_text


def _make_fake_pcm(seconds: float = 16.0, sr: int = 16000) -> bytes:
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    sig = 0.25 * np.sin(2.0 * np.pi * 110.0 * t)
    return np.clip(sig * 32767.0, -32768, 32767).astype("<i2").tobytes()


class MockGeminiClient:
    """Deterministic mock GeminiClient recording all Decisions and Hydrogen payloads."""

    def __init__(self, *, fail_ring: int | None = None, answer_yes: bool = True) -> None:
        self.decide_calls: list[dict[str, Any]] = []
        self.hydrogen_calls: list[dict[str, Any]] = []
        self.fail_ring = fail_ring
        self.answer_yes = answer_yes

    async def decide(
        self,
        *,
        input_blocks: list[dict[str, Any]],
        questions: Mapping[str, dict[str, Any]],
        label: str = "decisions",
        **kwargs: Any,
    ) -> ApiResult[Any]:
        self.decide_calls.append(
            {"label": label, "input_blocks": input_blocks, "questions": dict(questions)}
        )
        if self.fail_ring is not None and label == f"ring_{self.fail_ring}":
            return ApiResult(ok=False, latency_ms=12.0, error="Simulated failure", rate_limited=False)

        answers: dict[str, Any] = {}
        for qk, qspec in questions.items():
            qtype = qspec["type"]
            if qk.endswith("_section"):
                answers[qk] = SimpleNamespace(
                    choice="drop",
                    confidence=0.84,
                    probabilities={"drop": 0.80, "build": 0.15, "groove": 0.05},
                )
            elif qk.endswith("_drop_eta"):
                answers[qk] = SimpleNamespace(
                    choice="now",
                    confidence=0.82,
                    probabilities={"now": 0.82, "next_8_bars": 0.12, "within_32_bars": 0.04, "not_soon": 0.02},
                )
            elif qk.endswith("_rising"):
                answers[qk] = SimpleNamespace(noul=0.78, confidence=0.80)
            elif qk.endswith("_action"):
                # Agents with high keenness push; dancers dance; low energy rest
                inst = qspec["instructions"]
                if "energy 0.2" in inst or "energy 0.1" in inst:
                    act = "rest"
                elif "keen 0.9" in inst or "keen 0.8" in inst or "keen 1.0" in inst:
                    act = "push"
                else:
                    act = "hold"
                answers[qk] = SimpleNamespace(
                    choice=act,
                    confidence=0.75,
                    probabilities={"push": 0.7 if act == "push" else 0.2, "hold": 0.2, "rest": 0.1},
                )
            elif qk.endswith("_push"):
                inst = qspec["instructions"]
                sc = 1.9 if ("keen 0.9" in inst or "keen 1.0" in inst) else 0.8
                answers[qk] = SimpleNamespace(score=sc, confidence=0.77, probabilities={"0": 0.1, "1": 0.3, "2": 0.6})
            elif "_ans_" in qk:
                p = 0.78 if self.answer_yes else 0.22
                answers[qk] = SimpleNamespace(noul=p, confidence=0.81)
            elif qtype == "noul":
                answers[qk] = SimpleNamespace(noul=0.5, confidence=0.5)

        return ApiResult(
            ok=True,
            value=SimpleNamespace(answers=answers),
            latency_ms=42.0,
            input_tokens=300,
            output_tokens=0,
            total_tokens=300,
        )

    async def interact_structured(
        self,
        *,
        input_blocks: list[dict[str, Any]],
        schema_model: Type[Ask],
        system_instruction: Optional[str] = None,
        thinking_level: str = "low",
        store: bool = False,
        label: str = "hydrogen",
        **kwargs: Any,
    ) -> ApiResult[Ask]:
        self.hydrogen_calls.append(
            {
                "label": label,
                "system_instruction": system_instruction,
                "input_blocks": input_blocks,
                "schema_fields": set(schema_model.model_fields.keys()),
                "thinking_level": thinking_level,
                "store": store,
            }
        )
        ask = schema_model(text="Hey, this drop is my favorite, mind if I slip ahead?", claimed_keenness=0.92)
        return ApiResult(
            ok=True,
            value=ask,
            latency_ms=65.0,
            input_tokens=150,
            output_tokens=25,
            thought_tokens=20,
            total_tokens=195,
        )


def test_sections_and_wom() -> None:
    print("\n=== 1. SECTIONS, COARSE RING FEATURES & WORD OF MOUTH ===")
    probs = {"drop": 0.5, "build": 0.5}
    w = expected_section_weight(probs)
    expected = 0.5 * SECTION_WEIGHTS["drop"] + 0.5 * SECTION_WEIGHTS["build"]
    assert abs(w - expected) < 1e-6, f"Bad expected weight: {w} != {expected}"

    dsp = {"energy": 0.743, "trend": 0.184, "bass_gap": 12.2}
    fine = format_features_text(65.0, dsp, 2.14, coarse=False)
    coarse = format_features_text(65.0, dsp, 2.14, coarse=True)
    assert "energy=0.74" in fine and "trend=+0.18" in fine, fine
    assert "energy=0.7 " in coarse and "trend=+0.2 " in coarse, coarse

    parsed = parse_music_answers(
        {
            "r2_section": SimpleNamespace(choice="build", confidence=0.61, probabilities={"build": 0.71, "drop": 0.29}),
            "r2_drop_eta": SimpleNamespace(choice="next_8_bars", confidence=0.55, probabilities={"next_8_bars": 0.74}),
            "r2_rising": SimpleNamespace(noul=0.83),
        },
        "r2",
    )
    assert parsed["section"] == "build" and abs(parsed["section_p"] - 0.71) < 1e-4
    assert parsed["drop_eta"] == "next_8_bars" and abs(parsed["drop_eta_p"] - 0.74) < 1e-4
    assert abs(parsed["rising"] - 0.83) < 1e-4

    crowd = init_crowd(seed=42)
    wom = WordOfMouth()
    assert wom.format_word_for_ring(1) == ""
    assert wom.format_word_for_ring(2) == ""
    wom.update_from_tick(
        crowd,
        {
            1: {"section": "drop", "drop_eta": "now", "drop_eta_p": 0.85, "rising": 0.9},
            2: {"section": "build", "drop_eta": "next_8_bars", "drop_eta_p": 0.72, "rising": 0.7},
        },
    )
    assert wom.format_word_for_ring(1) == ""
    w2 = wom.format_word_for_ring(2)
    w3 = wom.format_word_for_ring(3)
    assert "ring 1" in w2.lower() and "drop now (0.85)" in w2, w2
    assert "ring 2" in w3.lower() and "drop next_8_bars (0.72)" in w3, w3

    nb = nearest_neighbors_text(crowd.agents, k=3)
    assert len(nb) == 100
    assert nb[0].count(",") == 2, f"Expected 3 comma-separated neighbors, got: {nb[0]}"
    print("PASS: expected w, coarse ring 4-5 features, 1-tick ring-to-ring WoM, and 3-neighbor summaries verified")


def test_ask_schema_and_contests() -> None:
    print("\n=== 2. EXACT ASK SCHEMA & FORWARD-CONE CONTEST DETECTION ===")
    assert set(Ask.model_fields.keys()) == {"text", "claimed_keenness"}, (
        f"Ask schema must have exact fields text & claimed_keenness, got {Ask.model_fields.keys()}"
    )

    crowd = init_crowd(seed=42)
    # Place A10 pushing directly behind A20 toward DJ (0.5, 0.0)
    for a in crowd.agents:
        a.x, a.y = 0.9, 0.9
        a.action = "hold"
        a.pushiness = 0.2
    asker = crowd.agents[10]
    blocker = crowd.agents[20]
    behind = crowd.agents[30]
    asker.x, asker.y, asker.action, asker.pushiness, asker.v = 0.50, 0.40, "push", 0.9, 0.64
    blocker.x, blocker.y, blocker.action, blocker.v = 0.50, 0.40 - 1.5 * AGENT_RADIUS, "hold", 0.70
    # Agent behind asker (should NOT be detected as blocker)
    behind.x, behind.y, behind.action, behind.v = 0.50, 0.40 + 1.5 * AGENT_RADIUS, "hold", 0.58

    contests = detect_contests(crowd, max_asks=8)
    assert len(contests) == 1 and contests[0].asker.id == 10 and contests[0].blocker.id == 20, contests

    # Yielding blocker should be excluded
    blocker.action = "yield"
    assert detect_contests(crowd, max_asks=8) == []
    blocker.action = "hold"

    # Prompt must NOT leak blocker's true k/s
    prompt, summary = build_ask_prompt(asker, blocker, crowd, {"section": "build", "drop_eta": "next_8_bars", "rising": 0.8})
    assert "apparent energy" in prompt
    assert f"keen={blocker.k:.2f}" not in prompt or abs(blocker.k - asker.k) < 0.01
    assert "A10" in summary and "A20" in summary
    print("PASS: Ask schema exact match, ±35° forward-cone contest detection, and info-hiding prompt verified")


def test_decide_questions_and_push_remap() -> None:
    print("\n=== 3. DECIDE ISOLATED STATE/TRAITS, EXHAUSTED SKIP & PUSHINESS REMAP ===")
    assert abs(map_push_score(0.0, "push", remap=True) - 0.40) < 1e-6
    assert abs(map_push_score(1.0, "push", remap=True) - 0.70) < 1e-6
    assert abs(map_push_score(2.0, "push", remap=True) - 1.00) < 1e-6
    assert abs(map_push_score(1.0, "push", remap=False) - 0.50) < 1e-6

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = RollingAudioRecorder()
            audio.start_run(condition="silent", seed=42, run_dir=Path(tmp) / "decide_run")
            audio.append_pcm_bytes(_make_fake_pcm(10.0))
            crowd = init_crowd(seed=42)
            # Mark A0 exhausted and place A5 in ring 1 with a pending ask to A6 in ring 1
            crowd.agents[0].exhausted = True
            crowd.agents[0].action = "retreat"
            crowd.agents[5].x, crowd.agents[5].y = 0.50, 0.14
            crowd.agents[6].x, crowd.agents[6].y = 0.50, 0.10
            assert ring_for_agent(crowd.agents[6]) == 1

            pending = [
                PendingAsk(
                    id="q1",
                    tick_created=1,
                    t_created=3.0,
                    asker_id=5,
                    blocker_id=6,
                    asker_archetype=crowd.agents[5].archetype,
                    blocker_archetype=crowd.agents[6].archetype,
                    true_k=crowd.agents[5].k,
                    text="Mind if I slip forward for this drop?",
                    claimed_keenness=0.95,
                    prompt_summary="test",
                    asker_yielded_before=0,
                    blocker_yielded_before=0,
                    hydrogen_latency_ms=50.0,
                )
            ]
            mock = MockGeminiClient(fail_ring=None)
            wom = WordOfMouth()
            outputs = await run_all_rings_parallel(
                client=mock,  # type: ignore[arg-type]
                crowd=crowd,
                audio=audio,
                wom=wom,
                pending_asks=pending,
                remap_pushiness=True,
            )
            assert len(outputs) >= 1
            all_q: dict[str, dict[str, Any]] = {}
            for call in mock.decide_calls:
                all_q.update(call["questions"])
            # Exhausted A0 must be skipped
            assert "a0_action" not in all_q and "a0_push" not in all_q
            # Active A5 must have state+traits in BOTH a5_action AND a5_push
            assert "a5_action" in all_q and "a5_push" in all_q
            for qk in ("a5_action", "a5_push"):
                inst = all_q[qk]["instructions"]
                assert "Traits: keen" in inst and "space" in inst, f"{qk} missing traits: {inst}"
                assert "Now: energy" in inst and "crowding" in inst and "happiness" in inst, f"{qk} missing state: {inst}"
            # Pending ask injected as noul
            assert "a6_ans_5" in all_q and all_q["a6_ans_5"]["type"] == "noul"
            assert "claims keenness 0.95" in all_q["a6_ans_5"]["instructions"]
            # Ring 1 returned answer for q1
            assert len(outputs[1].ask_answers) == 1 and outputs[1].ask_answers[0]["answer"] == "yes"

    asyncio.run(_run())
    print("PASS: isolated traits+state in both a{id}_action and a{id}_push, exhausted skip, and ask noul verified")


def test_tickloop_and_logging_end_to_end() -> None:
    print("\n=== 4. TICKLOOP END-TO-END (SHOVE, SILENT, TALK, YIELD 2s, LOGGING) ===")

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = RollingAudioRecorder()
            audio.start_run(condition="talk", seed=42, run_dir=Path(tmp) / "talk_run")
            audio.append_pcm_bytes(_make_fake_pcm(16.0))
            physics = PhysicsEngine()
            run_logger = RunLogger()
            broadcasts: list[dict[str, Any]] = []

            async def _bcast(msg: dict[str, Any]) -> None:
                broadcasts.append(msg)

            mock = MockGeminiClient(answer_yes=True)
            loop = TickLoop(
                client=mock,  # type: ignore[arg-type]
                audio=audio,
                physics=physics,
                run_logger=run_logger,
                broadcast_fn=_bcast,
            )

            # Talk condition with a guaranteed contest pair (A10 pushing right behind A20)
            crowd = init_crowd(seed=42, condition="talk")
            crowd.t = 15.0
            crowd.agents[10].x, crowd.agents[10].y, crowd.agents[10].k = 0.50, 0.25, 0.95
            crowd.agents[20].x, crowd.agents[20].y, crowd.agents[20].k = 0.50, 0.20, 0.50
            physics.step(crowd, dt=1.0 / 30.0)

            # Tick 1: decisions run + contest schedules async Hydrogen ask
            f1 = await loop.run_once(crowd)
            assert crowd.use_dsp_proxy is False and crowd.w > 1.0
            assert f1["house"]["section"] == "drop"
            assert "notable" in f1

            # Wait for background Hydrogen ask tasks to finish
            await asyncio.sleep(0.05)
            assert len(mock.hydrogen_calls) >= 1
            assert mock.hydrogen_calls[0]["schema_fields"] == {"text", "claimed_keenness"}
            assert mock.hydrogen_calls[0]["thinking_level"] == "low"
            assert mock.hydrogen_calls[0]["store"] is False

            # Tick 2: drains ready asks, broadcasts pending asks first, then resolves with YES -> 2 s yield
            crowd.t = 18.0
            f2 = await loop.run_once(crowd)
            assert len(f2["answers"]) >= 1
            ans0 = f2["answers"][0]
            assert ans0["answer"] == "yes"
            blocker = crowd.agents[ans0["to"]]
            asker_id = ans0["from"]
            assert blocker.action == "yield"
            assert blocker.yield_until >= crowd.t + 1.9
            assert crowd.yield_count(blocker.id, asker_id) == 1

            # Step physics 0.5s later: blocker MUST still be forced into yield!
            physics.step(crowd, dt=0.5)
            assert blocker.action == "yield" and blocker.pushiness <= 0.06

            # Check logs on disk (meta.json, ticks.jsonl, asks.jsonl, session.wav in same dir)
            assert run_logger.run_dir is not None
            run_dir = run_logger.run_dir
            run_logger.close()
            audio.close()

            meta = json.loads((run_dir / "meta.json").read_text())
            ticks = [json.loads(line) for line in (run_dir / "ticks.jsonl").read_text().splitlines() if line.strip()]
            asks = [json.loads(line) for line in (run_dir / "asks.jsonl").read_text().splitlines() if line.strip()]
            assert meta["condition"] == "talk" and meta["seed"] == 42 and len(meta["agents"]) == 100
            assert len(ticks) == 2 and ticks[0]["house"]["section"] == "drop"
            assert len(asks) >= 1 and "claimed_keenness" in asks[0] and "true_k" in asks[0]
            assert (run_dir / "session.wav").exists()

    asyncio.run(_run())
    print("PASS: TickLoop (talk), async Hydrogen -> next-tick noul -> 2s yield + yield_count + JSONL logs verified")


def main() -> None:
    test_sections_and_wom()
    test_ask_schema_and_contests()
    test_decide_questions_and_push_remap()
    test_tickloop_and_logging_end_to_end()
    print("\nALL PHASE D & E UNIT TESTS PASSED.")


if __name__ == "__main__":
    main()
