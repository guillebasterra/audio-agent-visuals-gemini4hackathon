"""House ear: unfiltered 15 s Decisions call estimating live musical section & weight w.

Plan.md sections 3, 6, 8.1:
- Runs once per tick on unfiltered last 15 s of audio.
- Same section / drop_eta / rising questions as ring calls, keyed `house_*`.
- Computes expected reward weight w = sum(p(section) * SECTION_WEIGHTS[section]).
- Agents are NEVER shown w or any house-ear outputs.
- On failure, keeps the previous w (1.0 before first answer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from server.config import SECTION_WEIGHTS
from server.gemini_client import ApiResult, GeminiClient

logger = logging.getLogger("agent_crowd.sections")

SECTION_CRITERIA: dict[str, str] = {
    "intro": "Sparse opening groove, drums or textures gradually entering.",
    "build": "Rising tension, snare rolls, risers, filter sweeps leading toward a drop.",
    "drop": "Full-energy peak section with heavy kick and bassline hitting hard.",
    "groove": "Steady rhythmic dance groove with kick and bass, neither peaking nor building.",
    "breakdown": "Drums or bass drop out, atmospheric or melodic calm before a build.",
    "outro": "Stripping elements down toward the end of the track.",
    "other": "Silence, speech, transition noise, or unclear.",
}

DROP_ETA_CRITERIA: dict[str, str] = {
    "now": "Drop is hitting right now or within 1-2 beats.",
    "next_8_bars": "Imminent build; drop likely arrives within about 8 bars.",
    "within_32_bars": "Tension or breakdown developing; drop likely within about 32 bars.",
    "not_soon": "Steady groove, intro, or outro with no drop imminent.",
}


def build_music_questions(prefix: str) -> dict[str, dict[str, Any]]:
    """Build the 3 shared music-perception questions (`section`, `drop_eta`, `rising`)."""
    return {
        f"{prefix}_section": {
            "type": "choice",
            "instructions": "Classify the current musical section heard in this dancefloor audio clip.",
            "criteria": SECTION_CRITERIA,
        },
        f"{prefix}_drop_eta": {
            "type": "choice",
            "instructions": "Based on the audio, how soon is a major beat drop?",
            "criteria": DROP_ETA_CRITERIA,
        },
        f"{prefix}_rising": {
            "type": "noul",
            "instructions": "Is the energy building right now?",
        },
    }


def format_features_text(t: float, dsp: dict[str, Any], kick_rate: float, coarse: bool = False) -> str:
    """Format supplemental DSP feature text sent alongside raw audio (plan.md section 6).

    In rings 4-5 (`coarse=True`), round numeric features to 1 decimal place.
    """
    t_sec = max(0, int(round(t)))
    mins, secs = divmod(t_sec, 60)
    energy = float(dsp.get("energy", 0.0) or 0.0)
    trend = float(dsp.get("trend", 0.0) or 0.0)
    bass_gap = float(dsp.get("bass_gap", 0.0) or 0.0)
    if coarse:
        return (
            f"t={mins}:{secs:02d} energy={energy:.1f} trend={trend:+.1f} "
            f"bass_gap={bass_gap:.0f}s kick_rate={kick_rate:.1f}/s"
        )
    return (
        f"t={mins}:{secs:02d} energy={energy:.2f} trend={trend:+.2f} "
        f"bass_gap={bass_gap:.0f}s kick_rate={kick_rate:.1f}/s"
    )


def parse_music_answers(answers: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    """Extract normalized section, drop_eta, rising fields from a Decisions response."""
    if not answers:
        return {}

    out: dict[str, Any] = {}
    sec_ans = answers.get(f"{prefix}_section")
    if sec_ans is not None:
        probs = dict(getattr(sec_ans, "probabilities", {}) or {})
        choice = str(getattr(sec_ans, "choice", "groove"))
        p_choice = float(probs.get(choice, getattr(sec_ans, "confidence", 0.0)))
        out["section"] = choice
        out["section_p"] = round(p_choice, 4)
        out["section_probs"] = {k: round(float(v), 4) for k, v in probs.items()}

    eta_ans = answers.get(f"{prefix}_drop_eta")
    if eta_ans is not None:
        probs = dict(getattr(eta_ans, "probabilities", {}) or {})
        choice = str(getattr(eta_ans, "choice", "not_soon"))
        p_choice = float(probs.get(choice, getattr(eta_ans, "confidence", 0.0)))
        out["drop_eta"] = choice
        out["drop_eta_p"] = round(p_choice, 4)
        out["drop_eta_probs"] = {k: round(float(v), 4) for k, v in probs.items()}

    rise_ans = answers.get(f"{prefix}_rising")
    if rise_ans is not None:
        out["rising"] = round(float(getattr(rise_ans, "noul", 0.5)), 4)

    return out


def expected_section_weight(section_probs: dict[str, float], fallback: float = 1.0) -> float:
    """w = sum_s p(s) * SECTION_WEIGHTS[s]."""
    if not section_probs:
        return fallback
    total_p = sum(float(p) for p in section_probs.values())
    if total_p <= 1e-6:
        return fallback
    w = sum(float(p) * SECTION_WEIGHTS.get(sec, 0.5) for sec, p in section_probs.items())
    return float(w / total_p)


@dataclass
class HouseEarOutput:
    ok: bool
    w: float
    section: str = "groove"
    section_p: float = 0.0
    section_probs: dict[str, float] = field(default_factory=dict)
    drop_eta: str = "not_soon"
    drop_eta_p: float = 0.0
    rising: float = 0.5
    api: ApiResult[Any] | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "w": round(self.w, 4),
            "section": self.section,
            "p": round(self.section_p, 4),
            "section_probs": self.section_probs,
            "drop_eta": self.drop_eta,
            "drop_eta_p": round(self.drop_eta_p, 4),
            "rising": round(self.rising, 4),
        }


async def run_house_ear(
    client: GeminiClient,
    b64_wav: str,
    features_text: str,
    prev_w: float = 1.0,
) -> HouseEarOutput:
    """Run one unfiltered 15 s Decisions call for the house ear."""
    questions = build_music_questions("house")
    res = await client.decide(
        input_blocks=[
            {"type": "audio", "data": b64_wav, "mime_type": "audio/wav"},
            {"type": "text", "text": features_text},
        ],
        questions=questions,
        label="house_ear",
    )
    if not res.ok or res.value is None:
        logger.warning("House ear failed (%s); keeping previous w=%.3f", res.error, prev_w)
        return HouseEarOutput(ok=False, w=prev_w, api=res)

    parsed = parse_music_answers(getattr(res.value, "answers", None), "house")
    sec_probs = parsed.get("section_probs", {})
    w = expected_section_weight(sec_probs, fallback=prev_w)
    return HouseEarOutput(
        ok=True,
        w=w,
        section=parsed.get("section", "groove"),
        section_p=parsed.get("section_p", 0.0),
        section_probs=sec_probs,
        drop_eta=parsed.get("drop_eta", "not_soon"),
        drop_eta_p=parsed.get("drop_eta_p", 0.0),
        rising=parsed.get("rising", 0.5),
        api=res,
    )
