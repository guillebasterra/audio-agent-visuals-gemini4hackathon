"""Offline Gemini 4 Neon narrative report generator (Phase F / H).

Sends `out/summary.json` + 15 stratified sampled asks from `talk/asks.jsonl`
(and optional `out/highlights.json`) to `gemini-4-neon` (Interactions API,
`thinking_level="high"`), validates `{takeaways: [{claim, number, quote}] x 5, caveats: [...]}`,
and writes `out/report.md` + `out/neon_report.json`.

Usage:
    .venv/bin/python -m analysis.neon_report out/summary.json <talk_dir>/asks.jsonl \
        [--highlights out/highlights.json] [--out out/report.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import MODEL_NEON
from server.gemini_client import GeminiClient


class Takeaway(BaseModel):
    claim: str = Field(description="Concise, insightful analytical claim grounded in the experiment data.")
    number: str = Field(description="Exact number(s) verbatim from summary.json backing the claim.")
    quote: str = Field(
        description="Verbatim quote from one of the 15 sampled asks (include agent IDs and yes/no outcome)."
    )


class NeonReport(BaseModel):
    takeaways: list[Takeaway] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 headline takeaways comparing shove, silent, and talk.",
    )
    caveats: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Honest methodological caveats, failure rates, sample-size limits, or confounds.",
    )


NEON_SYSTEM_PROMPT = (
    "You are a quantitative behavioral-economics analyst writing the findings report for 'Agent Crowd', "
    "a 100-agent house-music dancefloor experiment comparing three conditions on the exact same track and seed: "
    "(1) shove (scripted heuristic, no minds), (2) silent (Gemini Decisions on ring-filtered audio only), "
    "and (3) talk (Decisions + Gemini 4 Hydrogen natural-language asks with probabilistic yielding). "
    "No track labels are ever given to the agents — they only hear raw audio.\n"
    "Rules:\n"
    "- Return EXACTLY 5 sharp, non-overlapping takeaways covering: anticipation/acoustics, welfare & fairness, "
    "negotiation/honesty/reciprocity, pacing/energy or gridlock, and who wins (OLS / archetypes).\n"
    "- Every number MUST appear verbatim in the provided summary.json or sampled asks — never invent or round contradictorily.\n"
    "- Every quote MUST copy verbatim text from one of the 15 sampled asks, citing its asker -> blocker IDs and outcome.\n"
    "- Include 3-4 candid methodological caveats (single track/seed, probabilistic answer sampling, reciprocity sample size, any API errors/429s)."
)


def sample_stratified_asks(asks: list[dict[str, Any]], k: int = 15, seed: int = 42) -> list[dict[str, Any]]:
    """Sample k asks stratified by (outcome, asker_archetype), prioritizing vivid diversity."""
    if len(asks) <= k:
        return list(asks)

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in asks:
        key = (str(a.get("answer", "no")), str(a.get("asker_archetype", "between")))
        buckets[key].append(a)

    for b in buckets.values():
        rng.shuffle(b)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        aid = str(item.get("id"))
        if aid not in seen_ids and len(selected) < k:
            seen_ids.add(aid)
            selected.append(item)

    # First ensure up to 2 notable edge cases (highest keenness exaggeration that yielded, or repeat reciprocity)
    yielded_over = sorted(
        [a for a in asks if a.get("answer") == "yes"],
        key=lambda x: float(x.get("claimed_keenness", 0.0)) - float(x.get("true_k", 0.0)),
        reverse=True,
    )
    if yielded_over:
        _add(yielded_over[0])

    recip = [a for a in asks if int(a.get("asker_yielded_before", 0)) > 0]
    if recip:
        _add(recip[0])

    # Round-robin across the 6 (outcome, archetype) strata
    strata_order = [
        ("yes", "diehard"),
        ("no", "diehard"),
        ("yes", "between"),
        ("no", "between"),
        ("yes", "dancer"),
        ("no", "dancer"),
    ]
    made_progress = True
    while len(selected) < k and made_progress:
        made_progress = False
        for key in strata_order:
            bucket = buckets.get(key, [])
            while bucket:
                cand = bucket.pop()
                if str(cand.get("id")) not in seen_ids:
                    _add(cand)
                    made_progress = True
                    break
            if len(selected) >= k:
                break

    # Fill any remaining slots if some strata were empty
    if len(selected) < k:
        leftover = [a for a in asks if str(a.get("id")) not in seen_ids]
        rng.shuffle(leftover)
        for cand in leftover:
            _add(cand)
            if len(selected) >= k:
                break

    selected.sort(key=lambda x: float(x.get("t", 0.0)))
    return selected


def format_ask_compact(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": a.get("id"),
        "t": round(float(a.get("t", 0.0)), 1),
        "from": f"A{a['from']}",
        "to": f"A{a['to']}",
        "asker_archetype": a.get("asker_archetype"),
        "blocker_archetype": a.get("blocker_archetype"),
        "text": a.get("text"),
        "claimed_keenness": round(float(a.get("claimed_keenness", 0.0)), 2),
        "true_k": round(float(a.get("true_k", 0.0)), 2),
        "overstatement": round(float(a.get("claimed_keenness", 0.0)) - float(a.get("true_k", 0.0)), 3),
        "asker_yielded_before": int(a.get("asker_yielded_before", 0)),
        "answer": a.get("answer"),
        "answer_p": round(float(a.get("answer_p", 0.0)), 3),
    }


def render_markdown(
    report: NeonReport,
    summary: dict[str, Any],
    sampled_asks: list[dict[str, Any]],
    api_meta: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Agent Crowd — Experiment Report (Gemini 4 Neon)",
        "",
        f"- **Track:** `{summary.get('track')}` | **Seed:** `{summary.get('seed')}`",
        f"- **Analyst model:** `{MODEL_NEON}` (`thinking_level=\"high\"`, latency `{api_meta['latency_ms']:.0f} ms`, tokens `{api_meta['total_tokens']}`)",
        "",
        "## Five Key Takeaways",
        "",
    ]
    for i, t in enumerate(report.takeaways, start=1):
        lines.append(f"### {i}. {t.claim}")
        lines.append(f"- **Numbers:** {t.number}")
        lines.append(f"- **Evidence quote:** {t.quote}")
        lines.append("")

    lines.append("## Methodological Caveats")
    lines.append("")
    for c in report.caveats:
        lines.append(f"- {c}")

    lines.append("")
    lines.append("## Stratified Sample of 15 Asks Sent to Neon")
    lines.append("")
    lines.append("| t (s) | Pair | Archetypes | True k -> Claimed | Outcome (p) | Ask |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for a in sampled_asks:
        lines.append(
            f"| {a['t']:.1f} | {a['from']} → {a['to']} | "
            f"{a['asker_archetype']} → {a['blocker_archetype']} | "
            f"{a['true_k']:.2f} → {a['claimed_keenness']:.2f} ({a['overstatement']:+.2f}) | "
            f"{a['answer']} ({a['answer_p']:.2f}) | \"{a['text']}\" |"
        )
    lines.append("")
    return "\n".join(lines)


async def generate_report(args: argparse.Namespace) -> None:
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    asks_raw: list[dict[str, Any]] = []
    asks_path = Path(args.asks_jsonl)
    if asks_path.exists():
        for line in asks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                asks_raw.append(json.loads(line))

    sampled_raw = sample_stratified_asks(asks_raw, k=15, seed=42)
    sampled_compact = [format_ask_compact(a) for a in sampled_raw]

    highlights: dict[str, Any] | None = None
    if args.highlights and Path(args.highlights).exists():
        highlights = json.loads(Path(args.highlights).read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "summary": summary,
        "sampled_asks_15": sampled_compact,
    }
    if highlights is not None:
        payload["highlights"] = highlights

    prompt = (
        "Analyze the following Agent Crowd 3-condition experiment summary and 15 stratified asks.\n"
        "Produce the structured JSON report with exactly 5 takeaways ({claim, number, quote}) and 3-4 caveats.\n\n"
        + json.dumps(payload, indent=2)
    )

    client = GeminiClient()
    try:
        res = await client.interact_structured(
            model=MODEL_NEON,
            system_instruction=NEON_SYSTEM_PROMPT,
            input_blocks=[{"type": "text", "text": prompt}],
            schema_model=NeonReport,
            thinking_level="high",
            store=False,
            timeout=120.0,
            label="neon_report",
        )
    finally:
        await client.aclose()

    if not res.ok or res.value is None:
        raise RuntimeError(f"Neon report call failed: {res.error}")

    report: NeonReport = res.value
    api_meta = {
        "model": MODEL_NEON,
        "thinking_level": "high",
        "latency_ms": round(res.latency_ms, 1),
        "input_tokens": res.input_tokens,
        "thought_tokens": res.thought_tokens,
        "output_tokens": res.output_tokens,
        "total_tokens": res.total_tokens,
        "rate_limited": res.rate_limited,
    }

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_text = render_markdown(report, summary, sampled_compact, api_meta)
    out_md.write_text(md_text, encoding="utf-8")

    out_json = out_md.with_name("neon_report.json")
    out_json.write_text(
        json.dumps(
            {
                "api": api_meta,
                "report": report.model_dump(),
                "sampled_asks": sampled_compact,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote {out_md} and {out_json} "
        f"(latency={api_meta['latency_ms']:.0f}ms, tokens={api_meta['total_tokens']}, "
        f"thought={api_meta['thought_tokens']})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Gemini 4 Neon narrative report.")
    p.add_argument("summary_json", help="Path to out/summary.json")
    p.add_argument("asks_jsonl", help="Path to talk run's asks.jsonl")
    p.add_argument("--highlights", default="", help="Optional path to out/highlights.json")
    p.add_argument("--out", default="out/report.md", help="Output markdown report path")
    args = p.parse_args()
    asyncio.run(generate_report(args))


if __name__ == "__main__":
    main()
