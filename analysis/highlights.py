"""Highlight timestamp extractor for demo playback and narrative analysis (Phase H).

Usage:
    .venv/bin/python -m analysis.highlights <talk_run_dir> [--out out/highlights.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze import detect_drop_onsets, load_run


def extract_highlights(run_dir: Path) -> dict[str, Any]:
    run = load_run(run_dir)
    meta = run["meta"]
    ticks = run["ticks"]
    asks = run["asks"]
    agents_meta = {a["id"]: a for a in meta["agents"]}

    # 1. Biggest single-agent H swing (overall range + largest single-window surge/drop)
    h_series: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in ticks:
        t = float(row["t"])
        for a in row["agents"]:
            h_series[int(a["id"])].append((t, float(a["H"])))

    swings: list[dict[str, Any]] = []
    for aid, pts in h_series.items():
        if not pts:
            continue
        t_min, h_min = min(pts, key=lambda x: x[1])
        t_max, h_max = max(pts, key=lambda x: x[1])
        am = agents_meta.get(aid, {})
        swings.append(
            {
                "agent_id": aid,
                "label": f"A{aid}",
                "archetype": am.get("archetype"),
                "k": am.get("k"),
                "s": am.get("s"),
                "h_min": round(h_min, 4),
                "t_min": round(t_min, 1),
                "h_max": round(h_max, 4),
                "t_max": round(t_max, 1),
                "swing": round(h_max - h_min, 4),
                "direction": "rise" if t_max >= t_min else "fall",
            }
        )
    swings.sort(key=lambda x: x["swing"], reverse=True)

    # 2. Asks where claimed_keenness overstated true k by > 0.3 and blocker still yielded
    overstated_gt_03_yields: list[dict[str, Any]] = []
    overstated_gt_03_all: list[dict[str, Any]] = []
    all_yielded_by_diff: list[dict[str, Any]] = []

    for ask in asks:
        diff = float(ask.get("claimed_keenness", 0.0)) - float(ask.get("true_k", 0.0))
        entry = {
            "id": ask.get("id"),
            "t": round(float(ask.get("t", 0.0)), 1),
            "tick": int(ask.get("tick", 0)),
            "from": int(ask["from"]),
            "to": int(ask["to"]),
            "asker_archetype": ask.get("asker_archetype"),
            "blocker_archetype": ask.get("blocker_archetype"),
            "true_k": round(float(ask.get("true_k", 0.0)), 4),
            "claimed_keenness": round(float(ask.get("claimed_keenness", 0.0)), 4),
            "overstatement": round(diff, 4),
            "answer": ask.get("answer"),
            "answer_p": round(float(ask.get("answer_p", 0.0)), 4),
            "text": ask.get("text"),
        }
        if diff > 0.3:
            overstated_gt_03_all.append(entry)
            if ask.get("answer") == "yes":
                overstated_gt_03_yields.append(entry)
        if ask.get("answer") == "yes":
            all_yielded_by_diff.append(entry)

    overstated_gt_03_yields.sort(key=lambda x: x["overstatement"], reverse=True)
    overstated_gt_03_all.sort(key=lambda x: x["overstatement"], reverse=True)
    all_yielded_by_diff.sort(key=lambda x: x["overstatement"], reverse=True)

    # 3. Most-asked, most-yielding, and most-asking agents
    asked_counts: Counter[int] = Counter()
    yielded_counts: Counter[int] = Counter()
    asker_counts: Counter[int] = Counter()
    asker_yes_counts: Counter[int] = Counter()

    for ask in asks:
        fid = int(ask["from"])
        tid = int(ask["to"])
        asked_counts[tid] += 1
        asker_counts[fid] += 1
        if ask.get("answer") == "yes":
            yielded_counts[tid] += 1
            asker_yes_counts[fid] += 1

    def _agent_badge(aid: int, extra: dict[str, Any]) -> dict[str, Any]:
        am = agents_meta.get(aid, {})
        return {
            "agent_id": aid,
            "label": f"A{aid}",
            "archetype": am.get("archetype"),
            "k": am.get("k"),
            "s": am.get("s"),
            **extra,
        }

    most_asked = [
        _agent_badge(
            aid,
            {
                "times_asked": cnt,
                "times_yielded": yielded_counts[aid],
                "yield_rate": round(yielded_counts[aid] / cnt, 3) if cnt > 0 else 0.0,
            },
        )
        for aid, cnt in asked_counts.most_common(5)
    ]
    most_yielding = [
        _agent_badge(
            aid,
            {
                "times_yielded": cnt,
                "times_asked": asked_counts[aid],
                "yield_rate": round(cnt / asked_counts[aid], 3) if asked_counts[aid] > 0 else 0.0,
            },
        )
        for aid, cnt in yielded_counts.most_common(5)
    ]
    most_asking = [
        _agent_badge(
            aid,
            {
                "asks_made": cnt,
                "asks_accepted": asker_yes_counts[aid],
                "success_rate": round(asker_yes_counts[aid] / cnt, 3) if cnt > 0 else 0.0,
            },
        )
        for aid, cnt in asker_counts.most_common(5)
    ]

    # 4. Crowd's maximum compression before each drop (and overall peak compression)
    onsets = detect_drop_onsets(ticks)
    pre_drop_peaks: list[dict[str, Any]] = []
    for onset in onsets:
        t_drop = float(onset["t"])
        window = [r for r in ticks if (t_drop - 24.0) <= float(r["t"]) <= t_drop]
        if not window:
            continue
        best_row = max(window, key=lambda r: float(np.mean([a["v"] for a in r["agents"]])))
        mv = float(np.mean([a["v"] for a in best_row["agents"]]))
        mrho = float(np.mean([a["rho"] for a in best_row["agents"]]))
        pre_drop_peaks.append(
            {
                "drop_onset_t": round(t_drop, 1),
                "peak_pre_drop_t": round(float(best_row["t"]), 1),
                "seconds_before_drop": round(t_drop - float(best_row["t"]), 1),
                "mean_v": round(mv, 4),
                "mean_rho": round(mrho, 4),
                "house_section_at_peak": best_row["house"].get("section"),
            }
        )

    best_pre_drop = max(pre_drop_peaks, key=lambda x: x["mean_v"]) if pre_drop_peaks else None
    overall_peak_row = max(ticks, key=lambda r: float(np.mean([a["v"] for a in r["agents"]]))) if ticks else None
    overall_peak = (
        {
            "t": round(float(overall_peak_row["t"]), 1),
            "tick": int(overall_peak_row["tick"]),
            "mean_v": round(float(np.mean([a["v"] for a in overall_peak_row["agents"]])), 4),
            "mean_rho": round(float(np.mean([a["rho"] for a in overall_peak_row["agents"]])), 4),
            "house_section": overall_peak_row["house"].get("section"),
        }
        if overall_peak_row
        else None
    )

    try:
        rel_dir = str(run_dir.resolve().relative_to(ROOT))
    except ValueError:
        rel_dir = str(run_dir)
    return {
        "run_dir": rel_dir,
        "condition": run["condition"],
        "seed": meta.get("seed"),
        "track": meta.get("source_label"),
        "biggest_H_swings": swings[:5],
        "overstated_gt_0_3_and_yielded": overstated_gt_03_yields,
        "overstated_gt_0_3_all_outcomes": overstated_gt_03_all[:10],
        "top_exaggerated_yields": all_yielded_by_diff[:5],
        "most_asked_agents": most_asked,
        "most_yielding_agents": most_yielding,
        "most_asking_agents": most_asking,
        "max_pre_drop_compression": best_pre_drop,
        "all_pre_drop_compressions": pre_drop_peaks,
        "overall_peak_compression": overall_peak,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Extract timestamped highlights from a talk run.")
    p.add_argument("run_dir", help="Path to talk run directory (e.g. runs/*_talk_42)")
    p.add_argument("--out", default="out/highlights.json")
    args = p.parse_args()

    highlights = extract_highlights(Path(args.run_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(highlights, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(
        {
            "biggest_H_swing": highlights["biggest_H_swings"][0] if highlights["biggest_H_swings"] else None,
            "overstated_gt_0_3_yields_count": len(highlights["overstated_gt_0_3_and_yielded"]),
            "top_exaggerated_yield": highlights["top_exaggerated_yields"][0] if highlights["top_exaggerated_yields"] else None,
            "most_asked": highlights["most_asked_agents"][0] if highlights["most_asked_agents"] else None,
            "most_yielding": highlights["most_yielding_agents"][0] if highlights["most_yielding_agents"] else None,
            "max_pre_drop_compression": highlights["max_pre_drop_compression"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
