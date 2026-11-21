"""Post-run analysis and publication-grade plots for Agent Crowd (Phase F + Phase H).

Usage:
    .venv/bin/python -m analysis.analyze <shove_dir> <silent_dir> <talk_dir> [--out out]
    .venv/bin/python -m analysis.analyze --manifest out/experiment_manifest.json [--out out]

Outputs in `out/`:
  - summary.json           (complete metrics, reciprocity, keenness check, OLS, gridlock, contagion, ring agreement)
  - anticipation.png       (Core F: mean proximity v over time + house-ear bands + drop lead times)
  - welfare_fairness.png   (Core F: per-agent mean H distribution + Gini + archetype comparison)
  - ring_agreement.png     (Phase H: ring 1..5 section & drop_eta agreement vs house ear)
  - pacing.png             (Phase H: mean energy E over time + breakdown resting + exhaustion)
  - who_wins.png           (Phase H: per-agent mean H vs keenness k scatter + OLS fit)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Avoid matplotlib cache warning in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import AGENT_RADIUS, CONTAGION_RADIUS_MULT, FLOOR_Y_MAX, FLOOR_Y_MIN

CONTAGION_RADIUS = AGENT_RADIUS * CONTAGION_RADIUS_MULT

# TLC plate palette
PAPER_BG = "#f6ebe6"
PANEL_BG = "#fbf4f0"
INK = "#2b2126"
MUTED_INK = "#6e5d64"
GRID_COL = "#e4d6cf"

COND_ORDER = ["shove", "silent", "talk"]
COND_COLORS = {
    "shove": "#6b6eb8",   # TLC blue-violet
    "silent": "#8a3fa0",  # TLC purple
    "talk": "#d9481c",    # TLC red-orange
}
COND_LABELS = {
    "shove": "Shove (no minds)",
    "silent": "Silent (Decisions)",
    "talk": "Talk (Decisions + Hydrogen)",
}

ARCH_COLORS = {
    "diehard": "#d9481c",
    "between": "#8a3fa0",
    "dancer": "#5e68b5",
}

SECTION_BANDS = {
    "drop": ("#d9481c", 0.14),
    "build": ("#d12f7a", 0.09),
    "breakdown": ("#6f6fb8", 0.13),
}

Y_FRONT_EDGE = FLOOR_Y_MIN + (FLOOR_Y_MAX - FLOOR_Y_MIN) * 0.36
Y_MID_EDGE = FLOOR_Y_MIN + (FLOOR_Y_MAX - FLOOR_Y_MIN) * 0.68


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": "#cbbab2",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "text.color": INK,
            "grid.color": GRID_COL,
            "grid.linewidth": 0.7,
            "font.family": "monospace",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "legend.frameon": True,
            "legend.facecolor": PAPER_BG,
            "legend.edgecolor": "#cbbab2",
            "legend.fontsize": 8.5,
        }
    )


def gini_coefficient(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 1e-12:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x_sorted) / (n * total)) - (n + 1.0) / n)


def load_run(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    ticks_path = run_dir / "ticks.jsonl"
    asks_path = run_dir / "asks.jsonl"
    summary_path = run_dir / "run_summary.json"

    if not meta_path.exists() or not ticks_path.exists():
        raise FileNotFoundError(f"Missing meta.json or ticks.jsonl in {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ticks: list[dict[str, Any]] = []
    for line in ticks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ticks.append(json.loads(line))

    asks: list[dict[str, Any]] = []
    if asks_path.exists():
        for line in asks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                asks.append(json.loads(line))

    run_summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    )

    condition = meta.get("condition") or (ticks[0]["condition"] if ticks else "unknown")
    return {
        "dir": run_dir,
        "condition": condition,
        "meta": meta,
        "ticks": ticks,
        "asks": asks,
        "run_summary": run_summary,
    }


def _effective_house_regime(house: dict[str, Any]) -> str:
    """Classify house-ear tick regime using calibrated probabilities (purely from house ear)."""
    sec = str(house.get("section", "groove"))
    probs = house.get("section_probs") or {}
    p_drop = float(probs.get("drop", 0.0))
    p_build = float(probs.get("build", 0.0))
    p_bd = float(probs.get("breakdown", 0.0))
    w = float(house.get("w", 1.0))

    if sec == "drop" or p_drop >= 0.29:
        return "drop"
    if sec == "breakdown" or (p_bd >= 0.18 and w <= 1.12):
        return "breakdown"
    if sec == "build" or (p_build >= 0.22 and p_drop < 0.29):
        return "build"
    return sec


def extract_section_spans(ticks: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Build contiguous (t0, t1, regime) spans from house-ear probabilities."""
    if not ticks:
        return []
    spans: list[tuple[float, float, str]] = []
    cur_sec = _effective_house_regime(ticks[0]["house"])
    start_t = float(ticks[0]["t"])
    prev_t = start_t
    dt_est = float(ticks[1]["t"] - ticks[0]["t"]) if len(ticks) > 1 else 3.0

    for row in ticks[1:]:
        t = float(row["t"])
        sec = _effective_house_regime(row["house"])
        if sec != cur_sec:
            spans.append((start_t, t, cur_sec))
            cur_sec = sec
            start_t = t
        prev_t = t
    spans.append((start_t, prev_t + dt_est, cur_sec))
    return spans


def shade_sections(
    ax: plt.Axes, spans: list[tuple[float, float, str]], add_legend: bool = True
) -> None:
    labeled: set[str] = set()
    for t0, t1, sec in spans:
        if sec in SECTION_BANDS:
            col, alpha = SECTION_BANDS[sec]
            lbl = f"house: {sec}" if (add_legend and sec not in labeled) else None
            labeled.add(sec)
            ax.axvspan(t0, t1, color=col, alpha=alpha, lw=0, label=lbl)


def detect_drop_onsets(ticks: list[dict[str, Any]], min_gap_s: float = 40.0) -> list[dict[str, Any]]:
    """Identify post-breakdown house-ear drop onsets (purely from house ear w & section_probs).

    A drop onset is the first drop tick (`section == 'drop'` or `p_drop >= 0.29`) after a calm/breakdown
    stretch (>= 2 ticks with breakdown regime or house w <= 1.12 in the prior 36 s).
    """
    onsets: list[dict[str, Any]] = []
    last_onset_t = -1e9
    non_drop_streak = 0

    for i, row in enumerate(ticks):
        t = float(row["t"])
        house = row.get("house", {})
        sec = house.get("section", "groove")
        probs = house.get("section_probs", {}) or {}
        p_drop = float(probs.get("drop", 0.0))
        is_drop = (sec == "drop") or (p_drop >= 0.29)

        if not is_drop:
            non_drop_streak += 1
            continue

        # Check whether the preceding 36 s window contained a genuine calm/breakdown stretch
        prior_calm = 0
        for j in range(i - 1, -1, -1):
            dt_back = t - float(ticks[j]["t"])
            if dt_back > 36.0:
                break
            if dt_back >= 2.5:
                jh = ticks[j].get("house") or {}
                if _effective_house_regime(jh) == "breakdown" or float(jh.get("w", 1.5)) <= 1.12:
                    prior_calm += 1

        if t >= 40.0 and non_drop_streak >= 3 and prior_calm >= 2 and (t - last_onset_t) >= min_gap_s:
            onsets.append(
                {
                    "tick_idx": i,
                    "tick": int(row["tick"]),
                    "t": round(t, 2),
                    "p_drop": round(p_drop, 3),
                    "w": round(float(house.get("w", 1.0)), 3),
                }
            )
            last_onset_t = t
        non_drop_streak = 0

    return onsets


def compute_drop_lead_times(
    ticks: list[dict[str, Any]],
    onsets: list[dict[str, Any]],
    lookback_s: float = 30.0,
    rise_delta: float = 0.025,
) -> dict[str, Any]:
    """Compute lead time (s) between mean proximity v starting to rise and each house-ear drop onset."""
    ts = np.array([float(r["t"]) for r in ticks], dtype=np.float64)
    mean_v = np.array(
        [float(np.mean([a["v"] for a in r["agents"]])) for r in ticks], dtype=np.float64
    )

    events: list[dict[str, Any]] = []
    leads: list[float] = []
    pre_gains: list[float] = []

    for onset in onsets:
        t_drop = float(onset["t"])
        mask = (ts >= t_drop - lookback_s) & (ts <= t_drop + 3.0)
        idxs = np.where(mask)[0]
        if idxs.size < 3:
            continue

        # Locate pre-drop trough strictly before t_drop
        pre_idxs = idxs[ts[idxs] <= t_drop]
        if pre_idxs.size < 2:
            continue
        trough_local = pre_idxs[int(np.argmin(mean_v[pre_idxs]))]
        v_trough = float(mean_v[trough_local])
        t_trough = float(ts[trough_local])

        # Find value at drop onset
        drop_idx = int(np.argmin(np.abs(ts - t_drop)))
        v_at_drop = float(mean_v[drop_idx])

        # Find first tick after trough where mean_v rises by >= rise_delta and stays above trough
        t_rise: float | None = None
        for idx in idxs:
            if idx <= trough_local:
                continue
            if mean_v[idx] >= v_trough + rise_delta:
                t_rise = float(ts[idx])
                break

        lead_s = (t_drop - t_rise) if t_rise is not None else 0.0
        # 15 s pre-drop proximity gain
        idx_15 = int(np.argmin(np.abs(ts - (t_drop - 15.0))))
        pre_gain = float(v_at_drop - mean_v[idx_15])

        if t_rise is not None:
            leads.append(lead_s)
        pre_gains.append(pre_gain)
        events.append(
            {
                "drop_t": round(t_drop, 1),
                "trough_t": round(t_trough, 1),
                "trough_v": round(v_trough, 4),
                "rise_t": round(t_rise, 1) if t_rise is not None else None,
                "v_at_drop": round(v_at_drop, 4),
                "lead_time_s": round(lead_s, 2),
                "pre_15s_v_gain": round(pre_gain, 4),
            }
        )

    return {
        "n_drops": len(events),
        "mean_lead_time_s": round(float(np.mean(leads)), 2) if leads else 0.0,
        "median_lead_time_s": round(float(np.median(leads)), 2) if leads else 0.0,
        "mean_pre_15s_v_gain": round(float(np.mean(pre_gains)), 4) if pre_gains else 0.0,
        "events": events,
    }


def compute_spatial_contagion(ticks: list[dict[str, Any]], radius: float = CONTAGION_RADIUS) -> dict[str, float]:
    """Spatial Pearson correlation between agent H (and E) and neighbors' mean within CONTAGION_RADIUS."""
    h_self: list[float] = []
    h_nbr: list[float] = []
    e_self: list[float] = []
    e_nbr: list[float] = []
    r2 = radius * radius

    # Skip first 3 ticks so initial uniform state doesn't skew correlation
    for row in ticks[3:]:
        agents = row["agents"]
        coords = np.array([[a["x"], a["y"]] for a in agents], dtype=np.float64)
        hs = np.array([a["H"] for a in agents], dtype=np.float64)
        es = np.array([a["E"] for a in agents], dtype=np.float64)

        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist2, np.inf)
        nbr_mask = dist2 <= r2
        counts = nbr_mask.sum(axis=1)
        valid = counts > 0
        if not np.any(valid):
            continue
        mean_h_nbr = (nbr_mask[valid] @ hs) / counts[valid]
        mean_e_nbr = (nbr_mask[valid] @ es) / counts[valid]
        h_self.extend(hs[valid].tolist())
        h_nbr.extend(mean_h_nbr.tolist())
        e_self.extend(es[valid].tolist())
        e_nbr.extend(mean_e_nbr.tolist())

    h_corr = float(np.corrcoef(h_self, h_nbr)[0, 1]) if len(h_self) > 10 else 0.0
    e_corr = float(np.corrcoef(e_self, e_nbr)[0, 1]) if len(e_self) > 10 else 0.0
    return {
        "H_neighbor_corr": round(h_corr, 4),
        "E_neighbor_corr": round(e_corr, 4),
        "n_agent_ticks": len(h_self),
    }


def compute_gridlock(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of push-ticks with < 0.01 proximity progress on the next tick."""
    total_push = 0
    stalled_push = 0
    drop_push = 0
    drop_stalled = 0
    progress_vals: list[float] = []

    for i in range(len(ticks) - 1):
        cur = ticks[i]
        nxt = ticks[i + 1]
        is_drop = _effective_house_regime(cur["house"]) == "drop"
        nxt_v = {a["id"]: float(a["v"]) for a in nxt["agents"]}
        for a in cur["agents"]:
            if a.get("action") == "push" and not a.get("exhausted", False):
                dv = nxt_v.get(a["id"], float(a["v"])) - float(a["v"])
                total_push += 1
                progress_vals.append(dv)
                is_stall = dv < 0.01
                if is_stall:
                    stalled_push += 1
                if is_drop:
                    drop_push += 1
                    if is_stall:
                        drop_stalled += 1

    return {
        "gridlock_rate": round(stalled_push / total_push, 4) if total_push > 0 else 0.0,
        "push_ticks_total": total_push,
        "push_ticks_stalled": stalled_push,
        "drop_gridlock_rate": round(drop_stalled / drop_push, 4) if drop_push > 0 else 0.0,
        "mean_push_dv": round(float(np.mean(progress_vals)), 5) if progress_vals else 0.0,
    }


def compute_ring_agreement(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-ring (1..5) section and drop_eta agreement with the house ear."""
    by_ring: dict[str, dict[str, Any]] = {}
    for r in range(1, 6):
        rk = str(r)
        n = 0
        sec_agree = 0
        eta_agree = 0
        prob_l1s: list[float] = []
        drop_house_n = 0
        drop_ring_p_sum = 0.0
        drop_ring_hit = 0
        build_house_n = 0
        build_ring_soon_hit = 0

        for row in ticks:
            ring_d = (row.get("rings") or {}).get(rk)
            house = row.get("house") or {}
            if not ring_d or not house.get("ok", True):
                continue
            n += 1
            h_sec = house.get("section")
            r_sec = ring_d.get("section")
            h_eta = house.get("drop_eta")
            r_eta = ring_d.get("drop_eta")
            if r_sec == h_sec:
                sec_agree += 1
            if r_eta == h_eta:
                eta_agree += 1

            hp = house.get("section_probs") or {}
            rp = ring_d.get("section_probs") or {}
            if hp and rp:
                keys = set(hp.keys()) | set(rp.keys())
                l1 = sum(abs(float(hp.get(k, 0.0)) - float(rp.get(k, 0.0))) for k in keys)
                prob_l1s.append(1.0 - 0.5 * l1)  # overlap in [0, 1]

            if _effective_house_regime(house) == "drop":
                drop_house_n += 1
                drop_ring_p_sum += float(rp.get("drop", 0.0))
                if r_sec == "drop" or float(rp.get("drop", 0.0)) >= 0.29:
                    drop_ring_hit += 1
            if _effective_house_regime(house) == "build":
                build_house_n += 1
                if r_eta in ("next_8_bars", "now", "within_32_bars") or r_sec in ("build", "drop"):
                    build_ring_soon_hit += 1

        by_ring[rk] = {
            "n_ticks": n,
            "section_agreement": round(sec_agree / n, 4) if n > 0 else None,
            "drop_eta_agreement": round(eta_agree / n, 4) if n > 0 else None,
            "section_prob_overlap": round(float(np.mean(prob_l1s)), 4) if prob_l1s else None,
            "drop_recall": round(drop_ring_hit / drop_house_n, 4) if drop_house_n > 0 else None,
            "mean_p_drop_on_house_drop": round(drop_ring_p_sum / drop_house_n, 4)
            if drop_house_n > 0
            else None,
            "build_anticipation_hit": round(build_ring_soon_hit / build_house_n, 4)
            if build_house_n > 0
            else None,
        }
    return by_ring


def compute_ols(
    y: np.ndarray,
    X_cols: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    """Fit OLS with intercept and return coefficients, SE, t-stats, p-values, and R^2."""
    n = y.size
    active_cols: list[tuple[str, np.ndarray]] = []
    omitted: list[str] = []
    for name, col in X_cols:
        if np.std(col) > 1e-10:
            active_cols.append((name, col))
        else:
            omitted.append(name)

    names = ["intercept"] + [c[0] for c in active_cols]
    X = np.column_stack([np.ones(n, dtype=np.float64)] + [c[1] for c in active_cols])
    k = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 1e-12 else 0.0
    df = max(1, n - k)
    sigma2 = sse / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    t_vals = beta / se
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df=df))

    terms: dict[str, dict[str, float]] = {}
    for name, b, s_err, tv, pv in zip(names, beta, se, t_vals, p_vals):
        terms[name] = {
            "coef": round(float(b), 5),
            "se": round(float(s_err), 5),
            "t": round(float(tv), 3),
            "p": round(float(pv), 5),
        }

    return {
        "n": int(n),
        "r_squared": round(float(r2), 4),
        "adj_r_squared": round(float(1.0 - (1.0 - r2) * (n - 1) / df), 4),
        "omitted_zero_variance": omitted,
        "terms": terms,
    }


def analyze_asks(asks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute ask acceptance, reciprocity, keenness exaggeration, and archetype breakdowns."""
    n = len(asks)
    if n == 0:
        return {"asks_total": 0}

    yes_list = [a for a in asks if a.get("answer") == "yes"]
    no_list = [a for a in asks if a.get("answer") != "yes"]
    base_rate = len(yes_list) / n
    all_p = [float(a.get("answer_p", 0.0)) for a in asks]

    # Reciprocity: P(yes | asker previously yielded to blocker) vs P(yes | not)
    recip_asks = [a for a in asks if int(a.get("asker_yielded_before", 0)) > 0]
    non_recip_asks = [a for a in asks if int(a.get("asker_yielded_before", 0)) == 0]
    recip_rate = (
        sum(1 for a in recip_asks if a.get("answer") == "yes") / len(recip_asks)
        if recip_asks
        else None
    )
    non_recip_rate = (
        sum(1 for a in non_recip_asks if a.get("answer") == "yes") / len(non_recip_asks)
        if non_recip_asks
        else None
    )

    # Keenness honesty / exaggeration
    diffs = [float(a.get("claimed_keenness", 0.0)) - float(a.get("true_k", 0.0)) for a in asks]
    over_01 = [a for a, d in zip(asks, diffs) if d > 0.1]
    over_03 = [a for a, d in zip(asks, diffs) if d > 0.3]
    honest = [a for a, d in zip(asks, diffs) if abs(d) <= 0.1]

    by_asker_arch: dict[str, dict[str, Any]] = {}
    by_blocker_arch: dict[str, dict[str, Any]] = {}
    for arch in ("diehard", "between", "dancer"):
        a_sub = [a for a in asks if a.get("asker_archetype") == arch]
        if a_sub:
            a_diffs = [float(a["claimed_keenness"]) - float(a["true_k"]) for a in a_sub]
            by_asker_arch[arch] = {
                "count": len(a_sub),
                "yes_rate": round(sum(1 for a in a_sub if a["answer"] == "yes") / len(a_sub), 4),
                "mean_answer_p": round(float(np.mean([a["answer_p"] for a in a_sub])), 4),
                "mean_keenness_diff": round(float(np.mean(a_diffs)), 4),
                "mean_claimed_k": round(float(np.mean([a["claimed_keenness"] for a in a_sub])), 4),
                "mean_true_k": round(float(np.mean([a["true_k"] for a in a_sub])), 4),
            }
        b_sub = [a for a in asks if a.get("blocker_archetype") == arch]
        if b_sub:
            by_blocker_arch[arch] = {
                "count": len(b_sub),
                "yield_rate": round(sum(1 for a in b_sub if a["answer"] == "yes") / len(b_sub), 4),
                "mean_answer_p": round(float(np.mean([a["answer_p"] for a in b_sub])), 4),
            }

    return {
        "asks_total": n,
        "asks_yes": len(yes_list),
        "asks_no": len(no_list),
        "ask_acceptance_rate": round(base_rate, 4),
        "mean_answer_p": round(float(np.mean(all_p)), 4),
        "hydrogen_latency_p50_ms": round(
            float(np.median([float(a.get("hydrogen_latency_ms", 0.0)) for a in asks])), 1
        ),
        "hydrogen_latency_p95_ms": round(
            float(np.percentile([float(a.get("hydrogen_latency_ms", 0.0)) for a in asks], 95)), 1
        ),
        "reciprocity": {
            "base_yes_rate": round(base_rate, 4),
            "asker_yielded_before_n": len(recip_asks),
            "asker_yielded_before_yes_rate": round(recip_rate, 4) if recip_rate is not None else None,
            "asker_yielded_before_mean_p": round(
                float(np.mean([a["answer_p"] for a in recip_asks])), 4
            )
            if recip_asks
            else None,
            "no_prior_yield_n": len(non_recip_asks),
            "no_prior_yield_yes_rate": round(non_recip_rate, 4) if non_recip_rate is not None else None,
            "no_prior_yield_mean_p": round(
                float(np.mean([a["answer_p"] for a in non_recip_asks])), 4
            )
            if non_recip_asks
            else None,
        },
        "keenness_truth_check": {
            "mean_overstatement": round(float(np.mean(diffs)), 4),
            "median_overstatement": round(float(np.median(diffs)), 4),
            "max_overstatement": round(float(np.max(diffs)), 4),
            "min_overstatement": round(float(np.min(diffs)), 4),
            "share_overstated_gt_0_1": round(len(over_01) / n, 4),
            "count_overstated_gt_0_3": len(over_03),
            "yes_rate_when_overstated_gt_0_1": round(
                sum(1 for a in over_01 if a["answer"] == "yes") / len(over_01), 4
            )
            if over_01
            else None,
            "yes_rate_when_honest_within_0_1": round(
                sum(1 for a in honest if a["answer"] == "yes") / len(honest), 4
            )
            if honest
            else None,
        },
        "by_asker_archetype": by_asker_arch,
        "by_blocker_archetype": by_blocker_arch,
    }


def analyze_condition(
    run: dict[str, Any],
    ref_onsets: list[dict[str, Any]],
    ref_breakdown_ticks: set[int],
) -> dict[str, Any]:
    """Compute all core F + Phase H metrics for a single run."""
    meta = run["meta"]
    ticks = run["ticks"]
    asks = run["asks"]
    run_summary = run["run_summary"] or {}
    cond = run["condition"]

    agents_meta = {a["id"]: a for a in meta["agents"]}
    n_agents = len(agents_meta)

    # Per-tick time series
    ts = np.array([float(r["t"]) for r in ticks], dtype=np.float64)
    mean_v_series = np.array([float(np.mean([a["v"] for a in r["agents"]])) for r in ticks])
    mean_H_series = np.array([float(np.mean([a["H"] for a in r["agents"]])) for r in ticks])
    mean_E_series = np.array([float(np.mean([a["E"] for a in r["agents"]])) for r in ticks])
    w_series = np.array([float(r["house"].get("w", 1.0)) for r in ticks])
    energy_series = np.array([float((r.get("dsp") or {}).get("energy", 0.0)) for r in ticks])

    # Per-agent aggregates
    agent_H: dict[int, list[float]] = defaultdict(list)
    agent_v: dict[int, list[float]] = defaultdict(list)
    agent_E: dict[int, list[float]] = defaultdict(list)
    agent_rest_bd: dict[int, list[int]] = defaultdict(list)
    prev_exh = {aid: False for aid in agents_meta}
    tick_exhaustion_transitions = 0

    loud_v: list[float] = []
    quiet_v: list[float] = []
    drop_front_H: list[float] = []
    drop_back_H: list[float] = []
    drop_front_E: list[float] = []
    drop_back_E: list[float] = []
    drop_top20_dh: list[float] = []
    breakdown_rest_shares: list[float] = []

    for row in ticks:
        tick_num = int(row["tick"])
        house = row.get("house") or {}
        regime = _effective_house_regime(house)
        dsp_e = float((row.get("dsp") or {}).get("energy", 0.0))
        agents = row["agents"]

        mv = float(np.mean([a["v"] for a in agents]))
        is_drop = regime == "drop"
        is_breakdown = (regime == "breakdown") or (tick_num in ref_breakdown_ticks)
        is_loud = is_drop or dsp_e >= 0.55
        is_quiet = is_breakdown or dsp_e <= 0.30

        if is_loud:
            loud_v.append(mv)
        elif is_quiet:
            quiet_v.append(mv)

        if is_drop:
            f_ag = [a for a in agents if a["y"] <= Y_FRONT_EDGE]
            b_ag = [a for a in agents if a["y"] > Y_MID_EDGE]
            if f_ag:
                drop_front_H.append(float(np.mean([a["H"] for a in f_ag])))
                drop_front_E.append(float(np.mean([a["E"] for a in f_ag])))
            if b_ag:
                drop_back_H.append(float(np.mean([a["H"] for a in b_ag])))
                drop_back_E.append(float(np.mean([a["E"] for a in b_ag])))
            top20 = sorted(agents, key=lambda x: x["v"], reverse=True)[:20]
            drop_top20_dh.append(sum(1 for a in top20 if a["archetype"] == "diehard") / 20.0)

        if is_breakdown:
            rest_share = sum(1 for a in agents if a.get("action") in ("rest", "retreat")) / max(1, len(agents))
            breakdown_rest_shares.append(rest_share)

        for a in agents:
            aid = a["id"]
            agent_H[aid].append(float(a["H"]))
            agent_v[aid].append(float(a["v"]))
            agent_E[aid].append(float(a["E"]))
            if is_breakdown:
                agent_rest_bd[aid].append(1 if a.get("action") in ("rest", "retreat") else 0)
            exh = bool(a.get("exhausted", False))
            if exh and not prev_exh[aid]:
                tick_exhaustion_transitions += 1
            prev_exh[aid] = exh

    # Ask stats per agent (for OLS)
    asks_made_by: dict[int, int] = defaultdict(int)
    asks_yes_by: dict[int, int] = defaultdict(int)
    for ask in asks:
        fid = int(ask["from"])
        asks_made_by[fid] += 1
        if ask.get("answer") == "yes":
            asks_yes_by[fid] += 1

    # Per-agent summary vectors
    agent_ids = sorted(agents_meta.keys())
    per_agent_rows: list[dict[str, Any]] = []
    for aid in agent_ids:
        am = agents_meta[aid]
        made = asks_made_by.get(aid, 0)
        succ = (asks_yes_by.get(aid, 0) / made) if made > 0 else 0.0
        rb = float(np.mean(agent_rest_bd[aid])) if agent_rest_bd[aid] else 0.0
        per_agent_rows.append(
            {
                "id": aid,
                "archetype": am["archetype"],
                "lane": int(am["lane"]),
                "k": float(am["k"]),
                "s": float(am["s"]),
                "mean_H": float(np.mean(agent_H[aid])),
                "mean_v": float(np.mean(agent_v[aid])),
                "mean_E": float(np.mean(agent_E[aid])),
                "asks_made": made,
                "ask_success_rate": succ,
                "rested_in_breakdown": rb,
            }
        )

    H_vec = np.array([r["mean_H"] for r in per_agent_rows], dtype=np.float64)
    k_vec = np.array([r["k"] for r in per_agent_rows], dtype=np.float64)
    s_vec = np.array([r["s"] for r in per_agent_rows], dtype=np.float64)
    lane_vec = np.array([r["lane"] for r in per_agent_rows], dtype=np.float64)
    lane_cent_vec = np.array([1.0 - abs(r["lane"] - 4.5) / 4.5 for r in per_agent_rows], dtype=np.float64)
    ask_sr_vec = np.array([r["ask_success_rate"] for r in per_agent_rows], dtype=np.float64)
    rest_bd_vec = np.array([r["rested_in_breakdown"] for r in per_agent_rows], dtype=np.float64)

    ols_plan = compute_ols(
        H_vec,
        [
            ("k", k_vec),
            ("s", s_vec),
            ("start_lane", lane_vec),
            ("ask_success_rate", ask_sr_vec),
            ("rested_in_breakdown", rest_bd_vec),
        ],
    )
    ols_centrality = compute_ols(
        H_vec,
        [
            ("k", k_vec),
            ("s", s_vec),
            ("lane_centrality", lane_cent_vec),
            ("ask_success_rate", ask_sr_vec),
            ("rested_in_breakdown", rest_bd_vec),
        ],
    )

    by_arch_H: dict[str, float] = {}
    by_arch_v: dict[str, float] = {}
    for arch in ("diehard", "between", "dancer"):
        sub = [r for r in per_agent_rows if r["archetype"] == arch]
        if sub:
            by_arch_H[arch] = round(float(np.mean([r["mean_H"] for r in sub])), 4)
            by_arch_v[arch] = round(float(np.mean([r["mean_v"] for r in sub])), 4)

    # Energy at drop onsets
    E_at_drops: list[float] = []
    for onset in ref_onsets:
        idx = int(np.argmin(np.abs(ts - float(onset["t"]))))
        E_at_drops.append(float(mean_E_series[idx]))

    # API telemetry (combine ticks.jsonl + run_summary.json for exact Hydrogen totals)
    tick_lats = [float((r.get("api") or {}).get("latency_ms", 0.0)) for r in ticks]
    tick_calls = sum(int((r.get("api") or {}).get("calls", 0)) for r in ticks)
    tick_fails = sum(int((r.get("api") or {}).get("failures", 0)) for r in ticks)
    tick_rls = sum(int((r.get("api") or {}).get("rate_limits", 0)) for r in ticks)
    tick_tokens = sum(int((r.get("api") or {}).get("tokens", 0)) for r in ticks)

    st = run_summary.get("stats") or {}
    api_calls_total = int(st.get("api_calls_total", tick_calls))
    api_failures = int(st.get("api_failures", tick_fails))
    api_failure_rate = (api_failures / api_calls_total) if api_calls_total > 0 else 0.0
    rate_limit_errors = int(st.get("rate_limit_errors", tick_rls))
    total_tokens = int(st.get("total_tokens", tick_tokens))
    exhaustion_events = int(run_summary.get("exhaustion_events", tick_exhaustion_transitions))

    # Anticipation lead time on shared house-ear drop onsets
    anticipation = compute_drop_lead_times(ticks, ref_onsets)
    gridlock = compute_gridlock(ticks)
    contagion = compute_spatial_contagion(ticks)
    ring_agreement = compute_ring_agreement(ticks) if cond != "shove" else {}
    asks_metrics = analyze_asks(asks) if cond == "talk" else {"asks_total": 0}

    try:
        rel_dir = str(run["dir"].resolve().relative_to(ROOT))
    except ValueError:
        rel_dir = str(run["dir"])
    return {
        "condition": cond,
        "run_dir": rel_dir,
        "seed": meta.get("seed"),
        "track": meta.get("source_label"),
        "n_ticks": len(ticks),
        "duration_s": round(float(ts[-1]), 1) if ts.size else 0.0,
        "welfare": {
            "mean_H": round(float(np.mean(H_vec)), 4),
            "median_H": round(float(np.median(H_vec)), 4),
            "std_H": round(float(np.std(H_vec)), 4),
            "min_H": round(float(np.min(H_vec)), 4),
            "max_H": round(float(np.max(H_vec)), 4),
            "gini_H": round(gini_coefficient(H_vec), 4),
            "by_archetype_mean_H": by_arch_H,
            "by_archetype_mean_v": by_arch_v,
        },
        "crowd_dynamics": {
            "mean_v_overall": round(float(np.mean(mean_v_series)), 4),
            "mean_v_loud": round(float(np.mean(loud_v)), 4) if loud_v else 0.0,
            "mean_v_quiet": round(float(np.mean(quiet_v)), 4) if quiet_v else 0.0,
            "loud_minus_quiet_v": round(
                (float(np.mean(loud_v)) - float(np.mean(quiet_v))), 4
            )
            if (loud_v and quiet_v)
            else 0.0,
            "drop_front_H": round(float(np.mean(drop_front_H)), 4) if drop_front_H else 0.0,
            "drop_back_H": round(float(np.mean(drop_back_H)), 4) if drop_back_H else 0.0,
            "drop_front_minus_back_H": round(
                float(np.mean(drop_front_H)) - float(np.mean(drop_back_H)), 4
            )
            if (drop_front_H and drop_back_H)
            else 0.0,
            "drop_front_E": round(float(np.mean(drop_front_E)), 4) if drop_front_E else 0.0,
            "drop_back_E": round(float(np.mean(drop_back_E)), 4) if drop_back_E else 0.0,
            "drop_top20_diehard_share": round(float(np.mean(drop_top20_dh)), 4)
            if drop_top20_dh
            else 0.0,
        },
        "anticipation": anticipation,
        "pacing": {
            "mean_E_overall": round(float(np.mean(mean_E_series)), 4),
            "mean_E_at_drop_onsets": round(float(np.mean(E_at_drops)), 4) if E_at_drops else 0.0,
            "min_mean_E": round(float(np.min(mean_E_series)), 4),
            "breakdown_rest_share": round(float(np.mean(breakdown_rest_shares)), 4)
            if breakdown_rest_shares
            else 0.0,
            "exhaustion_events": exhaustion_events,
            "tick_exhaustion_transitions": tick_exhaustion_transitions,
        },
        "gridlock": gridlock,
        "contagion": contagion,
        "ring_agreement": ring_agreement,
        "asks": asks_metrics,
        "ols_regression": ols_plan,
        "ols_regression_with_centrality": ols_centrality,
        "api": {
            "api_calls_total": api_calls_total,
            "api_failures": api_failures,
            "api_failure_rate": round(api_failure_rate, 5),
            "rate_limit_errors": rate_limit_errors,
            "tick_latency_p50_ms": round(float(np.median(tick_lats)), 1) if tick_lats else 0.0,
            "tick_latency_p95_ms": round(float(np.percentile(tick_lats, 95)), 1) if tick_lats else 0.0,
            "tick_latency_max_ms": round(float(np.max(tick_lats)), 1) if tick_lats else 0.0,
            "total_tokens": total_tokens,
            "hydrogen_calls": int(st.get("hydrogen_calls", 0)),
            "hydrogen_failures": int(st.get("hydrogen_failures", 0)),
        },
        # Series kept in memory for plotting (stripped before writing summary.json)
        "_series": {
            "t": ts,
            "mean_v": mean_v_series,
            "mean_H": mean_H_series,
            "mean_E": mean_E_series,
            "w": w_series,
            "energy": energy_series,
            "per_agent": per_agent_rows,
        },
    }


def plot_anticipation(
    results: dict[str, dict[str, Any]],
    spans: list[tuple[float, float, str]],
    onsets: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, (ax, ax_w) = plt.subplots(
        2, 1, figsize=(11.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.4, 1.0]}
    )
    shade_sections(ax, spans, add_legend=True)
    shade_sections(ax_w, spans, add_legend=False)

    for onset in onsets:
        ax.axvline(onset["t"], color="#b02a14", ls="--", lw=1.0, alpha=0.65)
        ax.text(
            onset["t"] + 1.5,
            0.97,
            f"drop @{int(onset['t'])}s",
            transform=ax.get_xaxis_transform(),
            fontsize=7.5,
            color="#962210",
            va="top",
        )
        ax_w.axvline(onset["t"], color="#b02a14", ls="--", lw=0.9, alpha=0.55)

    for cond in COND_ORDER:
        if cond not in results:
            continue
        r = results[cond]
        s = r["_series"]
        lead = r["anticipation"]["mean_lead_time_s"]
        ax.plot(
            s["t"],
            s["mean_v"],
            color=COND_COLORS[cond],
            lw=2.1 if cond == "talk" else 1.8,
            label=f"{COND_LABELS[cond]}  (lead {lead:+.1f}s)",
        )

    ax.set_ylabel("Mean proximity to DJ (v)")
    ax.set_title("Crowd Anticipation & Compression Across Song Structure (House Ear Sections)")
    ax.grid(True, alpha=0.55)
    ax.set_ylim(0.12, 0.72)
    ax.legend(loc="lower right", ncol=2)

    # Bottom ribbon: house ear w & audio energy from talk (or first available run)
    ref_cond = "talk" if "talk" in results else next(iter(results.keys()))
    ref_s = results[ref_cond]["_series"]
    ax_w.plot(ref_s["t"], ref_s["w"], color="#b6314c", lw=1.5, label="House ear w(t)")
    ax_w.fill_between(ref_s["t"], 0.0, ref_s["energy"] * 2.5, color="#8a3fa0", alpha=0.14, label="Audio energy (×2.5)")
    ax_w.set_ylabel("House w")
    ax_w.set_xlabel("Track time (seconds)")
    ax_w.grid(True, alpha=0.55)
    ax_w.set_ylim(0.0, 2.8)
    ax_w.legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAPER_BG)
    plt.close(fig)


def plot_welfare_fairness(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    all_h = np.concatenate(
        [[a["mean_H"] for a in results[c]["_series"]["per_agent"]] for c in COND_ORDER if c in results]
    )
    lo = max(0.0, float(np.min(all_h)) - 0.04)
    hi = min(1.0, float(np.max(all_h)) + 0.04)
    bins = np.linspace(lo, hi, 22)
    xs = np.linspace(lo, hi, 240)

    for cond in COND_ORDER:
        if cond not in results:
            continue
        r = results[cond]
        vals = np.array([a["mean_H"] for a in r["_series"]["per_agent"]], dtype=np.float64)
        col = COND_COLORS[cond]
        mh = r["welfare"]["mean_H"]
        gini = r["welfare"]["gini_H"]
        ax1.hist(vals, bins=bins, density=True, alpha=0.22, color=col, edgecolor=col, lw=0.8)
        if np.std(vals) > 1e-6:
            kde = stats.gaussian_kde(vals)
            ax1.plot(
                xs,
                kde(xs),
                color=col,
                lw=2.2,
                label=f"{cond}: mean H={mh:.3f}, Gini={gini:.3f}",
            )
        ax1.axvline(mh, color=col, ls="--", lw=1.3, alpha=0.85)

    ax1.set_xlabel("Per-agent mean happiness (H)")
    ax1.set_ylabel("Density (100 agents)")
    ax1.set_title("Welfare & Fairness Distribution Across Conditions")
    ax1.grid(True, alpha=0.55)
    ax1.legend(loc="upper left")

    # Right panel: mean H by archetype across conditions
    archs = ["diehard", "between", "dancer"]
    x = np.arange(len(archs))
    present = [c for c in COND_ORDER if c in results]
    w = 0.25
    for idx, cond in enumerate(present):
        offset = (idx - (len(present) - 1) / 2.0) * w
        vals = [results[cond]["welfare"]["by_archetype_mean_H"].get(a, 0.0) for a in archs]
        bars = ax2.bar(
            x + offset,
            vals,
            width=w * 0.92,
            color=COND_COLORS[cond],
            alpha=0.88,
            label=cond,
        )
        for b, v in zip(bars, vals):
            ax2.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.008,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Diehard\n(high k)", "Between\n(mid k/s)", "Dancer\n(high s)"])
    ax2.set_ylabel("Mean happiness (H)")
    ax2.set_ylim(0.0, max(0.75, hi + 0.06))
    ax2.set_title("Mean Happiness by Archetype")
    ax2.grid(True, axis="y", alpha=0.55)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAPER_BG)
    plt.close(fig)


def plot_ring_agreement(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    conds = [c for c in ("silent", "talk") if c in results and results[c].get("ring_agreement")]
    if not conds:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    rings = [1, 2, 3, 4, 5]
    x = np.arange(len(rings))

    for cond in conds:
        ra = results[cond]["ring_agreement"]
        col = COND_COLORS[cond]
        sec_acc = [
            (ra[str(r)]["section_agreement"] * 100.0) if ra[str(r)]["section_agreement"] is not None else np.nan
            for r in rings
        ]
        eta_acc = [
            (ra[str(r)]["drop_eta_agreement"] * 100.0) if ra[str(r)]["drop_eta_agreement"] is not None else np.nan
            for r in rings
        ]
        p_drop = [
            ra[str(r)]["mean_p_drop_on_house_drop"]
            if ra[str(r)]["mean_p_drop_on_house_drop"] is not None
            else np.nan
            for r in rings
        ]
        drop_rec = [
            (ra[str(r)]["drop_recall"] * 100.0)
            if ra[str(r)]["drop_recall"] is not None
            else np.nan
            for r in rings
        ]
        overlap = [
            (ra[str(r)]["section_prob_overlap"] * 100.0)
            if ra[str(r)]["section_prob_overlap"] is not None
            else np.nan
            for r in rings
        ]

        ax1.plot(x, sec_acc, marker="o", lw=2.0, color=col, label=f"{cond}: section match")
        ax1.plot(x, eta_acc, marker="s", lw=1.6, ls="--", color=col, alpha=0.75, label=f"{cond}: drop_eta match")

        ax2.plot(x, drop_rec, marker="o", lw=2.1, color=col, label=f"{cond}: drop recall %")
        ax2.plot(x, overlap, marker="s", lw=1.6, ls=":", color=col, alpha=0.8, label=f"{cond}: prob overlap %")
        ax2.plot(
            x,
            [p * 100.0 if not np.isnan(p) else np.nan for p in p_drop],
            marker="^",
            lw=1.5,
            ls="--",
            color=col,
            alpha=0.75,
            label=f"{cond}: P(drop) during drops",
        )

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(["Ring 1\n(front, crisp)", "Ring 2", "Ring 3", "Ring 4", "Ring 5\n(back, muffled)"])
        ax.set_ylim(0, 106)
        ax.set_ylabel("Agreement / probability (%)")
        ax.grid(True, alpha=0.55)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower left", ncol=2, fontsize=7.2)

    ax1.set_title("Ring vs House Ear: Exact Section & Drop-ETA Match")
    ax2.set_title("Acoustic Muffling Effect Across Rings 1 → 5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAPER_BG)
    plt.close(fig)


def plot_pacing(
    results: dict[str, dict[str, Any]],
    spans: list[tuple[float, float, str]],
    onsets: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    shade_sections(ax, spans)
    for onset in onsets:
        ax.axvline(onset["t"], color="#b02a14", ls="--", lw=0.9, alpha=0.55)

    for cond in COND_ORDER:
        if cond not in results:
            continue
        r = results[cond]
        s = r["_series"]
        pac = r["pacing"]
        ax.plot(
            s["t"],
            s["mean_E"],
            color=COND_COLORS[cond],
            lw=2.1 if cond == "talk" else 1.8,
            label=(
                f"{COND_LABELS[cond]}  "
                f"(drop E={pac['mean_E_at_drop_onsets']:.2f}, bd rest={pac['breakdown_rest_share']*100:.0f}%, "
                f"exh={pac['exhaustion_events']})"
            ),
        )

    ax.set_xlabel("Track time (seconds)")
    ax.set_ylabel("Crowd mean energy (E)")
    ax.set_title("Energy Pacing Over Time, Breakdown Recovery & Drop Readiness")
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower left", fontsize=8.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAPER_BG)
    plt.close(fig)


def plot_who_wins(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    present = [c for c in COND_ORDER if c in results]
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.5), sharey=True)
    if len(present) == 1:
        axes = [axes]

    for ax, cond in zip(axes, present):
        r = results[cond]
        rows = r["_series"]["per_agent"]
        for arch in ("diehard", "between", "dancer"):
            sub = [a for a in rows if a["archetype"] == arch]
            ax.scatter(
                [a["k"] for a in sub],
                [a["mean_H"] for a in sub],
                c=ARCH_COLORS[arch],
                s=34,
                alpha=0.82,
                edgecolors="#fff8f4",
                linewidths=0.45,
                label=arch,
            )

        ks = np.array([a["k"] for a in rows], dtype=np.float64)
        hs = np.array([a["mean_H"] for a in rows], dtype=np.float64)
        slope, intercept = np.polyfit(ks, hs, 1)
        x_line = np.linspace(ks.min(), ks.max(), 50)
        ax.plot(x_line, intercept + slope * x_line, color=INK, lw=1.5, ls="--", alpha=0.8)

        ols = r["ols_regression"]
        bk = ols["terms"]["k"]["coef"]
        pk = ols["terms"]["k"]["p"]
        ax.set_title(
            f"{cond.upper()}\nOLS R²={ols['r_squared']:.2f} | β_k={bk:+.3f} (p={pk:.3g})",
            fontsize=10,
        )
        ax.set_xlabel("Keenness trait (k)")
        ax.grid(True, alpha=0.55)

    axes[0].set_ylabel("Per-agent mean happiness (H)")
    axes[-1].legend(loc="lower right", title="Archetype", title_fontsize=8)
    fig.suptitle("Who Wins? Per-Agent Mean Happiness vs Keenness (k)", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PAPER_BG, bbox_inches="tight")
    plt.close(fig)


def build_comparison_table(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a crisp side-by-side comparison table across shove / silent / talk."""
    table: dict[str, dict[str, Any]] = {}
    for cond in COND_ORDER:
        if cond not in results:
            continue
        r = results[cond]
        table[cond] = {
            "mean_H": r["welfare"]["mean_H"],
            "gini_H": r["welfare"]["gini_H"],
            "mean_v_loud": r["crowd_dynamics"]["mean_v_loud"],
            "mean_v_quiet": r["crowd_dynamics"]["mean_v_quiet"],
            "loud_minus_quiet_v": r["crowd_dynamics"]["loud_minus_quiet_v"],
            "drop_front_H": r["crowd_dynamics"]["drop_front_H"],
            "drop_back_H": r["crowd_dynamics"]["drop_back_H"],
            "drop_front_E": r["crowd_dynamics"]["drop_front_E"],
            "drop_back_E": r["crowd_dynamics"]["drop_back_E"],
            "drop_top20_diehard_share": r["crowd_dynamics"]["drop_top20_diehard_share"],
            "anticipation_mean_lead_s": r["anticipation"]["mean_lead_time_s"],
            "pre_15s_v_gain": r["anticipation"]["mean_pre_15s_v_gain"],
            "mean_E_at_drop_onsets": r["pacing"]["mean_E_at_drop_onsets"],
            "breakdown_rest_share": r["pacing"]["breakdown_rest_share"],
            "exhaustion_events": r["pacing"]["exhaustion_events"],
            "gridlock_rate": r["gridlock"]["gridlock_rate"],
            "H_spatial_contagion_r": r["contagion"]["H_neighbor_corr"],
            "E_spatial_contagion_r": r["contagion"]["E_neighbor_corr"],
            "ask_acceptance_rate": r["asks"].get("ask_acceptance_rate"),
            "mean_keenness_overstatement": (r["asks"].get("keenness_truth_check") or {}).get(
                "mean_overstatement"
            ),
            "api_calls_total": r["api"]["api_calls_total"],
            "api_failures": r["api"]["api_failures"],
            "api_failure_rate": r["api"]["api_failure_rate"],
            "rate_limit_errors": r["api"]["rate_limit_errors"],
            "tick_latency_p50_ms": r["api"]["tick_latency_p50_ms"],
            "tick_latency_p95_ms": r["api"]["tick_latency_p95_ms"],
        }
    return table


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Agent Crowd runs (Phase F + H).")
    p.add_argument("run_dirs", nargs="*", help="Run directories (e.g. runs/*_shove_42 ...)")
    p.add_argument("--manifest", type=str, default="", help="Optional experiment_manifest.json")
    p.add_argument("--out", type=str, default="out", help="Output directory for PNGs & summary.json")
    args = p.parse_args()

    dirs: list[Path] = [Path(d) for d in args.run_dirs]
    if args.manifest:
        m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for cond in COND_ORDER:
            if cond in m.get("runs", {}):
                dirs.append(Path(m["runs"][cond]["run_dir"]))
    if not dirs:
        p.error("Provide run directories or --manifest")

    _style_matplotlib()
    runs_by_cond: dict[str, dict[str, Any]] = {}
    for d in dirs:
        run = load_run(d)
        runs_by_cond[run["condition"]] = run

    # Use talk (or first available) house ear as reference timeline for shared drop onsets & breakdown spans
    ref_cond = "talk" if "talk" in runs_by_cond else next(iter(runs_by_cond.keys()))
    ref_ticks = runs_by_cond[ref_cond]["ticks"]
    spans = extract_section_spans(ref_ticks)
    ref_onsets = detect_drop_onsets(ref_ticks)
    ref_breakdown_ticks = {
        int(r["tick"]) for r in ref_ticks if _effective_house_regime(r.get("house") or {}) == "breakdown"
    }

    results: dict[str, dict[str, Any]] = {}
    for cond in COND_ORDER:
        if cond in runs_by_cond:
            results[cond] = analyze_condition(runs_by_cond[cond], ref_onsets, ref_breakdown_ticks)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots (Core F + Phase H)
    plot_anticipation(results, spans, ref_onsets, out_dir / "anticipation.png")
    plot_welfare_fairness(results, out_dir / "welfare_fairness.png")
    plot_ring_agreement(results, out_dir / "ring_agreement.png")
    plot_pacing(results, spans, ref_onsets, out_dir / "pacing.png")
    plot_who_wins(results, out_dir / "who_wins.png")

    # Strip heavy numpy series before writing summary.json
    clean_conditions = {
        cond: {k: v for k, v in data.items() if k != "_series"} for cond, data in results.items()
    }
    summary = {
        "reference_house_condition": ref_cond,
        "track": runs_by_cond[ref_cond]["meta"].get("source_label"),
        "seed": runs_by_cond[ref_cond]["meta"].get("seed"),
        "house_drop_onsets": ref_onsets,
        "comparison_table": build_comparison_table(results),
        "conditions": clean_conditions,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path} and 5 plots to {out_dir}/")
    print("\n=== COMPARISON TABLE ===")
    print(json.dumps(summary["comparison_table"], indent=2))


if __name__ == "__main__":
    main()
