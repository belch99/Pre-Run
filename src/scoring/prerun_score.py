"""
PRE-RUN score computation (SPEC #6-#19, #45-#47).

Phase 1-6 reality check:
Only momentum, volume, and volatility-compression + breakout-proximity have a
free, reliable, point-in-time data source wired up right now (price/volume
history). Catalyst, options activity, short interest, insider activity, and
news-attention (55 of the 100 raw points) require Phase 7-9 data sources that
are NOT yet connected (see src/catalysts, src/data/sec.py stubs).

Per SPEC #3 / #96 ("do not fake results", "never assign a fake score"), those
components are marked N/A and EXCLUDED from both the numerator and the
denominator, rather than defaulted to 0 or to some assumed "neutral" value.
The score is then rescaled to /100 using only the points that were actually
computed, and `max_possible_pts` records how much of the model was actually
active for that row, so a 62/62-available is NOT displayed the same as a
62/100-nominal. The dashboard must show both.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.utils.config import load_config

NA_COMPONENTS_PHASE1 = {"catalyst", "options_activity", "short_interest", "insider_activity", "news_attention"}


def _clip(x, lo=0, hi=1):
    if pd.isna(x):
        return np.nan
    return max(lo, min(hi, x))


def score_momentum(ind: dict, weight: float) -> tuple[float, str]:
    """
    Reward: positive but NOT extreme 5/10/20d returns, higher highs/lows,
    positive relative strength vs SPY/sector.
    Penalize: already-large recent expansion (SPEC #6 explicitly wants this).
    """
    r5, r10, r20 = ind.get("ret_5d"), ind.get("ret_10d"), ind.get("ret_20d")
    rs_spy, rs_sector = ind.get("rs_vs_spy_20d"), ind.get("rs_vs_sector_20d")
    slope = ind.get("trend_slope_20d")

    if all(pd.isna(v) for v in [r5, r10, r20]):
        return np.nan, "insufficient price history"

    # sweet spot: modest positive momentum, not a blow-off top
    def score_return(r, ideal=0.06, penalty_start=0.15):
        if pd.isna(r):
            return np.nan
        if r < -0.05:
            return 0.1
        if r <= ideal:
            return _clip(r / ideal) * 0.8 + 0.2
        if r <= penalty_start:
            return 1.0
        # penalize expansion beyond penalty_start
        return max(0.1, 1.0 - (r - penalty_start) * 3)

    parts = [p for p in [score_return(r5, 0.04, 0.10),
                          score_return(r10, 0.06, 0.15),
                          score_return(r20, 0.08, 0.25)] if not pd.isna(p)]
    base = np.mean(parts) if parts else np.nan

    rs_bonus = 0.0
    rs_vals = [v for v in [rs_spy, rs_sector] if not pd.isna(v)]
    if rs_vals:
        rs_bonus = _clip(np.mean(rs_vals) / 0.10, 0, 1) * 0.2

    slope_bonus = _clip(slope * 50, 0, 1) * 0.1 if not pd.isna(slope) else 0

    if pd.isna(base):
        return np.nan, "insufficient price history"

    frac = _clip(base * 0.7 + rs_bonus + slope_bonus)
    reason = f"5d/10d/20d returns {r5:.1%}/{r10:.1%}/{r20:.1%}" if not any(pd.isna(v) for v in [r5, r10, r20]) else "partial momentum data"
    return frac * weight, reason


def score_volume(ind: dict, weight: float) -> tuple[float, str]:
    rvol20 = ind.get("rvol_20d")
    vol5v20 = ind.get("vol5_vs_vol20")
    if pd.isna(rvol20) and pd.isna(vol5v20):
        return np.nan, "insufficient volume history"

    parts = []
    if not pd.isna(rvol20):
        # reward rising participation without requiring a single huge spike
        parts.append(_clip((rvol20 - 0.8) / 1.2))
    if not pd.isna(vol5v20):
        # quiet accumulation = 5d avg vol modestly above 20d avg, not a 1-day spike
        parts.append(_clip((vol5v20 - 0.9) / 0.8))

    frac = np.mean(parts) if parts else np.nan
    if pd.isna(frac):
        return np.nan, "insufficient volume history"
    reason = f"rel. volume {rvol20:.2f}x (20d), 5d/20d avg-volume ratio {vol5v20:.2f}" if not pd.isna(rvol20) and not pd.isna(vol5v20) else "partial volume data"
    return _clip(frac) * weight, reason


def score_compression(ind: dict, weight: float) -> tuple[float, str]:
    range_ratio = ind.get("range5_vs_range20")
    bb_width = ind.get("bb_width")
    inside_streak = ind.get("inside_day_streak", 0)

    if pd.isna(range_ratio) and pd.isna(bb_width):
        return np.nan, "insufficient volatility history"

    parts = []
    if not pd.isna(range_ratio):
        # lower ratio = more compressed = higher score
        parts.append(_clip(1 - range_ratio))
    if not pd.isna(bb_width):
        # narrower bands = higher score; normalize against a typical 0.08-0.20 range
        parts.append(_clip((0.20 - bb_width) / 0.15))
    streak_bonus = _clip(inside_streak / 5) * 0.15

    frac = (np.mean(parts) if parts else 0) * 0.85 + streak_bonus
    reason = f"5d/20d range ratio {range_ratio:.2f}, {inside_streak} inside-day streak" if not pd.isna(range_ratio) else "partial compression data"
    return _clip(frac) * weight, reason


def score_breakout_proximity(ind: dict, weight: float) -> tuple[float, str]:
    dist = ind.get("distance_to_breakout_pct")
    if pd.isna(dist):
        return np.nan, "no resistance level computable"
    if dist < 0:
        # already broke out
        frac = _clip(1 + dist * 3)  # decays fast the further past resistance
        reason = f"already {abs(dist):.1%} past 20d resistance"
    else:
        # closer to breakout = higher score, cap benefit inside ~10%
        frac = _clip(1 - dist / 0.10)
        reason = f"{dist:.1%} below 20d resistance (trigger ${ind.get('trigger_price', float('nan')):.2f})"
    return frac * weight, reason


COMPONENT_FUNCS = {
    "momentum_structure": score_momentum,
    "volume_accumulation": score_volume,
    "volatility_compression": score_compression,
    "breakout_proximity": score_breakout_proximity,
}


def compute_prerun_score(ind: dict, cfg: dict = None) -> dict:
    """
    Returns a dict with prerun_score (rescaled to /100 of AVAILABLE points),
    max_possible_pts (how much of the model was actually active), per-component
    points, classification, and a plain-English explanation (SPEC #45).
    """
    cfg = cfg or load_config()
    weights = cfg["weights"]
    na_components = set(cfg.get("unavailable_in_phase1", NA_COMPONENTS_PHASE1))

    points = {}
    reasons = []
    available_weight = 0.0
    earned = 0.0

    for comp, weight in weights.items():
        if comp in na_components:
            points[comp] = None  # explicit N/A, never faked
            continue
        func = COMPONENT_FUNCS.get(comp)
        if func is None:
            points[comp] = None
            continue
        pts, reason = func(ind, weight)
        if pd.isna(pts):
            points[comp] = None
            continue
        points[comp] = round(pts, 2)
        available_weight += weight
        earned += pts
        reasons.append(reason)

    if available_weight == 0:
        return {
            "prerun_score": None,
            "max_possible_pts": 0,
            "points": points,
            "classification": "N/A",
            "explanation": "Insufficient data to compute any score component.",
        }

    prerun_score = round(earned / available_weight * 100, 1)
    classification = classify(prerun_score, cfg)

    already_running = is_already_running(ind, cfg)

    explanation = build_explanation(prerun_score, points, reasons, already_running)

    return {
        "prerun_score": prerun_score,
        "max_possible_pts": available_weight,   # out of 100 nominal
        "points": points,
        "classification": classification,
        "already_running": already_running,
        "explanation": explanation,
    }


def classify(score: float, cfg: dict = None) -> str:
    cfg = cfg or load_config()
    bands = cfg["classification_bands"]
    for label, (lo, hi) in bands.items():
        if lo <= score <= hi:
            return label
    return "NO_SETUP"


def is_already_running(ind: dict, cfg: dict = None) -> bool:
    cfg = cfg or load_config()
    thresh = cfg["already_running"]
    r3 = ind.get("ret_5d")  # closest available proxy if ret_3d not computed
    r5 = ind.get("ret_5d")
    if not pd.isna(r5) and r5 >= thresh["return_5d_threshold"]:
        return True
    return False


def build_explanation(score, points, reasons, already_running) -> str:
    lines = [f"PRE-RUN {score}" + (" (already showing a large recent move)" if already_running else "")]
    for comp, pts in points.items():
        label = comp.replace("_", " ").title()
        if pts is None:
            lines.append(f"  {label}: N/A (data source not connected in this phase)")
        else:
            lines.append(f"  {label}: {pts:+.1f}")
    if reasons:
        lines.append("Why: " + "; ".join(reasons) + ".")
    return "\n".join(lines)
