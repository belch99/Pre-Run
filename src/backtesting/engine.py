"""
Historical backtest engine (SPEC #19-#22, #70).

THE SINGLE MOST IMPORTANT RULE IN THIS FILE:
When computing a signal "as of" date T, `history_up_to_t()` returns ONLY rows
with date <= T. Every indicator/score function downstream only ever sees that
slice. Forward returns are computed SEPARATELY, afterward, from date T+1
onward, and are never fed back into the score itself. This is what prevents
look-ahead bias (SPEC #19, #61).

Definitions of a "RUN" (SPEC #21) are computed here, not assumed anywhere else.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.indicators.technical import compute_all
from src.scoring.prerun_score import compute_prerun_score
from src.data.regime import classify_regime
from src.utils.config import load_config


def history_up_to_t(df: pd.DataFrame, t_idx: int) -> pd.DataFrame:
    """Return df[0:t_idx+1] -- i.e. everything known AS OF AND INCLUDING index t_idx.
    df must already be sorted ascending by date."""
    return df.iloc[: t_idx + 1]


def run_definitions_hit(entry_price: float, future_prices: pd.Series) -> dict:
    """
    Given the close price at signal (entry_price) and the sequence of future
    closes (future_prices, index 0 = t+1, 1 = t+2, ...), compute for each
    move threshold: whether it was hit, and after how many days.
    """
    out = {}
    if entry_price is None or entry_price <= 0 or len(future_prices) == 0:
        for pct in (5, 10, 15, 20, 30):
            out[f"hit_{pct}pct"] = None
            out[f"days_to_{pct}pct"] = None
        return out

    cum_ret = (future_prices - entry_price) / entry_price
    for pct in (5, 10, 15, 20, 30):
        thresh = pct / 100
        hit_mask = cum_ret >= thresh
        if hit_mask.any():
            first_idx = np.argmax(hit_mask.values)
            out[f"hit_{pct}pct"] = 1
            out[f"days_to_{pct}pct"] = int(first_idx + 1)
        else:
            out[f"hit_{pct}pct"] = 0
            out[f"days_to_{pct}pct"] = None
    return out


def forward_outcomes(entry_price: float, df: pd.DataFrame, t_idx: int, windows=(1, 3, 5, 10, 20)) -> dict:
    """Compute forward return, MFE, MAE for each window, using ONLY data after t_idx."""
    outcomes = {}
    n = len(df)
    for w in windows:
        end_idx = t_idx + w
        if end_idx >= n:
            outcomes[w] = None  # not enough forward data yet -- honestly incomplete, not faked
            continue
        window_slice = df.iloc[t_idx + 1: end_idx + 1]
        closes = window_slice["adj_close"]
        highs = window_slice["high"]
        lows = window_slice["low"]
        fwd_return = (closes.iloc[-1] - entry_price) / entry_price
        mfe = (highs.max() - entry_price) / entry_price
        mae = (lows.min() - entry_price) / entry_price
        run_hits = run_definitions_hit(entry_price, closes)
        outcomes[w] = {
            "fwd_return": fwd_return,
            "mfe": mfe,
            "mae": mae,
            **run_hits,
        }
    return outcomes


def generate_signal_for_date(ticker: str, price_df: pd.DataFrame, t_idx: int,
                              spy_df: pd.DataFrame = None, sector_df: pd.DataFrame = None,
                              cfg: dict = None) -> dict | None:
    """
    Compute the PRE-RUN score for `ticker` AS OF price_df.iloc[t_idx] only,
    using no information beyond that row. Returns None if there isn't enough
    history to compute anything meaningful.
    """
    cfg = cfg or load_config()
    hist = history_up_to_t(price_df, t_idx)
    if len(hist) < 60:  # require at least ~3 months before we trust any indicator
        return None

    spy_hist = history_up_to_t(spy_df, min(t_idx, len(spy_df) - 1)) if spy_df is not None else None
    sector_hist = history_up_to_t(sector_df, min(t_idx, len(sector_df) - 1)) if sector_df is not None else None

    ind = compute_all(hist, spy_hist, sector_hist)
    if not ind:
        return None

    score = compute_prerun_score(ind, cfg)
    if score.get("prerun_score") is None:
        return None

    regime = classify_regime(spy_hist) if spy_hist is not None else {"regime": "N/A"}

    row = hist.iloc[-1]
    return {
        "ticker": ticker,
        "date": row["date"],
        "price": row["adj_close"],
        "indicators": ind,
        "score": score,
        "market_regime": regime.get("regime", "N/A"),
    }


def backtest_ticker(ticker: str, price_df: pd.DataFrame, spy_df: pd.DataFrame = None,
                     sector_df: pd.DataFrame = None, cfg: dict = None,
                     min_score: float = 0, start_idx: int = 60) -> pd.DataFrame:
    """
    Walk every historical date for `ticker` from start_idx to len-1, generate
    a signal at each date, compute its forward outcomes, and return one row
    per date with score + all forward-return columns. This is intentionally
    exhaustive (every day, not just "good" days) so we can later compute
    false-positive / missed-move analysis (SPEC #79, #80) from the same table.
    """
    cfg = cfg or load_config()
    price_df = price_df.sort_values("date").reset_index(drop=True)
    records = []

    for t_idx in range(start_idx, len(price_df)):
        sig = generate_signal_for_date(ticker, price_df, t_idx, spy_df, sector_df, cfg)
        if sig is None:
            continue
        score_val = sig["score"]["prerun_score"]
        if score_val is None or score_val < min_score:
            continue

        outcomes = forward_outcomes(sig["price"], price_df, t_idx, cfg["forward_windows_days"])

        rec = {
            "ticker": ticker,
            "date": sig["date"],
            "price": sig["price"],
            "prerun_score": score_val,
            "classification": sig["score"]["classification"],
            "already_running": sig["score"].get("already_running", False),
            "max_possible_pts": sig["score"]["max_possible_pts"],
            "market_regime": sig["market_regime"],
        }
        for comp, pts in sig["score"]["points"].items():
            rec[f"pts_{comp}"] = pts
        for w, out in outcomes.items():
            if out is None:
                rec[f"fwd_{w}d_return"] = None
                rec[f"fwd_{w}d_mfe"] = None
                rec[f"fwd_{w}d_mae"] = None
            else:
                rec[f"fwd_{w}d_return"] = out["fwd_return"]
                rec[f"fwd_{w}d_mfe"] = out["mfe"]
                rec[f"fwd_{w}d_mae"] = out["mae"]
                for pct in (5, 10, 15, 20, 30):
                    rec[f"fwd_{w}d_hit_{pct}pct"] = out[f"hit_{pct}pct"]
        records.append(rec)

    return pd.DataFrame(records)


def score_bucket_performance(signals_df: pd.DataFrame, run_window: int = 10, run_pct: int = 10) -> pd.DataFrame:
    """
    SPEC #22 core table: for each score bucket, hit rate / avg return / etc
    for a given (window, pct) RUN definition. Only rows where the forward
    outcome was actually computable (not None, i.e. enough real future data
    existed) are included -- signals too recent to have a resolved outcome
    are excluded rather than assumed to have failed.
    """
    df = signals_df.copy()
    hit_col = f"fwd_{run_window}d_hit_{run_pct}pct"
    ret_col = f"fwd_{run_window}d_return"
    if hit_col not in df.columns:
        raise ValueError(f"No column {hit_col}; check run_window/run_pct against forward_windows_days config")

    df = df.dropna(subset=[hit_col, ret_col])
    bins = [0, 50, 60, 70, 80, 90, 95, 101]
    labels = ["<50", "50-59", "60-69", "70-79", "80-89", "90-94", "95-100"]
    df["bucket"] = pd.cut(df["prerun_score"], bins=bins, labels=labels, right=False)

    out = df.groupby("bucket", observed=True).agg(
        signals=("ticker", "count"),
        hit_rate=(hit_col, "mean"),
        avg_return=(ret_col, "mean"),
        median_return=(ret_col, "median"),
        max_return=(ret_col, "max"),
        max_drawdown=(f"fwd_{run_window}d_mae", "min"),
    ).reset_index()
    out["hit_rate"] = (out["hit_rate"] * 100).round(1)
    for c in ["avg_return", "median_return", "max_return", "max_drawdown"]:
        out[c] = (out[c] * 100).round(2)
    return out
