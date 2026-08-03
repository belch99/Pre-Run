"""Market regime classification (SPEC #16). VIX is best-effort (ticker ^VIX via
yfinance); if unavailable it's marked N/A and regime falls back to SPY/QQQ/IWM only."""
from __future__ import annotations
import pandas as pd
import numpy as np


def classify_regime(spy_df: pd.DataFrame, qqq_df: pd.DataFrame = None,
                     iwm_df: pd.DataFrame = None, vix_df: pd.DataFrame = None,
                     as_of_idx: int = -1) -> dict:
    if spy_df is None or len(spy_df) < 21:
        return {"regime": "N/A", "spy_ret_20d": np.nan, "vix_level": np.nan, "vix_status": "N/A"}

    def ret20(df):
        if df is None or len(df) < 21:
            return np.nan
        c = df["adj_close"]
        return (c.iloc[as_of_idx] - c.iloc[as_of_idx - 20]) / c.iloc[as_of_idx - 20]

    spy_ret = ret20(spy_df)
    qqq_ret = ret20(qqq_df) if qqq_df is not None else np.nan
    iwm_ret = ret20(iwm_df) if iwm_df is not None else np.nan

    vix_level, vix_status = np.nan, "N/A"
    if vix_df is not None and len(vix_df) > 0:
        vix_level = vix_df["adj_close"].iloc[as_of_idx]
        vix_status = "HIGH" if vix_level >= 25 else ("LOW" if vix_level <= 15 else "NORMAL")

    if pd.isna(spy_ret):
        regime = "N/A"
    elif vix_status == "HIGH":
        regime = "high_volatility"
    elif spy_ret > 0.03:
        regime = "bullish"
    elif spy_ret < -0.03:
        regime = "bearish"
    else:
        regime = "neutral"

    return {
        "regime": regime,
        "spy_ret_20d": spy_ret,
        "qqq_ret_20d": qqq_ret,
        "iwm_ret_20d": iwm_ret,
        "vix_level": vix_level,
        "vix_status": vix_status,
    }
