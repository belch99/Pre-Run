"""
Technical indicator calculations (SPEC PHASE 2 / #6-#9).

Every function here operates on a price DataFrame that must already be
truncated to "as of date T" before being passed in -- this module does not
enforce point-in-time correctness itself, the caller (scoring/prerun_score.py)
is responsible for slicing history correctly. See backtesting/engine.py for
where that slicing happens (SPEC #19, the most important rule in the project).

All functions return NaN (not 0, not a fake default) when there isn't enough
history to compute a value, so callers can distinguish "bad" from "unknown".
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _safe_pct(a: float, b: float) -> float:
    if b in (0, None) or pd.isna(b) or pd.isna(a):
        return np.nan
    return (a - b) / b


def returns(df: pd.DataFrame, windows=(5, 10, 20, 50)) -> dict:
    close = df["adj_close"]
    out = {}
    for w in windows:
        if len(close) > w:
            out[f"ret_{w}d"] = _safe_pct(close.iloc[-1], close.iloc[-1 - w])
        else:
            out[f"ret_{w}d"] = np.nan
    return out


def distance_from_high(df: pd.DataFrame, window: int) -> float:
    if len(df) < window:
        return np.nan
    high = df["adj_close"].iloc[-window:].max()
    return _safe_pct(df["adj_close"].iloc[-1], high)


def trend_slope(df: pd.DataFrame, window: int = 20) -> float:
    """Simple linear regression slope of close price over the window, normalized by price."""
    if len(df) < window:
        return np.nan
    y = df["adj_close"].iloc[-window:].values
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return slope / y.mean() if y.mean() else np.nan


def relative_strength(df: pd.DataFrame, bench_df: pd.DataFrame, window: int = 20) -> float:
    """Stock return minus benchmark return over the window."""
    if len(df) < window + 1 or len(bench_df) < window + 1:
        return np.nan
    stock_ret = _safe_pct(df["adj_close"].iloc[-1], df["adj_close"].iloc[-1 - window])
    bench_ret = _safe_pct(bench_df["adj_close"].iloc[-1], bench_df["adj_close"].iloc[-1 - window])
    if pd.isna(stock_ret) or pd.isna(bench_ret):
        return np.nan
    return stock_ret - bench_ret


def relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return np.nan
    avg = df["volume"].iloc[-window - 1:-1].mean()
    if not avg:
        return np.nan
    return df["volume"].iloc[-1] / avg


def volume_trend_ratio(df: pd.DataFrame, short=5, long=20) -> float:
    if len(df) < long:
        return np.nan
    avg_short = df["volume"].iloc[-short:].mean()
    avg_long = df["volume"].iloc[-long:].mean()
    return avg_short / avg_long if avg_long else np.nan


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["adj_close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    high, low, close, vol = df["high"], df["low"], df["adj_close"], df["volume"]
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfm = mfm.fillna(0)
    return (mfm * vol).cumsum()


def atr(df: pd.DataFrame, window: int = 14) -> float:
    if len(df) < window + 1:
        return np.nan
    high, low, close = df["high"], df["low"], df["adj_close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.iloc[-window:].mean()


def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    a = atr(df, window)
    last_close = df["adj_close"].iloc[-1]
    return a / last_close if last_close and not pd.isna(a) else np.nan


def bollinger_band_width(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> float:
    if len(df) < window:
        return np.nan
    close = df["adj_close"].iloc[-window:]
    mean, std = close.mean(), close.std()
    if not mean:
        return np.nan
    return (2 * n_std * std) / mean


def historical_volatility(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return np.nan
    log_ret = np.log(df["adj_close"] / df["adj_close"].shift(1)).dropna()
    return log_ret.iloc[-window:].std() * np.sqrt(252)


def range_compression_ratio(df: pd.DataFrame, short=5, long=20) -> float:
    """5-day high-low range vs 20-day high-low range. <1 = compressing."""
    if len(df) < long:
        return np.nan
    r5 = (df["high"].iloc[-short:].max() - df["low"].iloc[-short:].min())
    r20 = (df["high"].iloc[-long:].max() - df["low"].iloc[-long:].min())
    return r5 / r20 if r20 else np.nan


def inside_day_streak(df: pd.DataFrame) -> int:
    """Count consecutive most-recent days where high/low is inside the prior day's range."""
    streak = 0
    for i in range(len(df) - 1, 0, -1):
        if df["high"].iloc[i] <= df["high"].iloc[i - 1] and df["low"].iloc[i] >= df["low"].iloc[i - 1]:
            streak += 1
        else:
            break
    return streak


def resistance_levels(df: pd.DataFrame) -> dict:
    out = {}
    for label, window in [("resistance_20d", 20), ("resistance_50d", 50), ("resistance_52w", 252)]:
        if len(df) >= window:
            out[label] = df["adj_close"].iloc[-window:].max()
        else:
            out[label] = np.nan
    if len(df) >= 20:
        out["support_20d"] = df["adj_close"].iloc[-20:].min()
    else:
        out["support_20d"] = np.nan
    return out


def distance_to_breakout(last_price: float, resistance: float) -> float:
    if pd.isna(resistance) or resistance <= 0:
        return np.nan
    return (resistance - last_price) / resistance
    def ema(series: pd.Series, span: int) -> float:
    if len(series) < span:
        return np.nan
    return series.ewm(span=span, adjust=False).mean().iloc[-1]


def ema_trend_alignment(df: pd.DataFrame) -> dict:
    close = df["adj_close"]
    last = close.iloc[-1]
    e8, e21, e50, e200 = ema(close, 8), ema(close, 21), ema(close, 50), ema(close, 200)

    above_8 = bool(last > e8) if not pd.isna(e8) else None
    above_21 = bool(last > e21) if not pd.isna(e21) else None
    above_50 = bool(last > e50) if not pd.isna(e50) else None
    above_200 = bool(last > e200) if not pd.isna(e200) else None

    stacked = None
    if all(v is not None for v in [e8, e21, e50, e200]) and not any(pd.isna(v) for v in [e8, e21, e50, e200]):
        stacked = bool(e8 > e21 > e50 > e200) and bool(above_8)

    return {
        "ema_8": e8, "ema_21": e21, "ema_50": e50, "ema_200": e200,
        "above_ema_8": above_8, "above_ema_21": above_21,
        "above_ema_50": above_50, "above_ema_200": above_200,
        "stacked_uptrend": stacked,
    }


def compute_all(df: pd.DataFrame, spy_df: pd.DataFrame = None, sector_df: pd.DataFrame = None) -> dict:
    """
    Compute the full indicator set for the LAST ROW of df (i.e. 'as of' the
    final date present). Caller must have already truncated df to avoid
    look-ahead bias.
    """
    if df is None or len(df) < 5:
        return {}

    out = {}
    out.update(returns(df))
    out["dist_from_20d_high"] = distance_from_high(df, 20)
    out["dist_from_52w_high"] = distance_from_high(df, 252)
    out["trend_slope_20d"] = trend_slope(df, 20)

    if spy_df is not None and len(spy_df) > 0:
        out["rs_vs_spy_20d"] = relative_strength(df, spy_df, 20)
    else:
        out["rs_vs_spy_20d"] = np.nan

    if sector_df is not None and len(sector_df) > 0:
        out["rs_vs_sector_20d"] = relative_strength(df, sector_df, 20)
    else:
        out["rs_vs_sector_20d"] = np.nan

    out["rvol_20d"] = relative_volume(df, 20)
    out["rvol_50d"] = relative_volume(df, 50)
    out["vol5_vs_vol20"] = volume_trend_ratio(df, 5, 20)
    out["obv"] = obv(df).iloc[-1] if len(df) > 1 else np.nan
    out["ad_line"] = accumulation_distribution(df).iloc[-1] if len(df) > 1 else np.nan

    out["atr_14"] = atr(df, 14)
    out["atr_pct"] = atr_pct(df, 14)
    out["bb_width"] = bollinger_band_width(df, 20)
    out["hv_20d"] = historical_volatility(df, 20)
    out["range5_vs_range20"] = range_compression_ratio(df, 5, 20)
    out["inside_day_streak"] = inside_day_streak(df)

    out.update(resistance_levels(df))
    last_price = df["adj_close"].iloc[-1]
    out["distance_to_breakout_pct"] = distance_to_breakout(last_price, out.get("resistance_20d"))
    out["trigger_price"] = out.get("resistance_20d")

    out.update(ema_trend_alignment(df))

    return out
