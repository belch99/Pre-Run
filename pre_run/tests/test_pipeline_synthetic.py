"""
IMPORTANT: This test uses SYNTHETIC, randomly generated price data. It exists
ONLY to verify the pipeline runs end-to-end without crashing and produces
internally consistent output (e.g. forward returns math checks out).

It proves NOTHING about whether PRE-RUN has real predictive value. That
question can only be answered with real historical market data ingested via
src/data/prices.py on a machine with normal internet access -- Anthropic's
sandboxed dev environment used to build this project cannot reach
finance.yahoo.com or stooq.com, so real ingestion was not possible here.
Run `python scripts/run_first_experiment.py` on your own machine to get the
real answer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtesting.engine import backtest_ticker, score_bucket_performance
from src.scoring.prerun_score import compute_prerun_score, classify
from src.indicators.technical import compute_all


def make_synthetic_ohlcv(n=400, seed=42, drift=0.0004, vol=0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 20 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2023-01-01", periods=n).strftime("%Y-%m-%d")
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = np.abs(rng.normal(1_000_000, 300_000, n)) * (1 + np.abs(rets) * 10)
    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "adj_close": close, "volume": volume.astype(int),
    })
    return df


def test_indicators_run_without_crash():
    df = make_synthetic_ohlcv()
    ind = compute_all(df.iloc[:100])
    assert "ret_20d" in ind
    assert not pd.isna(ind["ret_5d"])
    print("indicators OK:", {k: round(v, 4) if isinstance(v, float) else v for k, v in list(ind.items())[:6]})


def test_score_has_na_for_unwired_components():
    df = make_synthetic_ohlcv()
    ind = compute_all(df.iloc[:100])
    score = compute_prerun_score(ind)
    assert score["points"]["catalyst"] is None, "catalyst should be N/A in Phase 1-6, not faked"
    assert score["points"]["options_activity"] is None
    assert 0 <= score["prerun_score"] <= 100
    print("score OK:", score["prerun_score"], score["classification"])
    print(score["explanation"])


def test_no_lookahead_forward_returns_use_only_future_rows():
    """Explicitly checks: outcome computed at t_idx never uses rows before t_idx+1."""
    df = make_synthetic_ohlcv(n=200)
    from src.backtesting.engine import forward_outcomes
    t_idx = 100
    entry_price = df["adj_close"].iloc[t_idx]
    out = forward_outcomes(entry_price, df, t_idx, windows=(1, 5))
    manual_5d_close = df["adj_close"].iloc[t_idx + 5]
    manual_5d_ret = (manual_5d_close - entry_price) / entry_price
    assert abs(out[5]["fwd_return"] - manual_5d_ret) < 1e-9
    print("forward return math OK:", out[5]["fwd_return"], "vs manual", manual_5d_ret)


def test_full_backtest_pipeline_runs():
    df = make_synthetic_ohlcv(n=400, seed=7)
    spy = make_synthetic_ohlcv(n=400, seed=1, drift=0.0002, vol=0.012)
    result = backtest_ticker("SYN1", df, spy_df=spy, min_score=0, start_idx=60)
    assert len(result) > 0
    print(f"backtest produced {len(result)} rows, columns: {len(result.columns)}")
    bucket_perf = score_bucket_performance(result, run_window=10, run_pct=10)
    print(bucket_perf.to_string(index=False))


if __name__ == "__main__":
    test_indicators_run_without_crash()
    print()
    test_score_has_na_for_unwired_components()
    print()
    test_no_lookahead_forward_returns_use_only_future_rows()
    print()
    test_full_backtest_pipeline_runs()
    print("\nAll synthetic pipeline tests passed.")
