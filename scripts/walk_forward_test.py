"""
Walk-forward validation (SPEC #27). Run AFTER run_first_experiment.py has
populated daily_prices for the universe.

Splits: train 2019-2022 -> test 2023; train 2019-2023 -> test 2024;
train 2019-2024 -> test 2025; then 2026-forward is the live/current period.
Adjust YEAR_SPLITS below to match however much real history you actually have
(free yfinance/stooq daily history commonly goes back further, but your
universe's IPO dates will vary -- small/microcaps often have much less).

This script does NOT retune any weights automatically. Its only job is to
report out-of-sample hit rates per split so you can see whether the *fixed*
v1.0 model holds up across different market periods, per SPEC #27.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import load_config, db_path
from src.data.prices import load_prices
from src.data.db import connect
from src.backtesting.engine import backtest_ticker, score_bucket_performance

YEAR_SPLITS = [
    {"train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"train": ("2019-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"train": ("2019-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    {"train": ("2019-01-01", "2025-12-31"), "test": ("2026-01-01", "2026-12-31")},
]


def main():
    cfg = load_config()
    dbp = db_path(cfg)
    universe = cfg["universe"]["seed_tickers"]
    spy_df = load_prices(cfg["benchmarks"]["market"], dbp)

    all_signals = []
    for t in universe:
        df = load_prices(t, dbp)
        if len(df) < 80:
            continue
        res = backtest_ticker(t, df, spy_df=spy_df, cfg=cfg, min_score=0, start_idx=60)
        all_signals.append(res)

    if not all_signals:
        print("No ticker had enough price history in the DB. Run run_first_experiment.py first.")
        return

    signals_df = pd.concat(all_signals, ignore_index=True)

    for split in YEAR_SPLITS:
        test_start, test_end = split["test"]
        test_slice = signals_df[(signals_df["date"] >= test_start) & (signals_df["date"] <= test_end)]
        label = f"TEST {test_start} to {test_end}"
        print(f"\n=== {label} ===")
        if test_slice.empty:
            print("  No signals in this window (likely not enough price history yet).")
            continue
        try:
            table = score_bucket_performance(test_slice, run_window=10, run_pct=10)
            print(table.to_string(index=False))
        except Exception as e:
            print(f"  Could not compute bucket table: {e}")

        with connect(dbp) as conn:
            conn.execute(
                """INSERT INTO backtest_runs
                   (model_version, config_snapshot, start_date, end_date, universe_mode,
                    dataset_split, n_signals, results_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    cfg["model_version"], json.dumps(cfg), test_start, test_end,
                    cfg["universe"]["mode"], "OUT_OF_SAMPLE", len(test_slice),
                    json.dumps({"bucket_table": table.to_dict(orient="records") if not test_slice.empty else []}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    print("\nDone. Check the Backtest dashboard page and filter to dataset_split=OUT_OF_SAMPLE "
          "before trusting any pattern (SPEC #27, #93).")


if __name__ == "__main__":
    main()
