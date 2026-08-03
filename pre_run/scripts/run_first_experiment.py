"""
FIRST MILESTONE (SPEC #87): answer

  "Show me the 20 stocks that had the strongest PRE-RUN characteristics at
   the end of yesterday's session, and what happened to them over the
   following 1/3/5/10/20 trading days."

Run this on a machine with normal internet access:

    cd pre_run
    pip install -r requirements.txt
    python scripts/run_first_experiment.py

What it does:
 1. Ingests price history for the seed universe (config.yaml universe.seed_tickers)
    + SPY/QQQ/IWM benchmarks, caching to SQLite.
 2. Walks every historical date and computes a point-in-time PRE-RUN score.
 3. Computes true forward outcomes (no look-ahead) for 1/3/5/10/20 days.
 4. Builds the score-bucket hit-rate table (SPEC #22).
 5. Stores the run in `backtest_runs` so it shows up in the dashboard.
 6. Prints the answer to the first-milestone question directly to console.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import load_config, db_path, get_logger
from src.data.prices import ingest_universe, load_prices
from src.data.db import connect, init_db
from src.backtesting.engine import backtest_ticker, score_bucket_performance

logger = get_logger("scripts.first_experiment")


def main():
    cfg = load_config()
    dbp = db_path(cfg)
    init_db(dbp)

    universe = cfg["universe"]["seed_tickers"]
    benchmarks = [cfg["benchmarks"]["market"], cfg["benchmarks"]["tech_growth"], cfg["benchmarks"]["small_cap"]]
    all_tickers = sorted(set(universe) | set(benchmarks))

    print(f"[1/5] Ingesting {len(all_tickers)} tickers (prices + benchmarks)...")
    ingest_result = ingest_universe(all_tickers)
    n_ok = (ingest_result["status"].isin(["OK", "CACHED"])).sum()
    print(f"      {n_ok}/{len(all_tickers)} tickers ingested successfully.")
    if n_ok == 0:
        print("\nNo price data could be ingested. This almost always means the machine "
              "running this script has no internet access to query1/query2.finance.yahoo.com "
              "or stooq.com. Check your network and try again.")
        return

    spy_df = load_prices(cfg["benchmarks"]["market"], dbp)
    qqq_df = load_prices(cfg["benchmarks"]["tech_growth"], dbp)

    print("[2/5] Running point-in-time backtest per ticker (no look-ahead)...")
    all_results = []
    for t in universe:
        df = load_prices(t, dbp)
        if len(df) < 80:
            print(f"      skipping {t}: only {len(df)} rows of history")
            continue
        res = backtest_ticker(t, df, spy_df=spy_df, sector_df=None, cfg=cfg, min_score=0, start_idx=60)
        all_results.append(res)
        print(f"      {t}: {len(res)} historical signals scored")

    if not all_results:
        print("No tickers had enough history to backtest.")
        return

    signals_df = pd.concat(all_results, ignore_index=True)

    print("[3/5] Building score-bucket hit-rate table (10% move within 10 trading days)...")
    bucket_table = score_bucket_performance(signals_df, run_window=10, run_pct=10)
    print(bucket_table.to_string(index=False))

    print("[4/5] Recording backtest_runs entry...")
    with connect(dbp) as conn:
        conn.execute(
            """INSERT INTO backtest_runs
               (model_version, config_snapshot, start_date, end_date, universe_mode,
                dataset_split, n_signals, results_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                cfg["model_version"], json.dumps(cfg), signals_df["date"].min(), signals_df["date"].max(),
                cfg["universe"]["mode"], "FULL_HISTORY_UNSPLIT", len(signals_df),
                json.dumps({"bucket_table": bucket_table.to_dict(orient="records")}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    print("\n[5/5] FIRST MILESTONE ANSWER")
    latest_date = signals_df["date"].max()
    top20 = signals_df[signals_df["date"] == latest_date].sort_values("prerun_score", ascending=False).head(20)
    if top20.empty:
        print(f"No signals were scored on {latest_date} (may be a data edge). "
              f"Inspect signals_df directly for the actual latest scored date per ticker.")
    else:
        print(f"Top PRE-RUN candidates as of {latest_date}:")
        print(top20[["ticker", "prerun_score", "classification", "fwd_5d_return", "fwd_10d_return"]]
              .to_string(index=False))

    print(
        "\nNOTE: dataset_split is FULL_HISTORY_UNSPLIT for this first run -- it has NOT "
        "been through train/validation/out-of-sample partitioning yet (SPEC #27). "
        "Treat this run as a sanity check that the pipeline works, not as proof the "
        "model has predictive value. Run scripts/walk_forward_test.py next."
    )


if __name__ == "__main__":
    main()
