"""
Daily after-close workflow (SPEC #25, #81, #82).

Run this once per trading day after the close, on a machine with internet
access (cron/Task Scheduler/launchd -- see README "Scheduling").

Steps (matches SPEC #25 exactly):
 1. Refresh price data for the universe (cached; only re-fetches if stale).
 2. Compute today's PRE-RUN score for every ticker, AS OF TODAY ONLY.
 3. Freeze + timestamp + store each qualifying signal (score >= threshold) --
    once written, signals.frozen=1 and nothing about that row is ever edited.
 4. Update existing open paper trades against today's OHLCV.
 5. Create new paper trades for today's new signals.
 6. Write a daily_snapshot row so today's full universe state can be
    reconstructed later even for tickers that weren't flagged (SPEC #42, #51).
 7. Print a plain-text morning-report-style summary (SPEC #82).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import load_config, db_path, get_logger
from src.data.prices import ingest_universe, load_prices
from src.data.db import connect, init_db
from src.backtesting.engine import generate_signal_for_date
from src.paper_trading.engine import create_paper_trade, update_trade_with_new_price

logger = get_logger("scripts.daily_scan")


def main():
    cfg = load_config()
    dbp = db_path(cfg)
    init_db(dbp)

    universe = cfg["universe"]["seed_tickers"]
    benchmarks = [cfg["benchmarks"]["market"], cfg["benchmarks"]["tech_growth"], cfg["benchmarks"]["small_cap"]]
    all_tickers = sorted(set(universe) | set(benchmarks))

    print(f"[1/6] Refreshing price data for {len(all_tickers)} tickers...")
    ingest_universe(all_tickers)

    spy_df = load_prices(cfg["benchmarks"]["market"], dbp)

    print("[2/6] Scoring universe as of today...")
    today_signals = []
    for t in universe:
        df = load_prices(t, dbp)
        if len(df) < 60:
            continue
        sig = generate_signal_for_date(t, df, len(df) - 1, spy_df=spy_df, cfg=cfg)
        if sig:
            today_signals.append(sig)

    with connect(dbp) as conn:
        now = datetime.now(timezone.utc).isoformat()
        for sig in today_signals:
            score = sig["score"]
            conn.execute(
                """INSERT INTO scores
                   (ticker, date, prerun_score, momentum_pts, volume_pts, compression_pts,
                    breakout_pts, catalyst_pts, options_pts, short_interest_pts, insider_pts,
                    news_pts, max_possible_pts, classification, already_running, universe_mode,
                    market_regime, model_version, explanation, computed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(ticker, date, model_version) DO UPDATE SET
                     prerun_score=excluded.prerun_score, explanation=excluded.explanation,
                     computed_at=excluded.computed_at""",
                (
                    sig["ticker"], sig["date"], score["prerun_score"],
                    score["points"].get("momentum_structure"), score["points"].get("volume_accumulation"),
                    score["points"].get("volatility_compression"), score["points"].get("breakout_proximity"),
                    score["points"].get("catalyst"), score["points"].get("options_activity"),
                    score["points"].get("short_interest"), score["points"].get("insider_activity"),
                    score["points"].get("news_attention"), score["max_possible_pts"],
                    score["classification"], int(score.get("already_running", False)),
                    cfg["universe"]["mode"], sig["market_regime"], cfg["model_version"],
                    score["explanation"], now,
                ),
            )
        print(f"      {len(today_signals)} tickers scored.")

        print("[3/6] Freezing qualifying signals (score >= breakout.prerun_min_score)...")
        threshold = cfg["breakout"]["prerun_min_score"]
        qualifying = [s for s in today_signals if s["score"]["prerun_score"] and s["score"]["prerun_score"] >= threshold]
        for sig in qualifying:
            ind = sig["indicators"]
            trigger = ind.get("trigger_price")
            atr = ind.get("atr_14") or 0
            stop = sig["price"] - 1.5 * atr if atr else sig["price"] * 0.95
            target1 = trigger * 1.05 if trigger else None
            target2 = trigger * 1.10 if trigger else None
            conn.execute(
                """INSERT INTO signals
                   (ticker, signal_date, signal_time, price_at_signal, prerun_score,
                    trigger_price, stop_price, target1, target2, setup_type, market_regime,
                    sector, catalyst_summary, model_version, frozen, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)""",
                (
                    sig["ticker"], sig["date"], now, sig["price"], sig["score"]["prerun_score"],
                    trigger, stop, target1, target2, "breakout", sig["market_regime"],
                    None, None, cfg["model_version"], now,
                ),
            )
        print(f"      {len(qualifying)} signals frozen.")

        print("[4/6] Updating open paper trades with today's OHLCV...")
        open_trades = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE status IN ('WAITING','ENTERED')", conn
        )
        updated = 0
        for _, tr in open_trades.iterrows():
            px = load_prices(tr["ticker"], dbp, start=sig["date"] if today_signals else None)
            if px.empty:
                continue
            last = px.iloc[-1]
            new_trade = update_trade_with_new_price(
                tr.to_dict(), last["date"], last["high"], last["low"], last["close"],
                last["volume"], px["volume"].tail(20).mean(), cfg,
            )
            if new_trade != tr.to_dict():
                cols = ", ".join(f"{k}=?" for k in new_trade if k != "trade_id")
                vals = [v for k, v in new_trade.items() if k != "trade_id"]
                conn.execute(f"UPDATE paper_trades SET {cols} WHERE trade_id=?",
                             vals + [tr["trade_id"]])
                updated += 1
        print(f"      {updated} trades updated.")

        print("[5/6] Creating new paper trades for today's fresh signals...")
        new_trades = 0
        for sig in qualifying:
            trade = create_paper_trade({
                "ticker": sig["ticker"], "signal_date": sig["date"], "signal_time": now,
                "price": sig["price"], "trigger_price": ind.get("trigger_price"),
                "stop_price": sig["price"] * 0.95, "target1": None, "target2": None,
                "score": sig["score"]["prerun_score"], "score_components": sig["score"]["points"],
                "market_regime": sig["market_regime"], "model_version": cfg["model_version"],
            }, cfg=cfg)
            conn.execute(
                """INSERT INTO paper_trades
                   (signal_id, ticker, signal_date, signal_time, entry_price, trigger_price,
                    stop_price, target1, target2, shares, capital_allocated, dollar_risk,
                    pct_risk, account_size, status, score, score_components, market_regime,
                    model_version, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    None, trade["ticker"], trade["signal_date"], trade["signal_time"],
                    trade["entry_price"], trade["trigger_price"], trade["stop_price"],
                    trade["target1"], trade["target2"], trade["shares"], trade["capital_allocated"],
                    trade["dollar_risk"], trade["pct_risk"], trade["account_size"], trade["status"],
                    trade["score"], trade["score_components"], trade["market_regime"],
                    trade["model_version"], trade["created_at"], trade["updated_at"],
                ),
            )
            new_trades += 1
        print(f"      {new_trades} new paper trades created.")

        print("[6/6] Writing daily snapshot...")
        top10 = sorted(today_signals, key=lambda s: s["score"]["prerun_score"] or 0, reverse=True)[:10]
        conn.execute(
            """INSERT INTO daily_snapshots
               (snapshot_date, n_tickers_scanned, n_qualifying_signals, market_regime,
                top_candidates_json, data_health_json, model_version, created_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(snapshot_date) DO UPDATE SET
                 n_tickers_scanned=excluded.n_tickers_scanned,
                 n_qualifying_signals=excluded.n_qualifying_signals,
                 top_candidates_json=excluded.top_candidates_json""",
            (
                today_signals[0]["date"] if today_signals else datetime.now().strftime("%Y-%m-%d"),
                len(today_signals), len(qualifying),
                today_signals[0]["market_regime"] if today_signals else "N/A",
                json.dumps([{"ticker": s["ticker"], "score": s["score"]["prerun_score"]} for s in top10]),
                json.dumps({}), cfg["model_version"], now,
            ),
        )

    print("\n=== PRE-RUN MORNING REPORT ===")
    for s in top10:
        print(f"  {s['ticker']:6s}  {s['score']['prerun_score']:.1f}  {s['score']['classification']}")


if __name__ == "__main__":
    main()
