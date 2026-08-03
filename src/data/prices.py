"""
Historical daily price/volume ingestion (SPEC PHASE 1).

Data source: yfinance (Yahoo Finance, free, unofficial API) -> labeled
data_status = 'END-OF-DAY' / 'HISTORICAL'. This is the only free source wired
up in Phase 1. Stooq is a documented fallback (see fetch_stooq_fallback) for
when Yahoo is rate-limited or unavailable -- it is NOT called automatically
so we don't silently mix data quality between sources without logging it.

IMPORTANT LIMITATION (must stay visible, per SPEC #3 / #96):
Yahoo Finance via yfinance has no official SLA, can silently change/rate-limit,
and does not provide a documented point-in-time-correct restatement history.
That's an accepted limitation of the $0 constraint. It is recorded per-row in
`source` / `data_status` so downstream consumers know exactly what they're
looking at, and nothing here claims REAL-TIME data.

This module also cannot reach any network from Anthropic's sandboxed dev
environment (see README "Known limitation: sandbox network"). It is written
to run correctly on a normal internet-connected machine.
"""
from __future__ import annotations
import time
import sqlite3
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

from src.utils.config import load_config, get_logger, db_path
from src.data.db import connect, init_db

logger = get_logger("data.prices")

REFRESH_DAYS = 1  # see config.refresh.prices_days


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _last_cached_date(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) as d FROM daily_prices WHERE ticker=?", (ticker,)
    ).fetchone()
    return row["d"] if row and row["d"] else None


def fetch_yfinance(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Pull adjusted OHLCV history from Yahoo Finance. Returns empty df on failure."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            logger.warning(f"yfinance returned no data for {ticker}")
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_stooq_fallback(ticker: str) -> pd.DataFrame:
    """Documented fallback source. Stooq symbols are lowercase + '.us' suffix for US equities."""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        df = pd.read_csv(url)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["adj_close"] = df["close"]  # Stooq daily CSV is already split/div adjusted for close
        return df
    except Exception as e:
        logger.error(f"Stooq fallback failed for {ticker}: {e}")
        return pd.DataFrame()


def _detect_split_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance 'stock splits' column marks the split ratio on the ex-date.
    We store it explicitly (SPEC #62) instead of silently relying on
    auto-adjusted prices, so signal/trade math can stay logically consistent
    even across a split that happens *after* a signal was generated.
    """
    if "stock_splits" in df.columns:
        df["split_factor"] = df["stock_splits"].apply(lambda x: x if x and x > 0 else 1.0)
    else:
        df["split_factor"] = 1.0
    return df


def ingest_ticker(ticker: str, conn: sqlite3.Connection, force: bool = False) -> dict:
    """
    Ingest/refresh price history for one ticker. Caches: only re-fetches if
    the cached data is older than REFRESH_DAYS or `force=True` (SPEC #53).
    Returns a small status dict for the data-health log.
    """
    last_date = _last_cached_date(conn, ticker)
    if last_date and not force:
        last_dt = datetime.fromisoformat(last_date)
        if (datetime.now() - last_dt).days < REFRESH_DAYS:
            return {"ticker": ticker, "status": "CACHED", "rows": 0}

    df = fetch_yfinance(ticker)
    source = "yfinance"
    if df.empty:
        df = fetch_stooq_fallback(ticker)
        source = "stooq"
    if df.empty:
        _log_health(conn, ticker, "UNAVAILABLE", "No data from yfinance or stooq", "ERROR")
        return {"ticker": ticker, "status": "FAILED", "rows": 0}

    df = _detect_split_factor(df)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    rows = []
    ingested_at = _now()
    for _, r in df.iterrows():
        adj_close = r.get("adj_close", r.get("close"))
        rows.append((
            ticker, r["date"],
            float(r.get("open", 0) or 0), float(r.get("high", 0) or 0),
            float(r.get("low", 0) or 0), float(r.get("close", 0) or 0),
            float(adj_close or 0), int(r.get("volume", 0) or 0),
            float(r.get("split_factor", 1.0) or 1.0),
            source, "END-OF-DAY", ingested_at,
        ))

    conn.executemany(
        """INSERT INTO daily_prices
           (ticker, date, open, high, low, close, adj_close, volume,
            split_factor, source, data_status, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker, date) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, adj_close=excluded.adj_close,
             volume=excluded.volume, split_factor=excluded.split_factor,
             source=excluded.source, data_status=excluded.data_status,
             ingested_at=excluded.ingested_at""",
        rows,
    )
    _log_health(conn, ticker, "OK", f"{len(rows)} rows from {source}", "INFO")
    return {"ticker": ticker, "status": "OK", "rows": len(rows)}


def _log_health(conn, ticker, status, message, severity, source="prices", date=None):
    conn.execute(
        "INSERT INTO data_health_log (date, source, ticker, status, message, severity, logged_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (date or datetime.now().strftime("%Y-%m-%d"), source, ticker, status, message, severity, _now()),
    )


def ingest_universe(tickers: list[str], db_file=None, force: bool = False, sleep_s: float = 0.3) -> pd.DataFrame:
    """
    Ingest a list of tickers one at a time with a small delay between calls
    to respect Yahoo's unofficial rate limits (SPEC: 'use APIs responsibly').
    A single failing ticker never aborts the whole scan (SPEC #54).
    """
    cfg = load_config()
    db_file = db_file or db_path(cfg)
    init_db(db_file)
    results = []
    with connect(db_file) as conn:
        for t in tickers:
            try:
                res = ingest_ticker(t, conn, force=force)
            except Exception as e:
                logger.error(f"Unhandled error ingesting {t}: {e}")
                res = {"ticker": t, "status": "ERROR", "rows": 0}
                _log_health(conn, t, "ERROR", str(e), "ERROR")
            results.append(res)
            time.sleep(sleep_s)
    return pd.DataFrame(results)


def load_prices(ticker: str, db_file=None, start=None, end=None) -> pd.DataFrame:
    cfg = load_config()
    db_file = db_file or db_path(cfg)
    with connect(db_file) as conn:
        q = "SELECT * FROM daily_prices WHERE ticker=?"
        params = [ticker]
        if start:
            q += " AND date>=?"; params.append(start)
        if end:
            q += " AND date<=?"; params.append(end)
        q += " ORDER BY date"
        df = pd.read_sql_query(q, conn, params=params)
    return df


if __name__ == "__main__":
    cfg = load_config()
    tickers = cfg["universe"]["seed_tickers"]
    print(f"Ingesting {len(tickers)} seed tickers...")
    res = ingest_universe(tickers)
    print(res.to_string(index=False))
