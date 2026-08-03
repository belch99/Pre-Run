"""
SQLite database layer for PRE-RUN.

Design notes (see PROJECT SPEC #52, #59):
- SQLite chosen over DuckDB for the transactional tables (signals, paper_trades,
  model_versions) because they're write-heavy and row-oriented.
- Every table that stores a computed value also stores the model_version and,
  where relevant, the point-in-time timestamp it was computed from, so a
  backtest is reproducible per SPEC #59.
- Nothing is ever UPDATEd in signals/paper_trades in a way that erases history
  (SPEC #24, #25) -- outcomes are appended to signal_outcomes / trade history
  rather than overwriting the original row.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
    industry TEXT,
    exchange TEXT,
    is_etf INTEGER DEFAULT 0,
    is_leveraged INTEGER DEFAULT 0,
    market_cap_category TEXT,          -- Micro/Small/Mid/Large/Mega
    first_seen TEXT,
    last_updated TEXT,
    active INTEGER DEFAULT 1           -- 0 if delisted; NEVER delete rows (survivorship bias, SPEC #63)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,                -- YYYY-MM-DD
    open REAL, high REAL, low REAL, close REAL,
    adj_close REAL, volume INTEGER,
    split_factor REAL DEFAULT 1.0,
    source TEXT,                       -- provider name
    data_status TEXT,                  -- REAL-TIME/DELAYED/END-OF-DAY/HISTORICAL
    ingested_at TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices(date);

CREATE TABLE IF NOT EXISTS technical_indicators (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    -- momentum
    ret_5d REAL, ret_10d REAL, ret_20d REAL, ret_50d REAL,
    dist_from_20d_high REAL, dist_from_52w_high REAL,
    trend_slope_20d REAL,
    rs_vs_spy_20d REAL, rs_vs_sector_20d REAL,
    -- volume
    rvol_20d REAL, rvol_50d REAL, vol5_vs_vol20 REAL,
    obv REAL, ad_line REAL,
    -- volatility / compression
    atr_14 REAL, atr_pct REAL, bb_width REAL, hv_20d REAL,
    range5_vs_range20 REAL, inside_day_streak INTEGER,
    -- breakout
    resistance_20d REAL, resistance_50d REAL, resistance_52w REAL,
    support_20d REAL, distance_to_breakout_pct REAL, trigger_price REAL,
    model_version TEXT,
    computed_at TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_ti_date ON technical_indicators(date);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    revenue_growth_yoy REAL, eps_growth_yoy REAL,
    gross_margin REAL, operating_margin REAL, free_cash_flow REAL,
    market_cap REAL, shares_outstanding REAL, float_shares REAL,
    source TEXT, data_status TEXT, ingested_at TEXT,
    PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS catalysts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    event_type TEXT,             -- earnings/fda/guidance/contract/etc
    event_date TEXT,
    classification TEXT,         -- POSITIVE/VOLATILITY/NEGATIVE/UNKNOWN
    description TEXT,
    source TEXT,
    published_at TEXT,           -- actual public availability timestamp (SPEC #61)
    ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_catalysts_ticker_date ON catalysts(ticker, event_date);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    headline TEXT,
    source TEXT,
    published_at TEXT NOT NULL,  -- must be true publish timestamp -- see SPEC #61
    url TEXT,
    ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_pub ON news(ticker, published_at);

CREATE TABLE IF NOT EXISTS insider_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    filer_name TEXT, role TEXT,
    transaction_type TEXT,      -- open_market_purchase/sale/option_exercise/tax_withholding
    shares REAL, price REAL, value REAL,
    filing_date TEXT, transaction_date TEXT,
    form_type TEXT DEFAULT 'Form4',
    source TEXT DEFAULT 'SEC EDGAR',
    ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker_date ON insider_transactions(ticker, filing_date);

CREATE TABLE IF NOT EXISTS options_activity (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    call_volume REAL, put_volume REAL, call_put_ratio REAL,
    call_oi REAL, put_oi REAL, iv_change REAL,
    unusual_flag INTEGER DEFAULT 0,
    source TEXT, data_status TEXT DEFAULT 'UNAVAILABLE',
    ingested_at TEXT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS short_interest (
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    short_interest_shares REAL, short_pct_float REAL, days_to_cover REAL,
    change_pct REAL, source TEXT DEFAULT 'FINRA/exchange bi-monthly report',
    data_status TEXT DEFAULT 'UNAVAILABLE',
    ingested_at TEXT,
    PRIMARY KEY (ticker, report_date)
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    prerun_score REAL,
    momentum_pts REAL, volume_pts REAL, compression_pts REAL,
    breakout_pts REAL, catalyst_pts REAL, options_pts REAL,
    short_interest_pts REAL, insider_pts REAL, news_pts REAL,
    max_possible_pts REAL,          -- denominator actually available (N/A components excluded)
    classification TEXT,
    already_running INTEGER DEFAULT 0,
    universe_mode TEXT,             -- normal/aggressive
    market_regime TEXT,
    model_version TEXT,
    explanation TEXT,               -- human-readable "why it ranked high"
    computed_at TEXT,
    UNIQUE(ticker, date, model_version)
);
CREATE INDEX IF NOT EXISTS idx_scores_date_score ON scores(date, prerun_score);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    signal_time TEXT,
    price_at_signal REAL,
    prerun_score REAL,
    score_id INTEGER,
    trigger_price REAL, stop_price REAL,
    target1 REAL, target2 REAL, target3 REAL,
    setup_type TEXT,               -- breakout/compression/accumulation/reversal/etc
    market_regime TEXT, sector TEXT,
    catalyst_summary TEXT,
    model_version TEXT,
    frozen INTEGER DEFAULT 1,      -- once written, score/prices here are never edited
    created_at TEXT,
    FOREIGN KEY(score_id) REFERENCES scores(id)
);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);

CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    fwd_days INTEGER NOT NULL,      -- 1/3/5/10/20
    fwd_return REAL,
    mfe REAL,                       -- max favorable excursion over window
    mae REAL,                       -- max adverse excursion over window
    hit_5pct INTEGER, hit_10pct INTEGER, hit_15pct INTEGER, hit_20pct INTEGER, hit_30pct INTEGER,
    days_to_5pct INTEGER, days_to_10pct INTEGER, days_to_15pct INTEGER,
    days_to_20pct INTEGER, days_to_30pct INTEGER,
    breakout_occurred INTEGER,
    computed_at TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id),
    UNIQUE(signal_id, fwd_days)
);

CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    ticker TEXT NOT NULL,
    signal_date TEXT, signal_time TEXT,
    entry_price REAL, trigger_price REAL, stop_price REAL,
    target1 REAL, target2 REAL,
    shares REAL, capital_allocated REAL,
    dollar_risk REAL, pct_risk REAL,
    account_size REAL,
    status TEXT DEFAULT 'WAITING',  -- WAITING/ENTERED/STOPPED/TARGET1/TARGET2/EXPIRED/CANCELLED
    entered_at TEXT, entry_fill_price REAL,
    exit_price REAL, exit_date TEXT, exit_reason TEXT,
    return_pct REAL, return_dollar REAL,
    mfe REAL, mae REAL,
    score REAL, score_components TEXT,   -- JSON
    market_regime TEXT, catalyst TEXT, notes TEXT,
    model_version TEXT,
    created_at TEXT, updated_at TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON paper_trades(status);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    created_at TEXT,
    weights_json TEXT,
    thresholds_json TEXT,
    notes TEXT,
    is_official INTEGER DEFAULT 0     -- 1 = official production model, 0 = model-lab experiment
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT,
    config_snapshot TEXT,        -- JSON dump of full config used
    start_date TEXT, end_date TEXT,
    universe_mode TEXT,
    dataset_split TEXT,          -- TRAINING/VALIDATION/OUT_OF_SAMPLE
    n_signals INTEGER,
    results_json TEXT,           -- summary stats
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    snapshot_date TEXT PRIMARY KEY,
    n_tickers_scanned INTEGER,
    n_qualifying_signals INTEGER,
    market_regime TEXT,
    top_candidates_json TEXT,
    data_health_json TEXT,
    model_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS market_regime (
    date TEXT PRIMARY KEY,
    spy_ret_20d REAL, qqq_ret_20d REAL, iwm_ret_20d REAL,
    vix_level REAL, vix_status TEXT,
    regime TEXT,                 -- bullish/neutral/bearish/high_vol/low_vol
    computed_at TEXT
);

CREATE TABLE IF NOT EXISTS data_health_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, source TEXT, ticker TEXT,
    status TEXT, message TEXT, severity TEXT,
    logged_at TEXT
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: str | Path):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/database/pre_run.db"
    init_db(path)
    print(f"Initialized database at {path}")
