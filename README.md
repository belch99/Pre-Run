# PRE-RUN

**Find the move before the move.**

An experimental, $0/month, research-driven system that tries to detect early
characteristics of upward stock moves, scores the setups, generates
hypothetical (paper) trades, and — most importantly — **measures whether the
scoring actually has predictive value.** It is not a stock picker and it does
not place real trades.

## What's built so far (Phases 1–6 of the project spec)

| Phase | Status |
|---|---|
| 1. Price/volume data | ✅ `src/data/prices.py` (yfinance primary, Stooq fallback) |
| 2. Technical indicators | ✅ `src/indicators/technical.py` |
| 3. Initial PRE-RUN score | ✅ `src/scoring/prerun_score.py` — momentum, volume, compression, breakout only (see below) |
| 4. Historical backtesting | ✅ `src/backtesting/engine.py` — point-in-time, no look-ahead |
| 5. Paper-trading engine | ✅ `src/paper_trading/engine.py` |
| 6. Dashboard | ✅ Streamlit, `app.py` + `pages/` (Command Center, Early Detection, Backtest, Paper Trades) |
| 7. SEC/insider data | ⏳ stub only — not connected |
| 8. Catalyst/news data | ⏳ stub only — not connected |
| 9. Options/short-interest data | ⏳ stub only — not connected |
| 10. Advanced modeling | ⏳ not started (correctly — SPEC #66 says prove the baseline first) |

## Known limitation: only 4 of 9 score components are live

The PRE-RUN score is nominally out of 100 across 9 weighted components. Only
four have a free, reliable, point-in-time data source wired up right now:
**Momentum Structure (15), Volume/Accumulation (20), Volatility Compression
(10), Breakout Proximity (15)** — 60 of 100 possible points.

**Catalyst (15), Options Activity (10), Short Interest (5), Insider Activity
(5), and News Attention (5)** are marked `N/A` and excluded from both the
numerator and denominator of the score (see `max_possible_pts` on every
score row) rather than defaulted to a fake neutral value. This is a direct
requirement of the project spec (never fabricate missing data) and it means
**the current score is really "Technical PRE-RUN," not the full model.**
Wiring up Phase 7–9 (SEC EDGAR for insiders/filings, a free news API, and
whatever free options/short-interest source proves reliable) is the next
priority before the score should be trusted for anything beyond a technical
signal.

## Known limitation: this was built inside a network-sandboxed environment

Anthropic's development sandbox used to write this code can only reach
package registries (PyPI, npm, GitHub) — it cannot reach
`query1/query2.finance.yahoo.com`, `stooq.com`, `sec.gov`, or any other market
data source. Every module was validated with:
1. **Unit/integration logic** against synthetic random-walk price data
   (`tests/test_pipeline_synthetic.py`) — this proves the code runs correctly
   end-to-end and that the no-look-ahead math is exact, but **proves nothing
   about real market behavior.**
2. Graceful, honest failure when real network calls are blocked (see the
   console output of `scripts/run_first_experiment.py` if you run it here) —
   it says "0/22 tickers ingested" and tells you why, rather than
   pretending it has data.

**You must run this on your own machine** (normal laptop with internet
access) for it to do anything real. Nothing about the architecture is
sandbox-specific — it's a stock $0-cost Python/SQLite/Streamlit stack.

## Quickstart (on your own machine)

```bash
git clone <wherever you put this> pre_run   # or just unzip
cd pre_run
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python run.py init                # create the SQLite database
python run.py experiment          # Phase 1-4: ingest seed universe + run first backtest
python run.py walkforward         # Phase 27: train/validation/out-of-sample splits
python run.py scan                # Phase 25: today's live/frozen signal generation
python run.py dashboard           # launch Streamlit at http://localhost:8501
```

`run.py experiment` answers the FIRST MILESTONE question directly in the
console: the 20 strongest PRE-RUN candidates as of the last cached trading
day, and what happened to them over the next 1/3/5/10/20 days.

## Scheduling the daily scan

Two free options, pick one:

**Option A — local cron/Task Scheduler** (runs on your own machine):
```
# crontab -e, run after US market close (4:30pm ET) on weekdays
30 16 * * 1-5 cd /path/to/pre_run && /path/to/venv/bin/python scripts/run_daily_scan.py >> logs/cron.log 2>&1
```

**Option B — GitHub Actions** (runs on GitHub's servers, nothing local, $0):
`.github/workflows/daily_scan.yml` is already set up to run the scan on a
schedule (weekdays after close) and commit the updated database back to the
repo. Just push this repo to GitHub — public repos get unlimited free
Actions minutes, private repos get 2,000 free minutes/month (a daily scan of
a small universe takes well under a minute). Combine with
[Streamlit Community Cloud](https://streamlit.io/cloud) (also free) pointed
at `app.py` in the same repo for a fully hosted dashboard with zero local
footprint. Note this also sidesteps the network restriction I hit building
this — GitHub's runners have normal internet access to Yahoo/Stooq/SEC.

## Expanding the universe

`config/config.yaml` ships with a small 22-ticker seed list so the first
backtest runs in seconds, not hours. To scan a broader universe, replace
`universe.seed_tickers` with a real listing file (e.g. NASDAQ/NYSE symbol
files, or an S&P 1500 constituent list) — `scripts/` is the natural home for
a `build_universe.py` that applies the liquidity/price filters in
`config.yaml -> universe.normal / universe.aggressive` (SPEC #4).

## Interpreting scores — read this before trusting anything

- A PRE-RUN score is **not a probability.** A 90 means "many characteristics
  the model currently considers favorable," not "90% chance of a run."
- Check `max_possible_pts` on every score. A 62 computed from 60 available
  points (technical-only) is not the same claim as a 62 computed from all
  100 (once Phase 7-9 lands).
- Always check the **OUT_OF_SAMPLE** `dataset_split` in the Backtest page
  before trusting any pattern. In-sample/training results are expected to
  look better than reality — that's overfitting, not edge.
- The honest core question this whole project exists to answer:
  **"Does a higher PRE-RUN score actually correspond to a higher probability
  of a future run — out of sample?"** If the score-bucket table doesn't show
  a roughly monotonic relationship between score and hit rate on
  out-of-sample data, the model does not currently work and should not be
  trusted, regardless of how good it looks in-sample.

## Project structure

See `MODEL_METHODOLOGY.md` for exact formulas behind every score component,
and the original project spec (provided separately) for the full 97-item
requirements list this was built against.

```
pre_run/
├── app.py                    # Streamlit entry point
├── run.py                    # convenience CLI
├── config/config.yaml        # ALL weights/thresholds live here, never hardcoded
├── src/
│   ├── data/                 # prices.py, db.py, regime.py (+ stubs: sec.py, news.py, options.py)
│   ├── indicators/           # technical.py
│   ├── scoring/              # prerun_score.py
│   ├── backtesting/          # engine.py -- the point-in-time no-look-ahead core
│   ├── paper_trading/        # engine.py
│   └── utils/                # config.py, logging
├── pages/                    # Streamlit multi-page dashboard
├── scripts/                  # run_first_experiment.py, run_daily_scan.py, walk_forward_test.py
└── tests/                    # test_pipeline_synthetic.py (+ add real-data tests once you have data)
```

## What is explicitly NOT done yet

- Phase 7-9 data sources (SEC/insider, catalyst/news, options/short interest)
  are stubbed as `N/A`, not implemented.
- No ML models (SPEC #66/#67 explicitly says prove the simple signals work
  first — none of the score-bucket tables have been run on real data yet,
  so ML would be premature).
- No alerting (Telegram/Discord/email) wired up — `.env.example` has the
  placeholders but `src/alerts/` is an empty stub.
- No real symbol-universe builder (uses a 22-ticker seed list).
- Correlation/sector-concentration controls (SPEC #74), risk flags (SPEC #75),
  and the $2K-account-specific dashboard view (SPEC #31) are designed in the
  schema/config but not yet built as dashboard pages.

None of this was skipped to cut corners — it's the natural result of
building in dependency order (data → indicators → score → backtest → paper
trading → dashboard) and stopping at a genuinely working Phase 1-6 rather
than half-wiring all ten phases. Pick up at Phase 7 next.
