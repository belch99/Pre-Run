# PRE-RUN Model Methodology (v1.0)

This documents the exact formula behind every currently-live score component.
Weights are in `config/config.yaml -> weights` and are intentionally
*not* claimed to be correct — SPEC #6 explicitly says "do not assume the
initial weights are perfect." They're a defensible starting point to be
revised once out-of-sample backtests exist.

## Live components (60 of 100 nominal points)

### Momentum Structure — 15 pts (`src/scoring/prerun_score.py: score_momentum`)
Inputs: 5/10/20-day returns, 20d trend slope, relative strength vs SPY and
sector ETF (20d).
Logic: each return window is scored via a "sweet spot" function — modest
positive return scores highest, near-zero/negative scores low, and returns
past a penalty threshold (10-25% depending on window) are actively
*penalized*, per SPEC #6's explicit instruction not to reward stocks that
already exploded. Relative strength vs SPY/sector adds a bonus on top.

### Volume/Accumulation — 20 pts (`score_volume`)
Inputs: relative volume vs 20d average, 5-day-avg-volume vs 20-day-avg-volume
ratio.
Logic: rewards *sustained*, gradually rising participation (5d/20d ratio
climbing above ~0.9-1.7x) over a single explosive one-day spike, matching
SPEC #7's "quiet accumulation > price+20%-on-10x-volume" framing. This is
the single highest-weighted component by design (SPEC #7 calls it "one of
the most important").

### Volatility Compression — 10 pts (`score_compression`)
Inputs: 5-day-range vs 20-day-range ratio, Bollinger Band width (20d, 2σ),
consecutive inside-day streak.
Logic: lower range ratio and narrower BB width both score higher (energy
buildup); a longer inside-day streak adds a bonus, capped at 5 days for full
credit.

### Breakout Proximity — 15 pts (`score_breakout_proximity`)
Inputs: distance from current price to 20-day high (used as the resistance/
trigger level in Phase 1-6 — a more sophisticated pattern-recognition
resistance detector, e.g. actual swing-high clustering, is a good Phase-10
upgrade).
Logic: distance <= 0 (price above resistance) is scored as "already broke
out" and decays fast; distance in (0%, 10%] scores linearly higher the
closer to the trigger, per SPEC #9's exact example (3% from resistance >
20% from resistance).

## N/A components (40 of 100 nominal points, not yet wired up)

| Component | Weight | Blocking dependency |
|---|---|---|
| Catalyst | 15 | Earnings calendar / FDA calendar / SEC 8-K feed with real publish timestamps |
| Options Activity | 10 | Free, reliable options volume/OI source (CBOE free feeds are delayed/limited; needs evaluation) |
| Short Interest | 5 | FINRA bi-monthly short interest file (free, but only bi-monthly — days_to_cover freshness will always lag) |
| Insider Activity | 5 | SEC EDGAR Form 4 full-text/XBRL parsing |
| News/Attention Acceleration | 5 | Free news API with real publish timestamps (Alpha Vantage/Finnhub free tiers, rate-limited) |

Per SPEC #3/#96, these are computed as `None` (not 0, not a guessed neutral
value) and excluded from both the numerator and denominator of the score.
`compute_prerun_score()` rescales the score to `/100 of points actually
available` and separately reports `max_possible_pts` so a technical-only 62
is never displayed identically to a full-model 62.

## Classification bands

| Score | Label |
|---|---|
| 0-49 | NO_SETUP |
| 50-64 | WATCH |
| 65-74 | EARLY |
| 75-84 | SETUP |
| 85-94 | IMMINENT |
| 95-100 | EXTREME_SETUP |

These are **not** probability bands (SPEC #18). Whether a 90 actually
outperforms a 70 out-of-sample is an empirical question this project exists
to answer, not an assumption baked into the labels.

## "Already running" flag
Currently: `ret_5d >= already_running.return_5d_threshold` (default 20%,
`config.yaml`). This is a simplified first pass — SPEC #46 also asks for
distance-above-moving-averages, ATR expansion, and gap-size inputs, which
are computed in `src/indicators/technical.py` but not yet folded into this
specific flag. Straightforward next iteration.

## Run definitions (forward outcome labels)
Defined once in `config.yaml -> run_definitions` and computed in
`backtesting/engine.py: run_definitions_hit()`:

- RUN_1: +5% within 5 trading days
- RUN_2: +10% within 5 trading days
- RUN_3: +15% within 10 trading days
- RUN_4: +20% within 20 trading days
- RUN_5: +30% within 20 trading days

Forward return, max favorable excursion (MFE), and max adverse excursion
(MAE) are computed for each of the 1/3/5/10/20-day windows in
`config.yaml -> forward_windows_days`, using ONLY price rows strictly after
the signal date (`forward_outcomes()` in `backtesting/engine.py`).

## No-look-ahead guarantee — implementation detail
`backtesting/engine.py: history_up_to_t(df, t_idx)` returns `df.iloc[:t_idx+1]`
— everything up to and including the signal date, nothing after. Every
indicator and score function receives only that slice. `forward_outcomes()`
is called separately, after scoring, and only reads `df.iloc[t_idx+1:]`.
`tests/test_pipeline_synthetic.py::test_no_lookahead_forward_returns_use_only_future_rows`
asserts this numerically against a manual calculation.

This does not yet enforce point-in-time correctness for anything *besides*
price (e.g. a fundamentals/news/insider join that uses `as_of_date` instead
of true publish timestamp would silently reintroduce look-ahead bias once
Phase 7-9 land) — SPEC #61 flags this explicitly as needing its own care
when those data sources are added; the `catalysts`/`news`/`insider_transactions`
tables already have `published_at` / `filing_date` columns for exactly this
reason, they just aren't populated or joined yet.
