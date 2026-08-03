"""
Paper trading engine (SPEC #23-25, #30-33, #73).

Rules enforced here:
- A signal creates a WAITING trade; it only becomes ENTERED when price
  actually crosses the trigger (SPEC #33) -- score alone never triggers a fill.
- Once created, entry/trigger/stop/target/score fields are frozen (SPEC #25).
  Only status, exit fields, and running MFE/MAE get updated as new price data
  arrives -- the original signal is never edited retroactively.
- Position sizing is risk-based: risk_per_trade * account_size / stop_distance,
  capped at whole shares (SPEC #30), and never allowed to exceed available cash
  (SPEC #73).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import pandas as pd

from src.utils.config import load_config, get_logger

logger = get_logger("paper_trading")


def _now():
    return datetime.now(timezone.utc).isoformat()


def position_size(account_size: float, risk_pct: float, entry: float, stop: float,
                   whole_shares_only: bool = True) -> dict:
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return {"shares": 0, "capital_allocated": 0, "dollar_risk": 0, "error": "invalid stop distance"}

    dollar_risk_budget = account_size * risk_pct
    raw_shares = dollar_risk_budget / stop_distance
    shares = int(raw_shares) if whole_shares_only else raw_shares

    capital_needed = shares * entry
    if capital_needed > account_size:
        shares = int(account_size / entry) if whole_shares_only else account_size / entry
        capital_needed = shares * entry

    dollar_risk = shares * stop_distance
    return {
        "shares": shares,
        "capital_allocated": round(capital_needed, 2),
        "dollar_risk": round(dollar_risk, 2),
        "pct_risk": round(dollar_risk / account_size, 4) if account_size else None,
    }


def create_paper_trade(signal: dict, account_size: float = None, risk_pct: float = None,
                        cfg: dict = None) -> dict:
    """
    signal: dict with keys ticker, signal_date, signal_time, price (entry ref),
    trigger_price, stop_price, target1, target2, score, score_components (dict),
    market_regime, catalyst, model_version.
    Returns a paper_trades row dict, status = WAITING, ready for DB insert.
    """
    cfg = cfg or load_config()
    account_size = account_size or cfg["accounts"]["default"]
    risk_pct = risk_pct or cfg["accounts"]["default_risk_per_trade"]

    entry_ref = signal.get("trigger_price") or signal["price"]
    stop = signal["stop_price"]
    sizing = position_size(account_size, risk_pct, entry_ref, stop,
                            cfg["accounts"].get("whole_shares_only", True))

    return {
        "signal_id": signal.get("signal_id"),
        "ticker": signal["ticker"],
        "signal_date": signal["signal_date"],
        "signal_time": signal.get("signal_time"),
        "entry_price": entry_ref,
        "trigger_price": signal.get("trigger_price"),
        "stop_price": stop,
        "target1": signal.get("target1"),
        "target2": signal.get("target2"),
        "shares": sizing["shares"],
        "capital_allocated": sizing["capital_allocated"],
        "dollar_risk": sizing["dollar_risk"],
        "pct_risk": sizing.get("pct_risk"),
        "account_size": account_size,
        "status": "WAITING",
        "score": signal.get("score"),
        "score_components": json.dumps(signal.get("score_components", {})),
        "market_regime": signal.get("market_regime"),
        "catalyst": signal.get("catalyst"),
        "notes": "",
        "model_version": signal.get("model_version"),
        "created_at": _now(),
        "updated_at": _now(),
    }


def update_trade_with_new_price(trade: dict, date: str, high: float, low: float, close: float,
                                 volume: float, avg_volume: float, cfg: dict = None) -> dict:
    """
    Advance a single trade's status given a new day's OHLCV. Pure function --
    returns an updated copy, caller writes it back to the DB. This is what
    runs once per day in the after-close workflow (SPEC #81).
    """
    cfg = cfg or load_config()
    t = dict(trade)
    status = t["status"]

    if status == "WAITING":
        trigger = t["trigger_price"]
        vol_confirm_mult = cfg["breakout"]["volume_confirmation_multiple"]
        volume_confirmed = (volume is not None and avg_volume and volume >= avg_volume * vol_confirm_mult)
        if trigger and high >= trigger:
            t["status"] = "ENTERED"
            t["entered_at"] = date
            # realistic fill: assume filled at trigger, not at the day's high
            t["entry_fill_price"] = trigger
            t["notes"] = (t.get("notes") or "") + f"; entered {date} at trigger {trigger}" + (
                " with volume confirmation" if volume_confirmed else " WITHOUT volume confirmation"
            )
        return t

    if status == "ENTERED":
        fill = t.get("entry_fill_price") or t["entry_price"]
        # update running MFE/MAE
        cur_mfe = t.get("mfe") or 0
        cur_mae = t.get("mae") or 0
        t["mfe"] = max(cur_mfe, (high - fill) / fill)
        t["mae"] = min(cur_mae, (low - fill) / fill)

        if t["stop_price"] and low <= t["stop_price"]:
            t["status"] = "STOPPED"
            t["exit_price"] = t["stop_price"]
            t["exit_date"] = date
            t["exit_reason"] = "stop hit"
        elif t["target1"] and high >= t["target1"]:
            t["status"] = "TARGET1"
            t["exit_price"] = t["target1"]
            t["exit_date"] = date
            t["exit_reason"] = "target 1 hit"
        else:
            return t  # still open

        t["return_pct"] = (t["exit_price"] - fill) / fill
        t["return_dollar"] = t["return_pct"] * t["shares"] * fill
        t["updated_at"] = _now()
        return t

    return t  # terminal states unchanged
