from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sqlalchemy.orm import Session

import yfinance as yf
from app.services.securities import resolve_symbol_from_cusip, should_overwrite_symbol

from app.infra.models import LMTransaction, DividendEvent

# --- Simple in-process TTL cache for yfinance profiles ---
_DIVIDEND_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}


def _yf_cache_ttl_seconds() -> int:
    try:
        return int(os.getenv("MIRA_YF_CACHE_TTL_SECONDS", "3600"))  # default 1h
    except Exception:
        return 3600


# ---------- Internal helpers ----------


def _parse_date(d: Any) -> date:
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return date.fromisoformat(d)
    raise ValueError(f"Unsupported date type: {type(d)}")


def _load_plaid_metadata(tx: LMTransaction) -> Dict[str, Any]:
    """
    LMTransaction.raw is JSON from Lunch Money.

    We expect raw to be either:
      - a dict (already parsed), or
      - a JSON string.
    Inside it there is often a 'plaid_metadata' field which itself
    is a JSON string from Plaid.
    """
    if not tx.raw:
        return {}

    if isinstance(tx.raw, dict):
        raw = tx.raw
    else:
        try:
            raw = json.loads(tx.raw)
        except Exception:
            return {}

    meta_str = raw.get("plaid_metadata")
    if not meta_str:
        return {}

    try:
        meta = json.loads(meta_str)
    except Exception:
        return {}

    return meta if isinstance(meta, dict) else {}


def _extract_cusip_from_metadata(meta: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to extract a CUSIP from Plaid/Lunch Money metadata.
    Parses the CUSIP from the 'name' field when it follows the pattern:
    "Dividend of 78433H303 $7.78 received. - DIVIDEND"
    """
    if not meta:
        return None

    name = meta.get("name") or ""
    if not name:
        return None

    marker = "Dividend of "
    if marker not in name:
        return None

    try:
        after_marker = name.split(marker, 1)[1]
        token = after_marker.split()[0]
        token = token.strip(",.$")
        if token and token.isalnum():
            return token.upper()
    except Exception:
        return None

    return None


def _is_dividend_tx(tx: LMTransaction) -> bool:
    """
    Heuristic to detect dividend transactions from LM + Plaid.
    - Prefer Plaid subtype == 'dividend'
    - Fallback: 'DIVIDEND' in payee/original_name.
    """
    meta = _load_plaid_metadata(tx)
    subtype = meta.get("subtype")

    if subtype == "dividend":
        return True

    payee = (tx.payee or "").upper()
    orig = ""
    try:
        if isinstance(tx.raw, dict):
            orig = (tx.raw.get("original_name") or "").upper()
        else:
            raw = json.loads(tx.raw) if tx.raw else {}
            orig = (raw.get("original_name") or "").upper()
    except Exception:
        pass

    if "DIVIDEND" in payee or "DIVIDEND" in orig:
        return True

    return False


def _tx_date(tx: LMTransaction) -> date:
    return _parse_date(tx.date)


@dataclass
class DividendCashEvent:
    tx_id: int
    tx_date: date         # pay date (from Lunch Money)
    amount: float         # cash amount (positive = cash to you)
    symbol: Optional[str] # we may not always know this reliably
    cusip: Optional[str]  # security identifier (when available) from Plaid/Lunch Money metadata
    raw_payee: str


# ---------- Public: realized dividends from LM ----------


def extract_dividend_events(
    transactions: Iterable[LMTransaction],
) -> List[DividendCashEvent]:
    """
    Extract dividend cashflows from a list of LMTransaction rows.
    This is purely "what hit the account" history.
    """
    events: List[DividendCashEvent] = []

    for tx in transactions:
        if not _is_dividend_tx(tx):
            continue

        # Lunch Money often stores inflows as negative amounts
        # (from the perspective of "spending"), so flip sign so
        # 'amount' here means "cash you received".
        amount = float(tx.amount or 0.0)
        if amount < 0:
            amount = -amount

        meta = _load_plaid_metadata(tx)
        cusip = _extract_cusip_from_metadata(meta)

        # We *may* be able to infer a symbol from metadata 'name'
        # like "JEPI dividend ..." but your sample used CUSIPs.
        name = (meta.get("name") or tx.payee or "").upper()

        symbol: Optional[str] = None

        # Very weak heuristic: if name looks like "XYZ dividend",
        # grab first token that is all letters.
        tokens = [t.strip(",.") for t in name.split()]
        for t in tokens:
            if t.isalpha() and len(t) <= 5:
                symbol = t
                break

        events.append(
            DividendCashEvent(
                tx_id=tx.id,
                tx_date=_tx_date(tx),
                amount=round(amount, 2),
                symbol=symbol,
                cusip=cusip,
                raw_payee=tx.payee or "",
            )
        )

    return events


def summarize_dividends(
    events: List[DividendCashEvent],
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Basic realized dividend summary:
      - total
      - by_date
      - by_month (YYYY-MM)
      - by_symbol (when symbol is known)
    """
    if start is None and end is None:
        # default: last 365 days from today
        end = date.today()
        start = end - timedelta(days=365)
    elif start is None and end is not None:
        start = end - timedelta(days=365)
    elif start is not None and end is None:
        end = date.today()

    assert start is not None and end is not None

    total = 0.0
    by_date: Dict[str, float] = defaultdict(float)
    by_month: Dict[str, float] = defaultdict(float)
    by_symbol: Dict[str, float] = defaultdict(float)

    for ev in events:
        if ev.tx_date < start or ev.tx_date > end:
            continue

        total += ev.amount
        ds = ev.tx_date.isoformat()
        ms = f"{ev.tx_date.year:04d}-{ev.tx_date.month:02d}"

        by_date[ds] += ev.amount
        by_month[ms] += ev.amount

        if ev.symbol:
            by_symbol[ev.symbol] += ev.amount

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_dividends": round(total, 2),
        "by_date": {k: round(v, 2) for k, v in sorted(by_date.items())},
        "by_month": {k: round(v, 2) for k, v in sorted(by_month.items())},
        "by_symbol": {k: round(v, 2) for k, v in sorted(by_symbol.items())},
        "events_count": len(events),
    }


# ---------- Public: forward / ex-date info via yfinance ----------


def get_symbol_dividend_profile(symbol: str, bypass_cache: bool = False) -> Dict[str, Any]:
    """
    Use yfinance to fetch historical dividends and a simple
    forward-looking estimate for one symbol.
    """
    now = datetime.utcnow()
    if not bypass_cache:
        entry = _DIVIDEND_PROFILE_CACHE.get(symbol)
        if entry and entry.get("expires_at") and entry["expires_at"] > now:
            return entry["profile"]

    ticker = yf.Ticker(symbol)

    # Historical ex-dividend dates (index = ex-date, value = cash/share)
    try:
        divs = ticker.dividends
    except Exception:
        divs = None

    history: List[Dict[str, Any]] = []
    trailing_12m = 0.0
    last_ex_date: Optional[str] = None

    if divs is not None and not divs.empty:
        cutoff = datetime.today().date() - timedelta(days=365)
        for dt, amt in divs.items():
            d = dt.date() if isinstance(dt, (date, datetime)) else dt
            history.append(
                {
                    "ex_date": d.isoformat(),
                    "amount_per_share": float(amt),
                }
            )
            if d >= cutoff:
                trailing_12m += float(amt)

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # Most recent ex-dividend date (Series index is ex-date)
    try:
        if divs is not None and len(divs):
            last_ex_date = divs.index.max().date().isoformat()
    except Exception:
        last_ex_date = None

    indicated_annual = float(info.get("dividendRate") or 0.0)
    indicated_yield = float(info.get("dividendYield") or 0.0)

    # Annualized forward per-share using recent ex-date history
    annualized_forward_per_share = 0.0
    forward_method = "indicated"
    try:
        if divs is not None and len(divs):
            today = date.today()
            cutoff = today - timedelta(days=365)
            total_12m = 0.0
            payouts_12m = 0
            earliest_dt: Optional[date] = None
            for idx, cash in divs.items():
                try:
                    d = idx.date()
                except Exception:
                    continue
                if earliest_dt is None or d < earliest_dt:
                    earliest_dt = d
                if d >= cutoff:
                    payouts_12m += 1
                    try:
                        total_12m += float(cash or 0.0)
                    except Exception:
                        pass

            has_12m_history = bool(earliest_dt and earliest_dt <= cutoff)
            try:
                last3 = float(divs.tail(3).sum())
            except Exception:
                last3 = 0.0
            try:
                last4 = float(divs.tail(4).sum())
            except Exception:
                last4 = 0.0

            if has_12m_history and total_12m > 0:
                annualized_forward_per_share = total_12m
                forward_method = "t12m"
            elif payouts_12m >= 6 and last3 > 0:
                annualized_forward_per_share = last3 * 4.0
                forward_method = "3mo_annualized"
            elif 3 <= payouts_12m <= 5 and last4 > 0:
                annualized_forward_per_share = last4
                forward_method = "4payouts"
            else:
                annualized_forward_per_share = indicated_annual
                forward_method = "indicated"
        else:
            annualized_forward_per_share = indicated_annual
            forward_method = "indicated"
    except Exception:
        annualized_forward_per_share = indicated_annual
        forward_method = "indicated"

    result = {
        "symbol": symbol,
        "trailing_12m_div_per_share": round(trailing_12m, 4),
        "indicated_annual_div_per_share": round(indicated_annual, 4),
        "indicated_dividend_yield": indicated_yield,
        "ex_div_history": history,
        "last_ex_date": last_ex_date,
        "annualized_forward_per_share": round(annualized_forward_per_share, 4),
        "forward_method": forward_method,
    }

    try:
        ttl = _yf_cache_ttl_seconds()
        _DIVIDEND_PROFILE_CACHE[symbol] = {
            "profile": result,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl),
        }
    except Exception:
        pass

    return result


def estimate_forward_dividends_for_holdings(
    holdings: List[Dict[str, Any]], bypass_cache: bool = False
) -> Dict[str, Any]:
    """
    Given holdings from the /lm/holdings snapshot (each with symbol & shares),
    attach a simple forward 12m dividend estimate per symbol and total.
    """
    per_symbol: Dict[str, Dict[str, Any]] = {}
    total_forward_12m = 0.0

    for h in holdings:
        symbol = h["symbol"]
        shares = float(h["shares"])
        profile = get_symbol_dividend_profile(symbol, bypass_cache=bypass_cache)

        # Prefer computed annualized rate from ex-date history; then indicated; then trailing 12m
        annual_per_share = float(
            profile.get("annualized_forward_per_share")
            or profile.get("indicated_annual_div_per_share")
            or profile.get("trailing_12m_div_per_share")
            or 0.0
        )

        forward_12m = round(annual_per_share * shares, 2)
        total_forward_12m += forward_12m
        projected_monthly_dividend = round(forward_12m / 12.0, 2)

        per_symbol[symbol] = {
            "symbol": symbol,
            "shares": shares,
            "annual_div_per_share": round(annual_per_share, 4),
            "forward_12m_dividend": forward_12m,
            "indicated_yield": profile["indicated_dividend_yield"],
            "last_ex_date": profile.get("last_ex_date"),
            "forward_method": profile.get("forward_method"),
            "projected_monthly_dividend": projected_monthly_dividend,
        }

    return {
        "total_forward_12m": round(total_forward_12m, 2),
        "by_symbol": per_symbol,
    }


# ---------- Public: persist dividend events ----------

def sync_dividend_events_from_transactions(
    db: Session,
    plaid_account_id: int,
    transactions: Iterable[LMTransaction],
) -> List[DividendEvent]:
    """Persist realized dividend cash events into dividend_events.

    Idempotent on lm_transaction_id: if an event already exists for a given
    LM transaction, it will be updated; otherwise it will be inserted.
    """
    # Index transactions by id so we can grab raw/currency, etc.
    tx_by_id: Dict[int, LMTransaction] = {int(tx.id): tx for tx in transactions}

    events = extract_dividend_events(tx_by_id.values())
    persisted: List[DividendEvent] = []

    # Resolve symbols from CUSIP when parser extracted a stop-word (e.g., "OF")
    for ev in events:
        if should_overwrite_symbol(ev.symbol) and ev.cusip:
            resolved = resolve_symbol_from_cusip(db, ev.cusip)
            if resolved:
                ev.symbol = resolved

    for ev in events:
        tx = tx_by_id.get(ev.tx_id)

        # Try to infer currency from LMTransaction; default to USD.
        currency = "USD"
        if tx is not None and getattr(tx, "currency", None):
            currency = str(tx.currency).upper()

        # Normalize raw payload to a dict if possible.
        raw_payload: Optional[Dict[str, Any]] = None
        if tx is not None and tx.raw is not None:
            if isinstance(tx.raw, dict):
                raw_payload = tx.raw
            else:
                try:
                    raw_payload = json.loads(tx.raw)
                except Exception:
                    raw_payload = None

        existing: Optional[DividendEvent] = (
            db.query(DividendEvent)
            .filter(DividendEvent.lm_transaction_id == ev.tx_id)
            .one_or_none()
        )

        if existing is None:
            de = DividendEvent(
                plaid_account_id=plaid_account_id,
                lm_transaction_id=ev.tx_id,
                symbol=ev.symbol,
                cusip=ev.cusip,
                pay_date=ev.tx_date,
                amount=ev.amount,
                currency=currency,
                source="plaid",
                raw=raw_payload,
            )
            # de.symbol already set above if resolved; ensure consistency
            db.add(de)
            persisted.append(de)
        else:
            existing.plaid_account_id = plaid_account_id
            existing.symbol = ev.symbol
            existing.cusip = ev.cusip
            existing.pay_date = ev.tx_date
            existing.amount = ev.amount
            existing.currency = currency
            existing.source = "plaid"
            existing.raw = raw_payload
            persisted.append(existing)

    return persisted


def backfill_dividend_cusips(
    db: Session,
    start: date,
    end: date,
) -> int:
    """Backfill CUSIP values on existing dividend_events between start and end dates."""
    updated = 0

    q = (
        db.query(DividendEvent)
        .filter(DividendEvent.pay_date >= start)
        .filter(DividendEvent.pay_date <= end)
    )

    for de in q:
        if de.cusip is not None:
            continue
        if de.lm_transaction_id is None:
            continue

        tx = db.query(LMTransaction).get(de.lm_transaction_id)
        if tx is None:
            continue

        meta = _load_plaid_metadata(tx)
        cusip = _extract_cusip_from_metadata(meta)
        if not cusip:
            continue

        de.cusip = cusip
        updated += 1

    if updated:
        db.commit()

    return updated
