from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sqlalchemy.orm import Session

import yfinance as yf

from app.infra.models import LMTransaction, DividendEvent


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


def get_symbol_dividend_profile(symbol: str) -> Dict[str, Any]:
    """
    Use yfinance to fetch historical dividends and a simple
    forward-looking estimate for one symbol.
    """
    ticker = yf.Ticker(symbol)

    # Historical ex-dividend dates (index = ex-date, value = cash/share)
    try:
        divs = ticker.dividends
    except Exception:
        divs = None

    history: List[Dict[str, Any]] = []
    trailing_12m = 0.0

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

    indicated_annual = float(info.get("dividendRate") or 0.0)
    indicated_yield = float(info.get("dividendYield") or 0.0)

    return {
        "symbol": symbol,
        "trailing_12m_div_per_share": round(trailing_12m, 4),
        "indicated_annual_div_per_share": round(indicated_annual, 4),
        "indicated_dividend_yield": indicated_yield,
        "ex_div_history": history,
    }


def estimate_forward_dividends_for_holdings(
    holdings: List[Dict[str, Any]],
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
        profile = get_symbol_dividend_profile(symbol)

        # Prefer indicated annual rate if present; fall back to trailing 12m.
        annual_per_share = profile["indicated_annual_div_per_share"] or profile[
            "trailing_12m_div_per_share"
        ]

        forward_12m = round(annual_per_share * shares, 2)
        total_forward_12m += forward_12m

        per_symbol[symbol] = {
            "symbol": symbol,
            "shares": shares,
            "annual_div_per_share": round(annual_per_share, 4),
            "forward_12m_dividend": forward_12m,
            "indicated_yield": profile["indicated_dividend_yield"],
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