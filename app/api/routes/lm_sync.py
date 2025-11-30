from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import json
import re
from decimal import Decimal
from collections import defaultdict

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.settings import settings
from app.infra.models import LMTransaction
from app.services.lm_client import LunchMoneyClient
from app.services.holdings import build_holdings_from_transactions
from app.services.dividends import sync_dividend_events_from_transactions

router = APIRouter(prefix="/lm/sync", tags=["lunchmoney"])



class LmTransactionsSyncRequest(BaseModel):
    plaid_account_id: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None


# New model for syncing all transactions (across all accounts)
class LmTransactionsSyncAllRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


def get_client() -> LunchMoneyClient:
    """Construct a LunchMoney client using the configured API key."""
    return LunchMoneyClient()


def _bucket_from_plaid_type(type_: str, subtype: str | None) -> str:
    """
    Very simple bucketing for v0:
      - depository -> cash_like
      - credit     -> credit
      - investment -> investment
      - loan       -> loan
      - everything else -> other
    """
    t = (type_ or "").lower()
    if t == "depository":
        return "cash_like"
    if t == "credit":
        return "credit"
    if t == "investment":
        return "investment"
    if t == "loan":
        return "loan"
    return "other"


@router.post("/plaid_accounts")
def sync_plaid_accounts(
    db: Session = Depends(get_db),  # reserved for when we start persisting
) -> Dict[str, Any]:
    """
    v0 sync:
    - Pull /plaid_accounts from Lunch Money
    - Compute simple totals by bucket
    - Return normalized accounts + totals (no DB writes yet)
    """
    client = get_client()
    try:
        raw_accounts: List[Dict[str, Any]] = client.get_plaid_accounts()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error from Lunch Money: {e}")

    normalized_accounts: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "cash_like": 0.0,
        "credit": 0.0,
        "investment": 0.0,
        "loan": 0.0,
        "other": 0.0,
    }

    for acc in raw_accounts:
        type_ = acc.get("type") or ""
        subtype = acc.get("subtype")
        balance_str = acc.get("balance") or "0"
        try:
            balance = float(balance_str)
        except (TypeError, ValueError):
            balance = 0.0

        bucket = _bucket_from_plaid_type(type_, subtype)
        totals[bucket] = totals.get(bucket, 0.0) + balance

        normalized_accounts.append(
            {
                "id": acc.get("id"),
                "name": acc.get("display_name") or acc.get("name"),
                "type": type_,
                "subtype": subtype,
                "institution_name": acc.get("institution_name"),
                "mask": acc.get("mask"),
                "currency": acc.get("currency"),
                "status": acc.get("status"),
                "balance": balance,
                "bucket": bucket,
                "plaid_last_successful_update": acc.get(
                    "plaid_last_successful_update"
                ),
            }
        )

    return {
        "totals": totals,
        "count": len(normalized_accounts),
        "accounts": normalized_accounts,
    }


@router.post("/transactions")
def sync_lm_transactions(
    payload: LmTransactionsSyncRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Pull transactions from Lunch Money (Plaid-backed) and store them in Mira DB.

    - Required: plaid_account_id (e.g. 317631 for M1 Div)
    - Optional: start_date, end_date (defaults: last 30 days)
    """
    client = get_client()

    end_date = payload.end_date or date.today()
    start_date = payload.start_date or (end_date - timedelta(days=30))
    plaid_account_id = payload.plaid_account_id

    try:
        txs: List[Dict[str, Any]] = client.get_transactions(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            plaid_account_id=plaid_account_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    created = 0
    skipped = 0

    for tx in txs:
        lm_tx_id = tx.get("id")
        if lm_tx_id is None:
            skipped += 1
            continue

        # Our LMTransaction.id *is* the Lunch Money transaction id
        existing = (
            db.query(LMTransaction)
            .filter(LMTransaction.id == lm_tx_id)
            .one_or_none()
        )

        if existing:
            skipped += 1
            continue

        amount_raw = tx.get("to_base") or tx.get("amount")
        try:
            amount = float(amount_raw) if amount_raw is not None else 0.0
        except (TypeError, ValueError):
            amount = 0.0

        obj = LMTransaction(
            id=lm_tx_id,
            lm_tx_id=lm_tx_id,
            plaid_account_id=tx.get("plaid_account_id"),
            date=tx.get("date"),
            amount=amount,
            payee=tx.get("payee"),
            raw=tx,  # assuming LMTransaction.raw is JSON/JSONB
        )
        db.add(obj)
        created += 1

    db.commit()

    # After persisting LM transactions, sync dividend events into dividend_events table
    tx_rows: List[LMTransaction] = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .filter(LMTransaction.date >= start_date.isoformat())
        .filter(LMTransaction.date <= end_date.isoformat())
        .all()
    )

    if tx_rows:
        sync_dividend_events_from_transactions(
            db=db,
            plaid_account_id=plaid_account_id,
            transactions=tx_rows,
        )
        db.commit()

    return {
        "source": "lunchmoney",
        "plaid_account_id": plaid_account_id,
        "created": created,
        "skipped": skipped,
        "count": len(txs),
    }


# Sync all Lunch Money transactions across all accounts for a date range
@router.post("/transactions/all")
def sync_all_lm_transactions(
    payload: LmTransactionsSyncAllRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Pull ALL transactions from Lunch Money (across all accounts) for a date range
    and store them in Mira DB.

    - No plaid_account_id filter is used, so this includes:
      • all Plaid-linked accounts (checking, savings, credit, investment)
      • any non-Plaid accounts managed in Lunch Money
    """
    client = get_client()

    end_date = payload.end_date or date.today()
    start_date = payload.start_date or (end_date - timedelta(days=30))

    try:
        txs: List[Dict[str, Any]] = client.get_transactions(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            plaid_account_id=None,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    created = 0
    skipped = 0

    for tx in txs:
        lm_tx_id = tx.get("id")
        if lm_tx_id is None:
            skipped += 1
            continue

        existing = (
            db.query(LMTransaction)
            .filter(LMTransaction.id == lm_tx_id)
            .one_or_none()
        )

        if existing:
            skipped += 1
            continue

        amount_raw = tx.get("to_base") or tx.get("amount")
        try:
            amount = float(amount_raw) if amount_raw is not None else 0.0
        except (TypeError, ValueError):
            amount = 0.0

        obj = LMTransaction(
            id=lm_tx_id,
            lm_tx_id=lm_tx_id,
            plaid_account_id=tx.get("plaid_account_id"),
            date=tx.get("date"),
            amount=amount,
            payee=tx.get("payee"),
            raw=tx,
        )
        db.add(obj)
        created += 1

    db.commit()

    # Collect unique plaid account IDs present in this batch for dividend sync
    plaid_ids: set[int] = set()
    for tx in txs:
        pid = tx.get("plaid_account_id")
        if pid is not None:
            try:
                plaid_ids.add(int(pid))
            except (TypeError, ValueError):
                continue

    # For each plaid account, sync dividend events over the same date window
    for pid in sorted(plaid_ids):
        tx_rows: List[LMTransaction] = (
            db.query(LMTransaction)
            .filter(LMTransaction.plaid_account_id == pid)
            .filter(LMTransaction.date >= start_date.isoformat())
            .filter(LMTransaction.date <= end_date.isoformat())
            .all()
        )

        if not tx_rows:
            continue

        sync_dividend_events_from_transactions(
            db=db,
            plaid_account_id=pid,
            transactions=tx_rows,
        )
        db.commit()

    return {
        "source": "lunchmoney",
        "mode": "all_accounts",
        "plaid_account_ids": sorted(plaid_ids),
        "created": created,
        "skipped": skipped,
        "count": len(txs),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

# --- Holdings reconstruction (v0) --- #

_SYMBOL_RE = re.compile(r"shares of ([A-Z\.]+) ")


def _extract_symbol_from_tx(raw_tx: dict) -> str | None:
    """
    Try to extract the ticker symbol from plaid_metadata.name or payee/original_name.
    Example name: "3.49046 shares of VNQ purchased. - PURCHASED"
    """
    name: str | None = None

    meta_str = raw_tx.get("plaid_metadata")
    if meta_str:
        try:
            meta = json.loads(meta_str)
            name = meta.get("name")
        except json.JSONDecodeError:
            pass

    if not name:
        name = raw_tx.get("payee") or raw_tx.get("original_name") or ""

    m = _SYMBOL_RE.search(name)
    return m.group(1) if m else None


def _reconstruct_holdings(
    db: Session,
    plaid_account_id: int,
    as_of: date,
) -> Dict[str, Any]:
    """
    Rebuild holdings for a Plaid-backed investment account from LMTransaction rows.
    - Uses plaid_metadata.quantity * sign(type) for share changes.
    - Sells reduce shares; fully sold positions are dropped.
    - Cost basis is sum of buy-side amounts (v0).
    """
    rows: List[LMTransaction] = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .filter(LMTransaction.date <= as_of.isoformat())
        .all()
    )

    holdings = defaultdict(lambda: {"shares": Decimal("0"), "cost_basis": Decimal("0"), "trades": 0})

    for row in rows:
        raw = row.raw
        if not isinstance(raw, dict):
            try:
                raw = json.loads(raw)
            except Exception:
                continue

        meta_str = raw.get("plaid_metadata")
        if not meta_str:
            continue

        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            continue

        inv_type = (meta.get("type") or meta.get("subtype") or "").lower()
        qty = Decimal(str(meta.get("quantity") or 0))
        price = Decimal(str(meta.get("price") or 0))
        symbol = _extract_symbol_from_tx(raw)

        if not symbol or qty == 0:
            continue

        # classify side
        if inv_type in ("buy", "dividend", "dividend reinvestment"):
            sign = Decimal("1")
        elif inv_type in ("sell", "sell_short", "sell to cover"):
            sign = Decimal("-1")
        else:
            # ignore fees, transfers, etc. for share counts
            continue

        delta_shares = sign * qty
        cost_delta = price * qty if sign > 0 else Decimal("0")

        h = holdings[symbol]
        h["shares"] += delta_shares
        h["cost_basis"] += cost_delta
        h["trades"] += 1

    result: List[Dict[str, Any]] = []
    for symbol, h in holdings.items():
        if h["shares"] <= 0:
            # fully sold or net short – skip for now
            continue
        avg_cost = (h["cost_basis"] / h["shares"]) if h["shares"] > 0 else Decimal("0")
        result.append(
            {
                "symbol": symbol,
                "shares": float(h["shares"]),
                "cost_basis": float(h["cost_basis"]),
                "avg_cost": float(avg_cost),
                "trades": h["trades"],
            }
        )

    return {
        "plaid_account_id": plaid_account_id,
        "as_of": as_of.isoformat(),
        "count": len(result),
        "holdings": result,
    }


@router.get("/holdings/reconstruct")
def holdings_reconstruct(
    plaid_account_id: int,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Reconstruct current holdings for a Plaid-backed account from LMTransaction rows.

    Example:
      GET /lm/sync/holdings/reconstruct?plaid_account_id=317631
    """
    as_of = as_of or date.today()
    return _reconstruct_holdings(db=db, plaid_account_id=plaid_account_id, as_of=as_of)
@router.get("/holdings/{plaid_account_id}")
def get_holdings(plaid_account_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Reconstruct current holdings for a given plaid_account_id
    using stored LMTransaction rows.
    """
    as_of = date.today()
    result = build_holdings_from_transactions(
        db=db,
        plaid_account_id=plaid_account_id,
        as_of=as_of,
    )
    return result