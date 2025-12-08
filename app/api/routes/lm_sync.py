from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import LMTransaction
from app.services.lm_client import LunchMoneyClient
from app.services.dividends import sync_dividend_events_from_transactions
from app.api.routes.holdings import get_valued_holdings_for_plaid_account

router = APIRouter(prefix="/lm/sync", tags=["lunchmoney"])



class LmTransactionsSyncAllRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


def get_client() -> LunchMoneyClient:
    """Construct a LunchMoney client using the configured API key."""
    return LunchMoneyClient()


# Sync all Lunch Money transactions across all accounts for a date range
@router.post("/transactions/all")
def sync_all_lm_transactions(
    payload: LmTransactionsSyncAllRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Sync M1 transactions for a date range (investment + Borrow) and store them in Mira DB.

    - Discovers M1 plaid_account_ids from Lunch Money and fetches each separately.
    - Persists LM transactions, then backfills dividend_events per account.
    """
    client = get_client()

    end_date = payload.end_date or date.today()
    start_date = payload.start_date or (end_date - timedelta(days=30))

    # Fetch LM transactions (M1-only; use plaid_account_id per API docs)
    try:
        accounts = client.get_plaid_accounts() or []
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    m1_accounts: List[Dict[str, Any]] = []
    for acc in accounts:
        inst = (acc.get("institution_name") or "").strip()
        name = (acc.get("name") or "").strip().lower()
        typ = (acc.get("type") or acc.get("account_type") or "").strip().lower()
        # two accounts only: M1 investment + M1 Borrow (loan)
        if inst == "M1 Finance" and (typ in {"investment", "loan"} or "borrow" in name):
            m1_accounts.append(acc)

    plaid_ids: List[int] = []
    txs: List[Dict[str, Any]] = []
    for acc in m1_accounts:
        pid = acc.get("id") or acc.get("plaid_account_id") or (acc.get("plaid_account") or {}).get("id")
        try:
            pid_int = int(pid)
        except Exception:
            continue
        plaid_ids.append(pid_int)

        try:
            batch = client.get_transactions(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                plaid_account_id=pid_int,   # <- the important part
            )
            if batch:
                txs.extend(batch)
        except Exception:
            # Skip this account but continue; we still want the other one
            continue

    # de-duplicate and sort the IDs (just in case)
    plaid_ids = sorted(list({int(x) for x in plaid_ids}))

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

    # For each plaid account, sync dividend events over the same date window
    for pid in plaid_ids:
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

    result = {
        "source": "lunchmoney",
        "mode": "m1_only",
        "plaid_account_ids": plaid_ids,
        "created": created,
        "skipped": skipped,
        "count": len(txs),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    # Rebuild snapshots for these accounts (best-effort) and warm dividend profiles
    try:
        today = date.today()
        for pid in plaid_ids:
            try:
                get_valued_holdings_for_plaid_account(
                    plaid_account_id=pid,
                    as_of=today,
                    refresh=True,
                    db=db,
                )
            except Exception:
                # don't fail the sync if snapshot rebuild has issues
                continue
    except Exception:
        pass

    return result
