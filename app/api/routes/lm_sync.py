from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.settings import settings
from app.services.lm_client import LunchMoneyClient

router = APIRouter(prefix="/lm/sync", tags=["lunchmoney-sync"])


def get_client() -> LunchMoneyClient:
    return LunchMoneyClient(api_key=settings.lunchmoney_api_key)


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