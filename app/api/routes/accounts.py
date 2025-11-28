from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from app.services.lm_client import LunchMoneyClient

router = APIRouter(
    prefix="/lm/accounts",
    tags=["accounts"],
)


def _bucket_from_plaid_type(type_: str, subtype: str | None) -> str:
    """
    Very simple bucketing aligned with lm_sync v0:
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


@router.get("/snapshot")
def get_accounts_snapshot() -> Dict[str, Any]:
    """
    Return a snapshot of all Plaid-linked accounts from Lunch Money.

    This is built directly on /plaid_accounts via LunchMoneyClient and mirrors
    the bucketing logic already used in /lm/sync/plaid_accounts.
    """
    client = LunchMoneyClient()
    try:
        raw_accounts: List[Dict[str, Any]] = client.get_plaid_accounts()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error from Lunch Money: {e}")

    accounts_payload: List[Dict[str, Any]] = []

    totals: Dict[str, float] = {
        "all": 0.0,
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

        # totals
        totals["all"] += balance
        totals[bucket] = totals.get(bucket, 0.0) + balance

        accounts_payload.append(
            {
                "id": acc.get("id"),
                "name": acc.get("display_name") or acc.get("name"),
                "type": type_,
                "subtype": subtype,
                "institution_name": acc.get("institution_name"),
                "mask": acc.get("mask"),
                "currency": (acc.get("currency") or "USD").lower(),
                "status": acc.get("status"),
                "balance": balance,
                "bucket": bucket,
                "plaid_last_successful_update": acc.get(
                    "plaid_last_successful_update"
                ),
            }
        )

    return {
        "as_of": date.today().isoformat(),
        "accounts": accounts_payload,
        "totals": totals,
    }