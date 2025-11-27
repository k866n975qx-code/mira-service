from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.settings import settings
from app.services.lm_client import LunchMoneyClient

router = APIRouter(prefix="/lm/raw", tags=["lunchmoney-raw"])


def get_client() -> LunchMoneyClient:
    return LunchMoneyClient()


@router.get("/plaid_accounts")
def get_lm_plaid_accounts(
    db: Session = Depends(get_db),  # kept for future if we want to persist
) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        plaid_accounts = client.get_plaid_accounts()
        return plaid_accounts
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/transactions")
def get_lm_transactions(
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        txns = client.get_transactions()
        return txns
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))