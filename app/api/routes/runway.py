# app/api/routes/runway.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import Bill
from app.services.lm_client import LunchMoneyClient

router = APIRouter(
    prefix="/lm/reserves",
    tags=["reserves"],
)


def _get_liquidity() -> float:
    """
    Same logic as the funding and reserves endpoints:
    sum of cash-like Plaid accounts.
    """
    client = LunchMoneyClient()
    try:
        accounts = client.get_plaid_accounts()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error from Lunch Money: {e}")

    total = 0.0
    for acc in accounts:
        type_ = (acc.get("type") or "").lower()
        if type_ != "depository":
            continue
        try:
            bal = float(acc.get("balance") or 0)
        except Exception:
            bal = 0.0
        total += bal
    return total


def _compute_next_due_date(bill: Bill) -> date | None:
    """
    If next_due_date exists, use that.
    Else, compute based on due_day.
    """
    if bill.next_due_date:
        return bill.next_due_date

    if not bill.due_day:
        return None

    today = date.today()
    # if bill due day hasn't occurred yet this month:
    if today.day <= bill.due_day:
        return today.replace(day=bill.due_day)

    # else push into next month
    year = today.year + (1 if today.month == 12 else 0)
    month = 1 if today.month == 12 else today.month + 1
    return date(year, month, bill.due_day)


@router.get("/runway")
def runway(days_ahead: int = 30, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Next 30-day obligations + runway estimate.
    """

    today = date.today()

    # Fetch active bills
    bills = (
        db.query(Bill)
        .filter(Bill.is_active.is_(True))
        .all()
    )

    upcoming: List[Dict[str, Any]] = []
    total_required = 0.0

    for b in bills:
        due = _compute_next_due_date(b)
        if not due:
            continue

        if today <= due <= today + timedelta(days=days_ahead):
            amount = float(b.amount or 0)
            total_required += amount
            upcoming.append(
                {
                    "id": b.id,
                    "name": b.name,
                    "amount": amount,
                    "due_date": due.isoformat(),
                }
            )

    liquidity = _get_liquidity()
    shortfall = max(0.0, total_required - liquidity)

    # Compute runway days
    if total_required == 0:
        runway_days = float("inf")
    else:
        # If proportional: runway = liquidity / (daily obligation avg)
        runway_days = liquidity / (total_required / days_ahead)

    return {
        "as_of": today.isoformat(),
        "days_ahead": days_ahead,
        "bills_next_30": upcoming,
        "total_required": round(total_required, 2),
        "actual_liquidity": round(liquidity, 2),
        "shortfall": round(shortfall, 2),
        "runway_days": round(runway_days, 2),
    }