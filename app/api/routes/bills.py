# app/api/routes/bills.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import Bill
from app.services.lm_client import LunchMoneyClient

router = APIRouter(
    prefix="/lm/bills",
    tags=["bills"],
)


# ---------- Pydantic Schemas ----------

class BillCreate(BaseModel):
    name: str
    category: Optional[str] = None
    amount: float
    frequency: str = "monthly"        # monthly|yearly|weekly
    due_day: Optional[int] = None
    auto_pay: bool = False
    linked_account_id: Optional[int] = None
    next_due_date: Optional[date] = None
    notes: Optional[str] = None


class BillUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[float] = None
    frequency: Optional[str] = None
    due_day: Optional[int] = None
    auto_pay: Optional[bool] = None
    linked_account_id: Optional[int] = None
    next_due_date: Optional[date] = None
    notes: Optional[str] = None
    funded_pct: Optional[float] = None
    is_active: Optional[bool] = None


# ---------- Routes ----------

@router.get("/")
def list_bills(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    bills = db.query(Bill).order_by(Bill.id.asc()).all()

    return [
        {
            "id": b.id,
            "name": b.name,
            "category": b.category,
            "amount": float(b.amount),
            "frequency": b.frequency,
            "due_day": b.due_day,
            "auto_pay": b.auto_pay,
            "funded_pct": float(b.funded_pct),
            "linked_account_id": b.linked_account_id,
            "next_due_date": (
                b.next_due_date.isoformat() if b.next_due_date else None
            ),
            "notes": b.notes,
            "is_active": b.is_active,
        }
        for b in bills
    ]


@router.post("/")
def create_bill(data: BillCreate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    bill = Bill(
        name=data.name,
        category=data.category,
        amount=data.amount,
        frequency=data.frequency,
        due_day=data.due_day,
        auto_pay=data.auto_pay,
        linked_account_id=data.linked_account_id,
        next_due_date=data.next_due_date,
        notes=data.notes,
        funded_pct=0,         # start unfunded
        is_active=True,
    )

    db.add(bill)
    db.commit()
    db.refresh(bill)

    return {"status": "ok", "id": bill.id}


@router.patch("/{bill_id}")
def update_bill(
    bill_id: int,
    data: BillUpdate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    bill = db.query(Bill).filter(Bill.id == bill_id).first()

    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")

    for field, value in data.dict(exclude_unset=True).items():
        setattr(bill, field, value)

    db.commit()
    db.refresh(bill)

    return {"status": "ok", "id": bill.id}


@router.get("/status")
def bills_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    bills = db.query(Bill).filter(Bill.is_active.is_(True)).all()

    counts = {"green": 0, "yellow": 0, "red": 0}

    def color(pct: float) -> str:
        if pct >= 100:
            return "green"
        elif pct >= 80:
            return "yellow"
        return "red"

    bill_list = []
    for b in bills:
        pct = float(b.funded_pct or 0)
        c = color(pct)
        counts[c] += 1

        bill_list.append(
            {
                "id": b.id,
                "name": b.name,
                "amount": float(b.amount),
                "funded_pct": pct,
                "color": c,
                "next_due_date": (
                    b.next_due_date.isoformat() if b.next_due_date else None
                ),
            }
        )

    return {
        "summary": counts,
        "bills": bill_list,
    }

def _get_cash_like_liquidity() -> float:
    """
    Sum balances of Plaid accounts considered 'cash_like'.

    For now: type == 'depository' (checking/savings).
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
        except (TypeError, ValueError):
            bal = 0.0

        total += bal

    return total


@router.post("/recalculate_funding")
def recalculate_bill_funding(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Recompute funded_pct for all active bills based on current cash-like liquidity.

    Greedy algo:
      - total_cash = sum(cash-like Plaid balances)
      - sort active bills by due_day (NULLS LAST), then id
      - allocate cash to each bill until exhausted
    """
    # fetch active bills
    bills = (
        db.query(Bill)
        .filter(Bill.is_active.is_(True))
        .order_by(Bill.due_day.is_(None).asc(), Bill.due_day.asc(), Bill.id.asc())
        .all()
    )

    if not bills:
        return {
            "status": "ok",
            "message": "No active bills to update",
            "total_liquidity": 0.0,
            "remaining_liquidity": 0.0,
            "bills": [],
        }

    total_cash = _get_cash_like_liquidity()
    remaining = total_cash

    updated = []

    for b in bills:
        amount = float(b.amount or 0.0)

        if amount <= 0:
            pct = 0.0
            allocated = 0.0
        else:
            allocated = min(remaining, amount)
            pct = (allocated / amount) * 100.0 if amount > 0 else 0.0

        # update remaining pool
        remaining = max(0.0, remaining - allocated)

        # clamp pct
        if pct > 100.0:
            pct = 100.0

        b.funded_pct = pct
        updated.append(
            {
                "id": b.id,
                "name": b.name,
                "amount": amount,
                "allocated": round(allocated, 2),
                "funded_pct": round(pct, 2),
            }
        )

    db.commit()

    return {
        "status": "ok",
        "total_liquidity": round(total_cash, 2),
        "remaining_liquidity": round(remaining, 2),
        "bills": updated,
    }