# app/api/routes/reserves.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import Bill, LMTransaction, ReserveSnapshot
from app.services.lm_client import LunchMoneyClient

router = APIRouter(
    prefix="/lm/reserves",
    tags=["reserves"],
)


def _color_from_coverage(pct: float) -> str:
    if pct >= 100:
        return "green"
    if pct >= 90:
        return "yellow"
    return "red"


def _get_actual_liquidity_from_plaid() -> float:
    """
    Use Lunch Money /plaid_accounts as the source of truth for liquid balances.

    For now, treat:
      - type == "depository" as cash_like and sum all of them.
    You can tighten this later if you want to exclude certain accounts.
    """
    client = LunchMoneyClient()
    try:
        accounts: List[Dict[str, Any]] = client.get_plaid_accounts()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error from Lunch Money: {e}")

    total_cash_like = 0.0
    for acc in accounts:
        type_ = (acc.get("type") or "").lower()
        if type_ == "depository":
            try:
                bal = float(acc.get("balance") or 0)
            except (TypeError, ValueError):
                bal = 0.0
            total_cash_like += bal

    return total_cash_like


def _estimate_avg_monthly_spend(db: Session, days: int = 90) -> float:
    """
    Estimate average monthly spending from LMTransaction.

    - Looks back `days` (default 90).
    - Uses NEGATIVE amounts as spending.
    - Returns a 30-day equivalent.
    """
    end = date.today()
    start = end - timedelta(days=days)

    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.date >= start)
        .filter(LMTransaction.date <= end)
        .all()
    )

    total_spend = 0.0
    for tx in txs:
        amt = float(tx.amount or 0)
        if amt < 0:
            total_spend += abs(amt)

    if days <= 0:
        return 0.0

    # scale to a 30-day "month"
    daily = total_spend / days
    return daily * 30.0


@router.get("/snapshot")
def get_reserves_snapshot(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Compute and persist a reserves snapshot.

    Based on your v1.5 blueprint:

      bills_buffer      = Σ(active bills)  [MVP: all active, refine to 30d window later]
      emergency_target  = avg_monthly_spend × 4
      margin_safety_net = 0 for now (no margin accounts wired yet)
      total_recommended = sum(all above)
      actual_liquidity  = sum cash-like Plaid balances
      coverage_pct      = (actual_liquidity / total_recommended) × 100
    """
    # 1) bills buffer
    active_bills = db.query(Bill).filter(Bill.is_active.is_(True)).all()
    bills_buffer = sum(float(b.amount or 0) for b in active_bills)

    # 2) emergency target from LMTransaction history (4 months by default)
    avg_monthly_spend = _estimate_avg_monthly_spend(db, days=90)
    emergency_months = 4  # can move to config later
    emergency_target = avg_monthly_spend * emergency_months

    # 3) margin safety net (placeholder 0 until margin accounts are wired)
    margin_safety_net = 0.0

    # 4) actual liquidity from Plaid (cash-like)
    actual_liquidity = _get_actual_liquidity_from_plaid()

    # 5) totals + coverage
    total_recommended = bills_buffer + emergency_target + margin_safety_net

    if total_recommended <= 0:
        coverage_pct = 0.0
    else:
        coverage_pct = (actual_liquidity / total_recommended) * 100.0

    status = _color_from_coverage(coverage_pct)

    today = date.today()

    # 6) persist snapshot
    snap = ReserveSnapshot(
        as_of=today,
        bills_buffer=bills_buffer,
        emergency_target=emergency_target,
        margin_safety_net=margin_safety_net,
        total_recommended=total_recommended,
        actual_liquidity=actual_liquidity,
        coverage_pct=coverage_pct,
        status=status,
        details={
            "avg_monthly_spend": avg_monthly_spend,
            "emergency_months": emergency_months,
            "bills_count": len(active_bills),
        },
    )

    db.add(snap)
    db.commit()
    db.refresh(snap)

    return {
        "as_of": today.isoformat(),
        "bills_buffer": bills_buffer,
        "emergency_target": emergency_target,
        "margin_safety_net": margin_safety_net,
        "total_recommended": total_recommended,
        "actual_liquidity": actual_liquidity,
        "coverage_pct": coverage_pct,
        "status": status,
        "details": snap.details,
        "snapshot_id": snap.id,
    }