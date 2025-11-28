# app/api/routes/liquidity.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.routes.accounts import get_accounts_snapshot
from app.api.routes.reserves import get_reserves_snapshot
from app.infra.models import LMTransaction
from app.infra.models import Bill

router = APIRouter(
    prefix="/lm/liquidity",
    tags=["liquidity"],
)


def _r(x: float | int | None) -> float:
    """Round values to 2 decimals, handle None safely."""
    if x is None:
        return 0.0
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0


def _compute_cashflow_snapshot(db: Session, days: int = 30) -> Dict[str, Any]:
    """
    Internal cashflow calc, same logic as /lm/cashflow/snapshot
    but without FastAPI Query objects.
    """
    end = date.today()
    start = end - timedelta(days=days)

    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.date >= start)
        .filter(LMTransaction.date <= end)
        .all()
    )

    income_total = 0.0
    spending_total = 0.0

    for tx in txs:
        amt = float(tx.amount or 0)
        if amt > 0:
            income_total += amt
        else:
            spending_total += abs(amt)

    net = income_total - spending_total

    return {
        "as_of": end.isoformat(),
        "range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "income_total": _r(income_total),
        "spending_total": _r(spending_total),
        "net_cashflow": _r(net),
        "transactions_count": len(txs),
    }


def _bills_funding_overview(db: Session) -> Dict[str, Any]:
    """
    Snapshot of bill funding, based on Bill.funded_pct.

    Returns:
      - summary: counts by color
      - unfunded: list of bills with funded_pct < 100
    """
    bills = (
        db.query(Bill)
        .filter(Bill.is_active.is_(True))
        .order_by(Bill.due_day.is_(None).asc(), Bill.due_day.asc(), Bill.id.asc())
        .all()
    )

    summary = {"green": 0, "yellow": 0, "red": 0}
    unfunded = []

    def color(pct: float) -> str:
        if pct >= 100:
            return "green"
        if pct >= 80:
            return "yellow"
        return "red"

    for b in bills:
        pct = float(b.funded_pct or 0)
        c = color(pct)
        summary[c] += 1

        amount = float(b.amount or 0)
        missing = max(0.0, amount - (amount * min(pct, 100.0) / 100.0))

        if pct < 100:
            unfunded.append(
                {
                    "id": b.id,
                    "name": b.name,
                    "amount": _r(amount),
                    "funded_pct": _r(pct),
                    "missing_amount": _r(missing),
                    "due_day": b.due_day,
                }
            )

    return {
        "summary": summary,
        "unfunded": unfunded,
    }

@router.get("/summary")
def liquidity_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Unified liquidity snapshot:
      - accounts layer
      - cashflow layer (last 30 days)
      - reserves layer (rounded)
      - bills funding overview
    """
    accounts = get_accounts_snapshot()
    cashflow = _compute_cashflow_snapshot(db=db)
    reserves = get_reserves_snapshot(db=db)
    bills_funding = _bills_funding_overview(db=db)

    # --- clean up account totals ---
    totals = accounts.get("totals") or {}
    accounts["totals"] = {k: _r(v) for k, v in totals.items()}

    # --- clean up reserves ---
    details = dict(reserves.get("details") or {})
    if "avg_monthly_spend" in details:
        details["avg_monthly_spend"] = _r(details["avg_monthly_spend"])

    reserves_clean = {
        "as_of": reserves["as_of"],
        "bills_buffer": _r(reserves["bills_buffer"]),
        "emergency_target": _r(reserves["emergency_target"]),
        "margin_safety_net": _r(reserves["margin_safety_net"]),
        "total_recommended": _r(reserves["total_recommended"]),
        "actual_liquidity": _r(reserves["actual_liquidity"]),
        "coverage_pct": _r(reserves["coverage_pct"]),
        "status": reserves["status"],
        "details": details,
        "snapshot_id": reserves["snapshot_id"],
    }

    return {
        "as_of": date.today().isoformat(),
        "accounts": accounts,
        "cashflow": cashflow,
        "reserves": reserves_clean,
        "bills_funding": bills_funding,
    }