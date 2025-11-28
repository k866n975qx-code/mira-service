# app/api/routes/cashflow.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import LMTransaction

router = APIRouter(
    prefix="/lm/cashflow",
    tags=["cashflow"],
)


@router.get("/snapshot")
def get_cashflow_snapshot(
    start: date = Query(None, description="Start date (YYYY-MM-DD)"),
    end: date = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Simple cashflow snapshot:
      - SUM(inflows)
      - SUM(outflows)
      - net_cashflow = inflows - outflows
    """

    # Default range: last 30 days
    if start is None or end is None:
        end = date.today()
        start = end - timedelta(days=30)

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

        # LMTransaction’s "amount" is POSITIVE for income, NEGATIVE for spending
        if amt > 0:
            income_total += amt
        else:
            spending_total += abs(amt)

    net = income_total - spending_total

    return {
        "as_of": date.today().isoformat(),
        "range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "income_total": income_total,
        "spending_total": spending_total,
        "net_cashflow": net,
        "transactions_count": len(txs),
    }