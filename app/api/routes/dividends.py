

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict
from app.api.deps import get_db
from app.infra.models import DividendEvent

router = APIRouter(
    prefix="/lm/dividends",
    tags=["dividends"],
)

@router.get("/events")
def get_dividend_events(
    plaid_account_id: Optional[int] = Query(
        default=None,
        description="Filter by Plaid account ID. If omitted, returns events for all accounts."
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Start of date range (inclusive, YYYY-MM-DD). If omitted, defaults to 30 days ago."
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="End of date range (inclusive, YYYY-MM-DD). If omitted, defaults to today."
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of events to return."
    ),
    db: Session = Depends(get_db),
) -> Dict:
    # Handle default dates
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    query = db.query(DividendEvent).filter(
        DividendEvent.pay_date >= start_date,
        DividendEvent.pay_date <= end_date,
    )
    if plaid_account_id is not None:
        query = query.filter(DividendEvent.plaid_account_id == plaid_account_id)
    query = query.order_by(
        DividendEvent.pay_date.desc(),
        DividendEvent.id.desc()
    ).limit(limit)

    events = []
    for row in query.all():
        events.append({
            "id": row.id,
            "plaid_account_id": row.plaid_account_id,
            "lm_transaction_id": row.lm_transaction_id,
            "pay_date": row.pay_date.isoformat() if row.pay_date else None,
            "amount": float(row.amount) if row.amount is not None else None,
            "currency": row.currency,
            "symbol": row.symbol,
            "cusip": getattr(row, "cusip", None),
            "source": row.source,
        })

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "count": len(events),
        "events": events,
    }