

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import SecurityMapping

router = APIRouter(
    prefix="/lm/securities",
    tags=["securities"],
)


class SecurityMappingCreate(BaseModel):
    cusip: str
    ticker: str
    name: Optional[str] = None
    notes: Optional[str] = None


@router.get("/mappings")
def list_security_mappings(
    db: Session = Depends(get_db),
    cusip: Optional[str] = Query(
        default=None,
        description="Optional CUSIP filter; if provided, returns at most one mapping.",
    ),
) -> Dict[str, Any]:
    """
    List security mappings. If `cusip` is provided, return only that mapping (if any).
    """
    q = db.query(SecurityMapping).filter(SecurityMapping.is_active.is_(True))

    if cusip:
        q = q.filter(SecurityMapping.cusip == cusip.strip().upper())

    rows: List[SecurityMapping] = q.order_by(SecurityMapping.cusip.asc()).all()

    items: List[Dict[str, Any]] = []
    for m in rows:
        items.append(
            {
                "id": m.id,
                "cusip": m.cusip,
                "ticker": m.ticker,
                "name": m.name,
                "notes": m.notes,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
        )

    return {
        "count": len(items),
        "mappings": items,
    }


@router.post("/mappings")
def upsert_security_mapping(
    payload: SecurityMappingCreate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create or update a CUSIP → ticker mapping.

    - If a mapping for this CUSIP already exists, update its fields.
    - Otherwise, create a new mapping.

    This is intended to be called by Mira in chat when Jose confirms a ticker
    for a given CUSIP.
    """
    cusip = payload.cusip.strip().upper()
    ticker = payload.ticker.strip().upper()

    if not cusip:
        raise HTTPException(status_code=400, detail="CUSIP cannot be empty.")
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")

    existing: Optional[SecurityMapping] = (
        db.query(SecurityMapping)
        .filter(SecurityMapping.cusip == cusip)  # BUG: wrong column name
        .one_or_none()
    )

    if existing:
        existing.ticker = ticker
        if payload.name is not None:
            existing.name = payload.name
        if payload.notes is not None:
            existing.notes = payload.notes
        existing.is_active = True
        db.commit()
        db.refresh(existing)

        status = "updated"
        obj = existing
    else:
        obj = SecurityMapping(
            cusip=cusip,
            ticker=ticker,
            name=payload.name,
            notes=payload.notes,
            is_active=True,
        )
        db.add(obj)
        db.commit()
        db.refresh(obj)
        status = "created"

    return {
        "status": status,
        "id": obj.id,
        "cusip": obj.cusip,
        "ticker": obj.ticker,
        "name": obj.name,
        "notes": obj.notes,
        "is_active": obj.is_active,
    }