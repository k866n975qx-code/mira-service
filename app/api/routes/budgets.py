# app/api/routes/budgets.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from app.api.deps import get_db
from app.infra.models import (
    LMTransaction,
    BudgetCategory,
    BudgetTarget,
    TransactionBudget,
)

router = APIRouter(
    prefix="/lm/budgets",
    tags=["budgets"],
)


class BudgetCategoryCreate(BaseModel):
    name: str
    description: str | None = None


class BudgetAssign(BaseModel):
    transaction_id: int
    category_id: int

class BudgetTargetCreate(BaseModel):
    category_id: int
    target_amount: float
    period: str = "monthly"

def _r(x: float | int | None) -> float:
    if x is None:
        return 0.0
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0

@router.get("/categories")
def list_budget_categories(db: Session = Depends(get_db)) -> Dict[str, Any]:
    rows = (
        db.query(BudgetCategory)
        .filter(BudgetCategory.is_active.is_(True))
        .order_by(BudgetCategory.name.asc())
        .all()
    )

    items: List[Dict[str, Any]] = []
    for cat in rows:
        items.append(
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "is_active": cat.is_active,
            }
        )

    return {
        "count": len(items),
        "categories": items,
    }


@router.post("/categories")
def create_budget_category(
    data: BudgetCategoryCreate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # check if a category with this name already exists
    existing = (
        db.query(BudgetCategory)
        .filter(BudgetCategory.name.ilike(data.name))
        .first()
    )
    if existing:
        return {
            "status": "exists",
            "id": existing.id,
            "name": existing.name,
        }

    cat = BudgetCategory(
        name=data.name,
        description=data.description,
        is_active=True,
    )
    db.add(cat)
    db.commit()
    db.refresh(cat)

    return {
        "status": "ok",
        "id": cat.id,
        "name": cat.name,
    }

@router.get("/targets")
def list_budget_targets(db: Session = Depends(get_db)) -> Dict[str, Any]:
    rows = (
        db.query(BudgetTarget, BudgetCategory)
        .join(BudgetCategory, BudgetCategory.id == BudgetTarget.budget_category_id)
        .filter(
            BudgetTarget.is_active.is_(True),
            BudgetTarget.period == "monthly",
            BudgetCategory.is_active.is_(True),
        )
        .order_by(BudgetCategory.name.asc())
        .all()
    )

    items: List[Dict[str, Any]] = []
    for target, cat in rows:
        items.append(
            {
                "id": target.id,
                "category_id": cat.id,
                "category_name": cat.name,
                "period": target.period,
                "target_amount": _r(target.target_amount),
                "effective_from": (
                    target.effective_from.isoformat()
                    if target.effective_from
                    else None
                ),
                "effective_to": (
                    target.effective_to.isoformat()
                    if target.effective_to
                    else None
                ),
                "is_active": target.is_active,
            }
        )

    return {
        "count": len(items),
        "targets": items,
    }


@router.post("/targets")
def upsert_budget_target(
    payload: BudgetTargetCreate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create or update a monthly target for a budget category.

    - If an active monthly target exists for this category, update it.
    - Otherwise, create a new one.
    """
    # ensure category exists
    cat = (
        db.query(BudgetCategory)
        .filter(
            BudgetCategory.id == payload.category_id,
            BudgetCategory.is_active.is_(True),
        )
        .first()
    )
    if not cat:
        return {
            "status": "error",
            "error": "category_not_found",
        }

    period = (payload.period or "monthly").lower()
    if period != "monthly":
        return {
            "status": "error",
            "error": "unsupported_period",
            "detail": "Only 'monthly' period is supported currently.",
        }

    target = (
        db.query(BudgetTarget)
        .filter(
            BudgetTarget.budget_category_id == payload.category_id,
            BudgetTarget.period == "monthly",
            BudgetTarget.is_active.is_(True),
        )
        .first()
    )

    if target:
        target.target_amount = payload.target_amount
    else:
        target = BudgetTarget(
            budget_category_id=payload.category_id,
            period="monthly",
            target_amount=payload.target_amount,
            is_active=True,
        )
        db.add(target)

    db.commit()
    db.refresh(target)

    return {
        "status": "ok",
        "id": target.id,
        "category_id": payload.category_id,
        "category_name": cat.name,
        "period": target.period,
        "target_amount": _r(target.target_amount),
    }

@router.get("/summary")
def budgets_summary(
    db: Session = Depends(get_db),
    days_for_avg: int = 90,
) -> Dict[str, Any]:
    """
    Summarize spending by budget category.

    - month_to_date: from first of current month to today
    - avg_monthly_spend: based on last `days_for_avg` days of history, scaled to 30d
    - target_monthly: from BudgetTarget (period='monthly'), if present
    """

    today = date.today()
    month_start = today.replace(day=1)

    avg_start = today - timedelta(days=days_for_avg)

    # ---- month-to-date spend per category ----
    mtd_rows = (
        db.query(
            TransactionBudget.budget_category_id,
            BudgetCategory.name,
            func.sum(func.abs(LMTransaction.amount)).label("spend_mtd"),
        )
        .select_from(LMTransaction)
        .outerjoin(
            TransactionBudget,
            TransactionBudget.lm_transaction_id == LMTransaction.id,
        )
        .outerjoin(
            BudgetCategory,
            BudgetCategory.id == TransactionBudget.budget_category_id,
        )
        .filter(
            LMTransaction.amount < 0,
            LMTransaction.date >= month_start,
            LMTransaction.date <= today,
        )
        .group_by(TransactionBudget.budget_category_id, BudgetCategory.name)
        .all()
    )

    # ---- last-N-days total spend per category for avg ----
    avg_rows = (
        db.query(
            TransactionBudget.budget_category_id,
            BudgetCategory.name,
            func.sum(func.abs(LMTransaction.amount)).label("spend_window"),
        )
        .select_from(LMTransaction)
        .outerjoin(
            TransactionBudget,
            TransactionBudget.lm_transaction_id == LMTransaction.id,
        )
        .outerjoin(
            BudgetCategory,
            BudgetCategory.id == TransactionBudget.budget_category_id,
        )
        .filter(
            LMTransaction.amount < 0,
            LMTransaction.date >= avg_start,
            LMTransaction.date <= today,
        )
        .group_by(TransactionBudget.budget_category_id, BudgetCategory.name)
        .all()
    )

    # ---- targets per category ----
    target_rows = (
        db.query(
            BudgetTarget.budget_category_id,
            BudgetCategory.name,
            BudgetTarget.target_amount,
        )
        .join(BudgetCategory, BudgetCategory.id == BudgetTarget.budget_category_id)
        .filter(
            BudgetTarget.is_active.is_(True),
            BudgetTarget.period == "monthly",
        )
        .all()
    )

    mtd_map: Dict[Optional[int], float] = {}
    avg_map: Dict[Optional[int], float] = {}
    target_map: Dict[Optional[int], float] = {}
    name_map: Dict[Optional[int], str] = {}

    for cat_id, name, spend in mtd_rows:
        mtd_map[cat_id] = float(spend or 0)
        if cat_id not in name_map:
            name_map[cat_id] = name or "Uncategorized"

    for cat_id, name, spend in avg_rows:
        if days_for_avg > 0:
            daily = float(spend or 0) / days_for_avg
            avg_monthly = daily * 30.0
        else:
            avg_monthly = 0.0
        avg_map[cat_id] = avg_monthly
        if cat_id not in name_map:
            name_map[cat_id] = name or "Uncategorized"

    for cat_id, name, target in target_rows:
        target_map[cat_id] = float(target or 0)
        if cat_id not in name_map:
            name_map[cat_id] = name or "Uncategorized"

    # Ensure we have an entry for uncategorized (None key)
    if None not in name_map:
        name_map[None] = "Uncategorized"

    categories: List[Dict[str, Any]] = []

    # Build unified list of category IDs (including None)
    all_cat_ids = set(mtd_map.keys()) | set(avg_map.keys()) | set(target_map.keys()) | {
        None
    }

    for cat_id in sorted(
        all_cat_ids, key=lambda x: (x is None, name_map.get(x, ""))
    ):
        name = name_map.get(cat_id, "Uncategorized")
        mtd = mtd_map.get(cat_id, 0.0)
        avg_monthly_spend = avg_map.get(cat_id, 0.0)
        target = target_map.get(cat_id, 0.0)

        remaining = target - mtd if target > 0 else 0.0

        if target <= 0:
            status = "no_target"
        elif mtd <= target:
            status = "under_target"
        else:
            status = "over_target"

        categories.append(
            {
                "category_id": cat_id,
                "name": name,
                "month_to_date": _r(mtd),
                "avg_monthly_spend": _r(avg_monthly_spend),
                "target_monthly": _r(target),
                "remaining_to_target": _r(remaining),
                "status": status,
            }
        )

    # Total MTD spend for all categories (including uncategorized)
    total_mtd_spend = sum(c["month_to_date"] for c in categories)

    return {
        "as_of": today.isoformat(),
        "range": {
            "start": month_start.isoformat(),
            "end": today.isoformat(),
        },
        "days_for_avg": days_for_avg,
        "total_month_to_date_spend": _r(total_mtd_spend),
        "categories": categories,
    }
@router.get("/uncategorized")
def budgets_uncategorized(
    db: Session = Depends(get_db),
    days: int = 60,
) -> Dict[str, Any]:
    """
    Return spending transactions (amount < 0) within the last `days`
    that do NOT have a budget category assigned.

    These are the items the user needs to sort into buckets.
    """

    today = date.today()
    start = today - timedelta(days=days)

    rows = (
        db.query(
            LMTransaction.id,
            LMTransaction.date,
            LMTransaction.payee,
            LMTransaction.amount,
            LMTransaction.currency,
            LMTransaction.account_id,
            LMTransaction.raw,
        )
        .outerjoin(
            TransactionBudget,
            TransactionBudget.lm_transaction_id == LMTransaction.id,
        )
        .filter(
            LMTransaction.amount < 0,         # spending
            LMTransaction.date >= start,      # last N days
            LMTransaction.date <= today,
            TransactionBudget.id.is_(None),   # no budget mapping
        )
        .order_by(LMTransaction.date.desc())
        .all()
    )

    tx_list = []
    for (tx_id, tx_date, payee, amount, currency, acct_id, raw) in rows:
        tx_list.append(
            {
                "transaction_id": tx_id,
                "date": tx_date.isoformat(),
                "payee": payee or "",
                "amount": float(amount or 0),
                "currency": currency or "usd",
                "account_id": acct_id,
                "raw": raw,
            }
        )

    return {
        "as_of": today.isoformat(),
        "range": {
            "start": start.isoformat(),
            "end": today.isoformat(),
        },
        "count": len(tx_list),
        "transactions": tx_list,
    }
@router.post("/assign")
def assign_transaction_to_budget(
    payload: BudgetAssign,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Attach a transaction to a budget category.

    - If a mapping already exists, update it.
    - If not, create it.
    """

    # make sure transaction exists
    tx = db.query(LMTransaction).filter(LMTransaction.id == payload.transaction_id).first()
    if not tx:
        return {
            "status": "error",
            "error": "transaction_not_found",
        }

    # make sure category exists and is active
    cat = (
        db.query(BudgetCategory)
        .filter(
            BudgetCategory.id == payload.category_id,
            BudgetCategory.is_active.is_(True),
        )
        .first()
    )
    if not cat:
        return {
            "status": "error",
            "error": "category_not_found",
        }

    link = (
        db.query(TransactionBudget)
        .filter(TransactionBudget.lm_transaction_id == payload.transaction_id)
        .first()
    )

    if link:
        link.budget_category_id = payload.category_id
    else:
        link = TransactionBudget(
            lm_transaction_id=payload.transaction_id,
            budget_category_id=payload.category_id,
        )
        db.add(link)

    db.commit()
    db.refresh(link)

    return {
        "status": "ok",
        "transaction_id": payload.transaction_id,
        "category_id": payload.category_id,
        "category_name": cat.name,
    }