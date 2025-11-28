# app/api/routes/insights.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.infra.models import Bill
from app.infra.models import BudgetCategory, BudgetTarget
from app.api.routes.liquidity import (
    _compute_cashflow_snapshot,
    _bills_funding_overview,
    _r,
)
from app.api.routes.reserves import get_reserves_snapshot
from app.api.routes.runway import _compute_next_due_date, _get_liquidity

router = APIRouter(
    prefix="/lm/liquidity",
    tags=["liquidity"],
)


def _build_sinking_plan(db: Session) -> Dict[str, Any]:
    """
    For each bill, compute a recommended monthly sinking amount.

    Rules:
      - monthly:   full amount per month (effectively immediate)
      - yearly:    amount spread over months until next due date
      - weekly:    assume ~4 occurrences per month (amount * 4)
    """
    today = date.today()
    bills = (
        db.query(Bill)
        .filter(Bill.is_active.is_(True))
        .all()
    )

    items: List[Dict[str, Any]] = []
    total_required = 0.0

    for b in bills:
        amount = float(b.amount or 0.0)
        freq = (b.frequency or "monthly").lower()
        due = _compute_next_due_date(b)

        if amount <= 0:
            continue

        if freq == "weekly":
            # Approximate 4 weeks per month
            months_until_due = 1
            monthly_req = amount * 4.0
        elif freq == "yearly" and due is not None:
            months_until_due = max(
                1, (due.year - today.year) * 12 + (due.month - today.month)
            )
            monthly_req = amount / months_until_due
        else:
            # monthly or fallback
            months_until_due = 1
            monthly_req = amount

        total_required += monthly_req

        items.append(
            {
                "id": b.id,
                "name": b.name,
                "frequency": freq,
                "amount": _r(amount),
                "next_due_date": due.isoformat() if due else None,
                "months_until_due": months_until_due,
                "recommended_monthly": _r(monthly_req),
            }
        )

    return {
        "total_sinking_required_monthly": _r(total_required),
        "items": items,
    }
def _per_paycheck_factor(frequency: str) -> float:
    """
    Convert a monthly amount into a per-paycheck amount.

    Uses realistic 'per year' counts:

      - weekly   -> 52
      - biweekly -> 26
      - semimonthly -> 24
      - monthly  -> 12

    factor = 12 / paychecks_per_year
    per_paycheck = monthly_amount * factor
    """
    freq = (frequency or "").lower()
    if freq == "weekly":
        per_year = 52
    elif freq == "biweekly":
        per_year = 26
    elif freq in ("semimonthly", "semi-monthly"):
        per_year = 24
    else:
        # default: monthly-ish
        per_year = 12

    return 12.0 / per_year   

@router.get("/insights")
def liquidity_insights(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    High-level insights + allocation suggestions:

      - cashflow (last 30d)
      - reserves status
      - bill funding + sinking plan
      - allocation of excess:
          * up to 5% of excess -> emergency fund
          * rest -> investments (after sinking funds)
    """
    today = date.today()

    # --- base layers ---
    cashflow = _compute_cashflow_snapshot(db=db)  # last 30d
    reserves = get_reserves_snapshot(db=db)
    bills_funding = _bills_funding_overview(db=db)
    sinking_plan = _build_sinking_plan(db=db)
    liquidity = _get_liquidity()

    # --- core numbers ---
    net_30d = float(cashflow["net_cashflow"])
    excess_monthly = max(0.0, net_30d)

    emergency_target = float(reserves["emergency_target"])
    actual_liquidity = float(reserves["actual_liquidity"])
    emergency_gap = max(0.0, emergency_target - actual_liquidity)

    # rule: cap EF contribution at 5% of excess per month
    max_ef_contribution = min(emergency_gap, excess_monthly * 0.05)

    total_sinking_required = float(
        sinking_plan["total_sinking_required_monthly"]
    )

    available_after_ef = max(0.0, excess_monthly - max_ef_contribution)
    sinking_shortfall = max(0.0, total_sinking_required - available_after_ef)

    # whatever remains after EF + full sinking coverage -> investments
    investment_allocation = max(
        0.0, available_after_ef - total_sinking_required
    )

    # --- clean/round ---
    excess_monthly = _r(excess_monthly)
    emergency_gap = _r(emergency_gap)
    max_ef_contribution = _r(max_ef_contribution)
    total_sinking_required = _r(total_sinking_required)
    sinking_shortfall = _r(sinking_shortfall)
    investment_allocation = _r(investment_allocation)

    messages: List[str] = []

    # EF message
    if emergency_gap <= 0:
        messages.append("Emergency fund is at or above target.")
    else:
        messages.append(
            f"Emergency fund is {emergency_gap:.2f} below target; "
            f"allocate up to {max_ef_contribution:.2f} this month (5% cap of excess)."
        )

    # Sinking fund message
    if total_sinking_required == 0:
        messages.append("No sinking-fund style bills configured yet.")
    else:
        if sinking_shortfall > 0:
            messages.append(
                f"Sinking funds need {total_sinking_required:.2f}/month; "
                f"you are short by ~{sinking_shortfall:.2f} given current excess."
            )
        else:
            messages.append(
                f"Sinking funds of {total_sinking_required:.2f}/month can be fully covered by current excess."
            )

    # Investment message
    if investment_allocation > 0:
        messages.append(
            f"After EF and sinking funds, you can allocate ~{investment_allocation:.2f}/month towards investments."
        )
    else:
        messages.append(
            "No consistent monthly excess available for investments after EF and sinking funds right now."
        )

    return {
        "as_of": today.isoformat(),
        "cashflow": {
            "income_total": cashflow["income_total"],
            "spending_total": cashflow["spending_total"],
            "net_30d": cashflow["net_cashflow"],
            "excess_monthly": excess_monthly,
        },
        "reserves": {
            "coverage_pct": _r(reserves["coverage_pct"]),
            "status": reserves["status"],
            "emergency_target": _r(emergency_target),
            "actual_liquidity": _r(actual_liquidity),
            "emergency_gap": emergency_gap,
        },
        "bills": {
            "funding": bills_funding,
            "sinking_plan": sinking_plan,
        },
        "allocation_plan": {
            "excess_monthly": excess_monthly,
            "max_emergency_fund_contribution": max_ef_contribution,
            "total_sinking_required_monthly": total_sinking_required,
            "sinking_shortfall": sinking_shortfall,
            "investment_allocation": investment_allocation,
            "liquidity_now": _r(liquidity),
        },
        "messages": messages,
    }
def _compute_spending_budgets_for_paycheck(
    db: Session,
    to_spend: float,
    frequency: str,
) -> Dict[str, Any]:
    """
    Split the 'to_spend' amount into budget categories based on monthly targets.

    - Takes monthly budget targets (Food, Misc, etc.)
    - Converts them into per-paycheck nominal amounts
    - Scales them down if they exceed `to_spend`
    - Returns per-category per-paycheck suggestions + unallocated residual
    """
    factor = _per_paycheck_factor(frequency)

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

    per_cat_nominal = []
    total_nominal = 0.0

    for target, cat in rows:
        monthly_target = float(target.target_amount or 0.0)
        if monthly_target <= 0:
            continue

        per_check_target = monthly_target * factor
        total_nominal += per_check_target
        per_cat_nominal.append(
            {
                "category_id": cat.id,
                "name": cat.name,
                "monthly_target": monthly_target,
                "per_check_nominal": per_check_target,
            }
        )

    if to_spend <= 0 or total_nominal <= 0 or not per_cat_nominal:
        # nothing to allocate
        return {
            "per_paycheck_total": 0.0,
            "unallocated": _r(to_spend),
            "categories": [],
        }

    # Scale budgets down if they exceed the spend bucket
    if total_nominal > to_spend:
        scale = to_spend / total_nominal
    else:
        scale = 1.0

    assigned_total = 0.0
    categories_out = []

    for item in per_cat_nominal:
        assigned = item["per_check_nominal"] * scale
        assigned_total += assigned
        categories_out.append(
            {
                "category_id": item["category_id"],
                "name": item["name"],
                "per_paycheck": _r(assigned),
                "monthly_target": _r(item["monthly_target"]),
            }
        )

    unallocated = max(0.0, to_spend - assigned_total)

    return {
        "per_paycheck_total": _r(assigned_total),
        "unallocated": _r(unallocated),
        "categories": categories_out,
    }
@router.get("/paycheck_plan")
def paycheck_plan(
    amount: float,
    frequency: str = "biweekly",
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Take a paycheck amount and suggest how to split it:

      - to emergency fund (capped at 5% of monthly excess, converted per-paycheck)
      - to sinking funds (bills + future credit payoff as we add it)
      - to investments
      - to spend
      - and split 'to_spend' into budget categories based on monthly targets
    """

    # Reuse the core allocation logic we already built
    insights = liquidity_insights(db=db)
    alloc = insights["allocation_plan"]

    monthly_excess = float(alloc["excess_monthly"])
    monthly_to_ef = float(alloc["max_emergency_fund_contribution"])
    monthly_to_sinking = float(alloc["total_sinking_required_monthly"])
    monthly_to_invest = float(alloc["investment_allocation"])

    factor = _per_paycheck_factor(frequency)

    # Nominal per-paycheck suggestions (before scaling to fit paycheck amount)
    pp_ef_nominal = monthly_to_ef * factor
    pp_sinking_nominal = monthly_to_sinking * factor
    pp_invest_nominal = monthly_to_invest * factor

    total_nominal = pp_ef_nominal + pp_sinking_nominal + pp_invest_nominal

    if total_nominal <= 0:
        # Nothing to allocate per the policy, everything is "to spend"
        spending_budgets = _compute_spending_budgets_for_paycheck(
            db=db,
            to_spend=amount,
            frequency=frequency,
        )

        return {
            "as_of": insights["as_of"],
            "frequency": frequency,
            "paycheck_amount": _r(amount),
            "per_year_factor": _per_paycheck_factor(frequency),
            "base_monthly_allocation": alloc,
            "per_paycheck_plan": {
                "to_emergency_fund": 0.0,
                "to_sinking_funds": 0.0,
                "to_investments": 0.0,
                "to_spend": _r(amount),
                "spending_budgets": spending_budgets,
            },
            "notes": [
                "No positive excess / allocations from policy; entire paycheck is available to spend."
            ],
        }

    # Make sure we never allocate more than the paycheck amount:
    if total_nominal > amount:
        scale = amount / total_nominal
    else:
        scale = 1.0

    to_ef = pp_ef_nominal * scale
    to_sinking = pp_sinking_nominal * scale
    to_invest = pp_invest_nominal * scale

    allocated = to_ef + to_sinking + to_invest
    to_spend = amount - allocated

    # Split the 'to_spend' chunk into budgets
    spending_budgets = _compute_spending_budgets_for_paycheck(
        db=db,
        to_spend=to_spend,
        frequency=frequency,
    )

    plan = {
        "to_emergency_fund": _r(to_ef),
        "to_sinking_funds": _r(to_sinking),
        "to_investments": _r(to_invest),
        "to_spend": _r(to_spend),
        "spending_budgets": spending_budgets,
    }

    return {
        "as_of": insights["as_of"],
        "frequency": frequency,
        "paycheck_amount": _r(amount),
        "per_year_factor": _per_paycheck_factor(frequency),
        "base_monthly_allocation": alloc,
        "per_paycheck_plan": plan,
    }