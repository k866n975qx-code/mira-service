from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.services.lm_client import LunchMoneyClient
from app.api.deps import get_db
from app.infra.models import LMTransaction, LMAccount, HoldingSnapshot
from app.services.holdings import reconstruct_holdings
from app.services.pricing import get_latest_prices
from app.services.dividends import (
    extract_dividend_events,
    summarize_dividends,
    estimate_forward_dividends_for_holdings,
)


router = APIRouter(prefix="/lm/holdings", tags=["holdings"])


def _round_holding_numbers(holdings: list[Dict[str, Any]]) -> None:
    """
    Normalize all per-holding numeric fields to 3 decimal places so the snapshot
    is readable and stable for GPT / dashboards.
    """
    numeric_fields_3 = [
        "shares",
        "cost_basis",
        "avg_cost",
        "last_price",
        "market_value",
        "unrealized_pnl",
        "unrealized_pct",
        "weight_pct",
        "forward_12m_dividend",
        "current_yield_pct",
        "yield_on_cost_pct",
    ]
    for h in holdings:
        for key in numeric_fields_3:
            val = h.get(key)
            if isinstance(val, (int, float)):
                h[key] = round(val, 3)


@router.get("/{plaid_account_id}")
def get_holdings_for_plaid_account(
    plaid_account_id: int,
    as_of: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Reconstruct holdings for a given Plaid-backed investment account
    using the stored LMTransaction rows.
    """
    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .all()
    )

    if not txs:
        raise HTTPException(
            status_code=404,
            detail=f"No transactions found for plaid_account_id={plaid_account_id}",
        )

    result = reconstruct_holdings(txs, plaid_account_id=plaid_account_id, as_of=as_of)

    # tack on which account this is for
    result["plaid_account_id"] = plaid_account_id
    return result


@router.get("/{plaid_account_id}/snapshot")
def get_valued_holdings_for_plaid_account(
    plaid_account_id: int,
    as_of: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Reconstruct holdings for a Plaid investment account and attach yfinance prices.
    """
    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .all()
    )

    if not txs:
        raise HTTPException(
            status_code=404,
            detail=f"No transactions found for plaid_account_id={plaid_account_id}",
        )

    # base holdings from LM transactions
    result = reconstruct_holdings(txs, plaid_account_id=plaid_account_id, as_of=as_of)

    holdings = result.get("holdings", [])
    symbols = [h["symbol"] for h in holdings]

    # fetch latest prices
    price_map = get_latest_prices(symbols)

    total_value = 0.0
    missing: list[str] = []
    total_cost_basis = 0.0

    for h in holdings:
        sym = h["symbol"]
        price = price_map.get(sym)
        if price is None:
            h["last_price"] = None
            h["market_value"] = None
            missing.append(sym)
            continue

        if price is not None:
            h["last_price"] = round(price, 3)
        mv = round(price * h["shares"], 3)
        h["market_value"] = round(mv, 3)
        total_value += mv

        cost_basis = float(h.get("cost_basis") or 0.0)
        total_cost_basis += cost_basis

        if mv is not None and cost_basis:
            unrealized_pnl = round(mv - cost_basis, 3)
            h["unrealized_pnl"] = round(unrealized_pnl, 3)
            if cost_basis > 0:
                unrealized_pct = round((unrealized_pnl / cost_basis) * 100.0, 3)
            else:
                unrealized_pct = None
            h["unrealized_pct"] = round(unrealized_pct, 3) if unrealized_pct is not None else None

    result["total_market_value"] = round(total_value, 3)
    result["prices_as_of"] = date.today().isoformat()
    result["missing_prices"] = missing
    result["plaid_account_id"] = plaid_account_id

    portfolio_value = float(result.get("total_market_value") or 0.0)
    for h in holdings:
        mv = h.get("market_value")
        if isinstance(mv, (int, float)):
            if portfolio_value > 0:
                h["weight_pct"] = round(mv / portfolio_value * 100.0, 3)
            else:
                h["weight_pct"] = None

    # --- margin loan info (from Lunch Money plaid accounts) ---
    margin_loan_balance: Optional[float] = None
    margin_to_portfolio_pct: Optional[float] = None

    try:
        lm_client = LunchMoneyClient()
        plaid_accounts = lm_client.get_plaid_accounts()

        # find the M1 Borrow / loan account
        loan_acct = None
        for acct in plaid_accounts:
            if (
                acct.get("type") == "loan"
                and acct.get("institution_name") == "M1 Finance"
                and acct.get("mask") == "0295"
            ):
                loan_acct = acct
                break

        if loan_acct:
            raw_balance = loan_acct.get("balance")
            margin_loan_balance = (
                float(raw_balance) if raw_balance is not None else None
            )

            # portfolio value from snapshot result; fallback to recompute if missing
            portfolio_value = float(result.get("total_market_value") or 0.0)
            if portfolio_value <= 0:
                portfolio_value = 0.0
                for h in holdings:
                    mv = h.get("market_value")
                    if isinstance(mv, (int, float)):
                        portfolio_value += mv

            if margin_loan_balance is not None and portfolio_value > 0:
                margin_to_portfolio_pct = round(
                    abs(margin_loan_balance) / portfolio_value * 100.0, 3
                )

    except Exception:
        # don't blow up snapshot if LM is flaky
        margin_loan_balance = None
        margin_to_portfolio_pct = None

    result["margin_loan_balance"] = margin_loan_balance
    result["margin_to_portfolio_pct"] = margin_to_portfolio_pct

    forward = estimate_forward_dividends_for_holdings(holdings)
    forward_by_symbol = forward.get("by_symbol", {})
    forward_total = float(forward.get("total_forward_12m") or 0.0)

    for h in holdings:
        symbol = h["symbol"]
        f_sym = forward_by_symbol.get(symbol) or {}
        f_div = float(f_sym.get("forward_12m_dividend") or 0.0)
        h["forward_12m_dividend"] = round(f_div, 3)

        mv = h.get("market_value") or 0.0
        cost_basis = float(h.get("cost_basis") or 0.0)

        if mv > 0:
            h["current_yield_pct"] = round((f_div / mv) * 100.0, 3)
        else:
            h["current_yield_pct"] = None

        if cost_basis > 0:
            h["yield_on_cost_pct"] = round((f_div / cost_basis) * 100.0, 3)
        else:
            h["yield_on_cost_pct"] = None

        # wrap with round if numeric
        if isinstance(h.get("forward_12m_dividend"), (int, float)):
            h["forward_12m_dividend"] = round(h["forward_12m_dividend"], 3)
        if isinstance(h.get("current_yield_pct"), (int, float)):
            h["current_yield_pct"] = round(h["current_yield_pct"], 3)
        if isinstance(h.get("yield_on_cost_pct"), (int, float)):
            h["yield_on_cost_pct"] = round(h["yield_on_cost_pct"], 3)

    # final pass: normalize numeric precision for all per-holding fields
    _round_holding_numbers(holdings)

    total_cost_basis = sum(float(h.get("cost_basis") or 0.0) for h in holdings)
    portfolio_value = float(result.get("total_market_value") or 0.0)

    result["totals"] = {
        "cost_basis": round(total_cost_basis, 3),
        "market_value": round(portfolio_value, 3),
        "unrealized_pnl": round(portfolio_value - total_cost_basis, 3),
        "unrealized_pct": round(
            ((portfolio_value - total_cost_basis) / total_cost_basis) * 100.0, 3
        )
        if total_cost_basis > 0
        else None,
        "margin_loan_balance": margin_loan_balance,
        "margin_to_portfolio_pct": margin_to_portfolio_pct,
    }

    result["income"] = {
        "forward_12m_total": round(forward_total, 3),
        "projected_monthly_income": round(forward_total / 12.0, 3) if forward_total else 0.0,
        "portfolio_current_yield_pct": round((forward_total / portfolio_value) * 100.0, 3)
        if portfolio_value > 0 and forward_total
        else None,
        "portfolio_yield_on_cost_pct": round((forward_total / total_cost_basis) * 100.0, 3)
        if total_cost_basis > 0 and forward_total
        else None,
    }

    # --- realized dividends + projected vs received (month-to-date) ---
    try:
        # reuse same LM transactions we already loaded
        events = extract_dividend_events(txs)

        # month-to-date window (from first of month through as_of)
        mtd_start = as_of.replace(day=1)
        realized_mtd = summarize_dividends(events, start=mtd_start, end=as_of)

        mtd_realized = float(realized_mtd.get("total_dividends") or 0.0)
        projected_monthly = float(result["income"].get("projected_monthly_income") or 0.0)

        projected_vs_received = {
            "window": {
                "label": "month_to_date",
                "start": mtd_start.isoformat(),
                "end": as_of.isoformat(),
            },
            "projected": round(projected_monthly, 3),
            "received": round(mtd_realized, 3),
            "difference": round(mtd_realized - projected_monthly, 3),
            "pct_of_projection": round(
                (mtd_realized / projected_monthly) * 100.0, 3
            )
            if projected_monthly > 0
            else None,
        }

        result["dividends"] = {
            "realized_mtd": realized_mtd,
            "projected_vs_received": projected_vs_received,
        }
    except Exception:
        # don't let dividend math break the snapshot
        result["dividends"] = {
            "realized_mtd": None,
            "projected_vs_received": None,
        }

    # --- persist snapshot for time-travel / history ---
    try:
        existing = (
            db.query(HoldingSnapshot)
            .filter(
                HoldingSnapshot.plaid_account_id == plaid_account_id,
                HoldingSnapshot.as_of_date == as_of,
            )
            .one_or_none()
        )

        snapshot_payload = dict(result)  # already plain JSON-ish types

        if existing:
            existing.snapshot = snapshot_payload
        else:
            snap = HoldingSnapshot(
                plaid_account_id=plaid_account_id,
                as_of_date=as_of,
                snapshot=snapshot_payload,
            )
            db.add(snap)

        db.commit()
    except Exception:
        # don't break the API just because history write failed
        db.rollback()

    return result



@router.get("/{plaid_account_id}/dividends")
def get_dividends_for_plaid_account(
    plaid_account_id: int,
    start: Optional[date] = Query(default=None),
    end: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Dividend view for a Plaid-backed investment account:

    - realized: cash dividends from LMTransaction (pay dates, actual amounts)
    - forward: simple 12m dividend estimate from current holdings + yfinance
    """
    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .all()
    )

    if not txs:
        raise HTTPException(
            status_code=404,
            detail=f"No transactions found for plaid_account_id={plaid_account_id}",
        )

    # 1) realized cashflow (LM transactions)
    events = extract_dividend_events(txs)
    realized = summarize_dividends(events, start=start, end=end)

    # 2) current holdings (same reconstruction logic as /lm/holdings)
    as_of = end or date.today()
    holdings_result = reconstruct_holdings(
        txs, plaid_account_id=plaid_account_id, as_of=as_of
    )
    holdings = holdings_result.get("holdings", [])

    # 3) forward 12-month estimate from yfinance profiles
    forward = estimate_forward_dividends_for_holdings(holdings)

    return {
        "plaid_account_id": plaid_account_id,
        "as_of": as_of.isoformat(),
        "realized": realized,
        "forward": forward,
    }