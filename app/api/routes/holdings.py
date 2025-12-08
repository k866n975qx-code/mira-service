from __future__ import annotations

from datetime import date, datetime, timedelta
from math import ceil, log
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.services.lm_client import LunchMoneyClient
from app.api.deps import get_db
from app.infra.models import DividendEvent, HoldingSnapshot, LMTransaction
from app.services.holdings import reconstruct_holdings
from app.services.pricing import get_latest_prices, _price_cache_ttl_seconds
from app.services.securities import resolve_symbol_from_cusip, should_overwrite_symbol
from app.services.dividends import (
    extract_dividend_events,
    summarize_dividends,
    estimate_forward_dividends_for_holdings,
    _yf_cache_ttl_seconds,
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


def _quarter_start(d: date) -> date:
    qm = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, qm, 1)


def _resolve_event_symbols(db: Session, events: list) -> None:
    """Mutates events in-place: if symbol is missing/stopword and we have CUSIP, resolve from CSV/DB."""
    for ev in events:
        sym = getattr(ev, "symbol", None)
        if should_overwrite_symbol(sym):
            cusip = getattr(ev, "cusip", None)
            if cusip:
                resolved = resolve_symbol_from_cusip(db, cusip)
                if resolved:
                    ev.symbol = resolved


def _summarize_events_from_db(
    db: Session, plaid_account_id: int, start: date, end: date
) -> Dict[str, Any]:
    """
    Summarize realized dividend cash events from the DB for a date window.
    Returns: { start, end, total_dividends, by_date, by_month, by_symbol }
    Also self-heals any bad symbols (e.g., "OF") using cusip CSV.
    """
    rows = (
        db.query(DividendEvent)
        .filter(DividendEvent.plaid_account_id == plaid_account_id)
        .filter(DividendEvent.pay_date >= start)
        .filter(DividendEvent.pay_date <= end)
        .order_by(DividendEvent.pay_date.asc(), DividendEvent.id.asc())
        .all()
    )

    total: float = 0.0
    by_date: Dict[str, float] = {}
    by_month: Dict[str, float] = {}
    by_symbol: Dict[str, float] = {}
    changed = False

    for row in rows:
        amt = float(getattr(row, "amount", 0.0) or 0.0)
        total += amt

        if getattr(row, "pay_date", None) is not None:
            ds = row.pay_date.isoformat()
            ms = f"{row.pay_date.year:04d}-{row.pay_date.month:02d}"
            by_date[ds] = round(by_date.get(ds, 0.0) + amt, 3)
            by_month[ms] = round(by_month.get(ms, 0.0) + amt, 3)

        sym = getattr(row, "symbol", None)
        if should_overwrite_symbol(sym):
            c = getattr(row, "cusip", None)
            if c:
                resolved = resolve_symbol_from_cusip(db, c)
                if resolved:
                    sym = resolved
                    try:
                        row.symbol = resolved
                        db.add(row)
                        changed = True
                    except Exception:
                        pass

        if sym:
            by_symbol[sym] = round(by_symbol.get(sym, 0.0) + amt, 3)

    if changed:
        try:
            db.commit()
        except Exception:
            db.rollback()

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_dividends": round(total, 3),
        "by_date": by_date,
        "by_month": by_month,
        "by_symbol": by_symbol,
    }


@router.get("/{plaid_account_id}/snapshot")
def get_valued_holdings_for_plaid_account(
    plaid_account_id: int,
    as_of: Optional[date] = Query(None),
    refresh: bool = Query(False),
    goal_monthly: Optional[float] = Query(None, description="Target monthly dividend income (defaults to 2000)"),
    symbols: Optional[str] = Query(None, description="Comma-separated tickers to include (e.g. JEPI,SVOL)"),
    apr_current_pct: Optional[float] = Query(None, description="Current promo APR (e.g., 4.4)"),
    apr_future_pct: Optional[float] = Query(None, description="Future APR after promo (e.g., 5.9)"),
    apr_future_date: Optional[date] = Query(None, description="When the future APR begins (e.g., 2026-11-01)"),
    margin_mode: Optional[str] = Query(None, description="preferred mode to highlight: conservative|balanced|aggressive"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Reconstruct holdings for a Plaid investment account and attach yfinance prices.
    """
    def _inject_goal_progress(payload: Dict[str, Any], as_of_date: date) -> None:
        """Compute goal_progress in-place using existing fields in payload."""
        target = float(goal_monthly or 2000.0)
        income = payload.get("income") or {}
        current = float(income.get("projected_monthly_income") or 0.0)
        shortfall = max(0.0, target - current)
        progress_pct = round((current / target * 100.0), 2) if target > 0 else None
        totals = payload.get("totals") or {}
        portfolio_value = float(totals.get("market_value") or 0.0)
        portfolio_yield_pct = income.get("portfolio_current_yield_pct")

        dividends = payload.get("dividends") or {}
        windows = dividends.get("windows") or {}
        ytd = windows.get("ytd") or {}
        by_month = (ytd.get("by_month") or {})

        asof_key = as_of_date.strftime("%Y-%m")
        month_keys = sorted([k for k in by_month.keys() if k < asof_key])
        last3_keys = month_keys[-3:] if len(month_keys) >= 3 else month_keys
        last_vals = [float(by_month[k]) for k in last3_keys]

        growth = None
        months_to_goal = None
        est_date = None
        try:
            if len(last_vals) >= 2 and last_vals[-2] > 0:
                g = (last_vals[-1] / last_vals[-2]) - 1.0
                g = min(max(g, -0.5), 0.5)
                growth = g
                if g > 0 and current > 0 and target > current:
                    months_to_goal = int(ceil(log(target / current) / log(1.0 + g)))
                    est_date = (as_of_date + timedelta(days=months_to_goal * 30)).isoformat()
        except Exception:
            pass

        payload["goal_progress"] = {
            "target_monthly": target,
            "current_projected_monthly": round(current, 2),
            "shortfall": round(shortfall, 2),
            "progress_pct": progress_pct,
            "portfolio_yield_pct": portfolio_yield_pct,
            "months_to_goal": months_to_goal,
            "estimated_goal_date": est_date,
            "assumptions": "based_on_last_3_full_months_realized",
            "growth_window_months": len(last3_keys),
            "required_portfolio_value_at_goal": (
                round((target * 12.0) / (float(portfolio_yield_pct) / 100.0), 2)
                if portfolio_yield_pct and float(portfolio_yield_pct) > 0.0
                else None
            ),
            "additional_investment_needed": (
                round(
                    max(
                        0.0,
                        ((target * 12.0) / (float(portfolio_yield_pct) / 100.0))
                        - portfolio_value,
                    ),
                    2,
                )
                if portfolio_yield_pct and float(portfolio_yield_pct) > 0.0
                else None
            ),
        }

    def _inject_margin_guidance(payload: Dict[str, Any], as_of_date: date) -> None:
        """
        Compute risk-aware margin guidance for three modes and add to payload.margin_guidance.
        Uses only fields already present in the snapshot.
        """
        totals = payload.get("totals") or {}
        income = payload.get("income") or {}
        portfolio_value = float(totals.get("market_value") or 0.0)
        current_margin = float(payload.get("margin_loan_balance") or 0.0)
        current_margin = abs(current_margin)
        monthly_income = float(income.get("projected_monthly_income") or 0.0)

        cur_apr = float(apr_current_pct if apr_current_pct is not None else 4.4)
        fut_apr = float(apr_future_pct if apr_future_pct is not None else 5.9)
        fut_date = apr_future_date or date(2026, 11, 1)

        presets = {
            "conservative": {"stress_drawdown_pct": 40.0, "min_income_coverage": 2.0, "max_margin_pct": 20.0},
            "balanced": {"stress_drawdown_pct": 30.0, "min_income_coverage": 1.5, "max_margin_pct": 25.0},
            "aggressive": {"stress_drawdown_pct": 20.0, "min_income_coverage": 1.2, "max_margin_pct": 30.0},
        }
        modes = ["conservative", "balanced", "aggressive"]
        preferred = (margin_mode or "balanced").lower()
        if preferred not in presets:
            preferred = "balanced"

        def _calc_for(mode_name: str) -> Dict[str, Any]:
            p = presets[mode_name]
            stress = p["stress_drawdown_pct"] / 100.0
            max_ltv = p["max_margin_pct"] / 100.0
            cov = p["min_income_coverage"]
            r_now = cur_apr / 100.0 / 12.0
            r_fut = fut_apr / 100.0 / 12.0

            ltv_now = (current_margin / portfolio_value * 100.0) if portfolio_value > 0 else None
            ltv_stress = (
                (current_margin / (portfolio_value * (1.0 - stress)) * 100.0)
                if portfolio_value > 0 and (1.0 - stress) > 0
                else None
            )

            cap1 = max_ltv * portfolio_value - current_margin
            cap2 = max_ltv * (portfolio_value * (1.0 - stress)) - current_margin
            cap3 = (monthly_income / (cov * r_now) - current_margin) if r_now > 0 else float("inf")

            caps = [cap for cap in (cap1, cap2, cap3) if cap is not None]
            borrow_capacity = max(0.0, min(caps)) if caps else 0.0
            repay_needed = max(0.0, max([-c for c in caps if c < 0.0], default=0.0))

            if repay_needed > 0.0:
                action = "repay"
                amount = round(repay_needed, 2)
            elif borrow_capacity > 0.0:
                action = "borrow_up_to"
                amount = round(borrow_capacity, 2)
            else:
                action = "maintain"
                amount = 0.0

            i_now = round(r_now * current_margin, 2) if r_now > 0 else 0.0
            cov_now = round((monthly_income / i_now), 3) if i_now > 0 else None
            i_future = round(r_fut * current_margin, 2) if r_fut > 0 else 0.0
            cov_future = round((monthly_income / i_future), 3) if i_future > 0 else None

            return {
                "mode": mode_name,
                "action": action,
                "amount": amount,
                "ltv_now_pct": round(ltv_now, 3) if ltv_now is not None else None,
                "ltv_stress_pct": round(ltv_stress, 3) if ltv_stress is not None else None,
                "monthly_interest_now": i_now,
                "income_interest_coverage_now": cov_now,
                "monthly_interest_future": i_future,
                "income_interest_coverage_future": cov_future,
                "constraints": {
                    "max_margin_pct": p["max_margin_pct"],
                    "stress_drawdown_pct": p["stress_drawdown_pct"],
                    "min_income_coverage": p["min_income_coverage"],
                },
            }

        payload["margin_guidance"] = {
            "selected_mode": preferred,
            "rates": {
                "apr_current_pct": cur_apr,
                "apr_future_pct": fut_apr,
                "apr_future_date": fut_date.isoformat(),
            },
            "modes": [_calc_for(m) for m in modes],
        }

    def _get_last_transaction_sync_at() -> Optional[datetime]:
        try:
            return (
                db.query(func.max(LMTransaction.created_at))
                .filter(LMTransaction.plaid_account_id == plaid_account_id)
                .scalar()
            )
        except Exception:
            return None

    last_transaction_sync_at = _get_last_transaction_sync_at()

    def _build_meta(served_from: str, snapshot_created_at: datetime) -> Dict[str, Any]:
        now = datetime.utcnow()
        age_days: Optional[int] = None
        if snapshot_created_at:
            try:
                age_days = max(0, (now.date() - snapshot_created_at.date()).days)
            except Exception:
                age_days = None
        return {
            "served_from": served_from,
            "snapshot_created_at": snapshot_created_at.isoformat() if snapshot_created_at else None,
            "snapshot_age_days": age_days,
            "last_transaction_sync_at": (
                last_transaction_sync_at.isoformat() if last_transaction_sync_at else None
            ),
            "cache": {
                "yf_dividends": {
                    "ttl_seconds": _yf_cache_ttl_seconds(),
                    "bypassed": bool(refresh),
                },
                "pricing": {
                    "ttl_seconds": _price_cache_ttl_seconds(),
                    "bypassed": bool(refresh),
                },
            },
        }

    # --- Fast path: return latest cached snapshot unless refresh requested ---
    if not refresh:
        if as_of is not None:
            existing = (
                db.query(HoldingSnapshot)
                .filter(
                    HoldingSnapshot.plaid_account_id == plaid_account_id,
                    HoldingSnapshot.as_of_date == as_of,
                )
                .one_or_none()
            )
        else:
            existing = (
                db.query(HoldingSnapshot)
                .filter(HoldingSnapshot.plaid_account_id == plaid_account_id)
                .order_by(HoldingSnapshot.as_of_date.desc())
                .first()
            )
        if existing and existing.snapshot:
            payload = dict(existing.snapshot)
            payload.setdefault("plaid_account_id", plaid_account_id)
            payload.setdefault("as_of", existing.as_of_date.isoformat())
            payload["cached"] = True
            _inject_goal_progress(payload, existing.as_of_date)
            _inject_margin_guidance(payload, existing.as_of_date)
            created_at = getattr(existing, "created_at", None) or datetime.utcnow()
            payload["meta"] = _build_meta("db", created_at)
            return payload

    as_of = as_of or date.today()

    # Validate that this plaid_account_id is an M1 investment account
    try:
        lm_client = LunchMoneyClient()
        plaid_accounts = lm_client.get_plaid_accounts()
        allowed = False
        for acct in (plaid_accounts or []):
            try:
                pid = int(
                    acct.get("id")
                    or acct.get("plaid_account_id")
                    or (acct.get("plaid_account") or {}).get("id")
                )
            except Exception:
                continue
            if (
                pid == plaid_account_id
                and acct.get("institution_name") == "M1 Finance"
                and (acct.get("type") or "").lower() == "investment"
            ):
                allowed = True
                break
        if not allowed:
            raise HTTPException(
                status_code=404,
                detail=f"plaid_account_id={plaid_account_id} is not an M1 investment account",
            )
    except Exception:
        # If LM is unreachable, proceed without the check (best-effort), holdings logic will still work from DB
        pass
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
    symbols_set = None
    if symbols:
        try:
            symbols_set = {s.strip().upper() for s in symbols.split(",") if s.strip()}
        except Exception:
            symbols_set = None
    if symbols_set:
        holdings = [h for h in holdings if (h.get("symbol") or "").upper() in symbols_set]
    # propagate filtered holdings back onto the response
    result["holdings"] = holdings
    # capture provided symbols even if nothing matched
    if symbols is not None:
        result["filters"] = {"symbols": sorted(list(symbols_set or []))}

    # unique list of symbols for pricing/history
    symbols_list = sorted({h["symbol"] for h in holdings if h.get("symbol")})

    # fetch latest prices
    price_map = get_latest_prices(symbols_list, bypass_cache=refresh)

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

    forward = estimate_forward_dividends_for_holdings(holdings, bypass_cache=refresh)
    forward_by_symbol = forward.get("by_symbol", {})
    forward_total = float(forward.get("total_forward_12m") or 0.0)

    for h in holdings:
        symbol = h["symbol"]
        f_sym = forward_by_symbol.get(symbol) or {}
        f_div = float(f_sym.get("forward_12m_dividend") or 0.0)
        h["forward_12m_dividend"] = round(f_div, 3)
        h["last_ex_date"] = f_sym.get("last_ex_date")
        h["forward_method"] = f_sym.get("forward_method")
        h["projected_monthly_dividend"] = f_sym.get("projected_monthly_dividend")

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

    # Defensive: ensure no 'trend' key lingers on holdings
    for h in holdings:
        h.pop("trend", None)

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
        # Compute realized dividends from persisted DB events (faster & symbol-correct)
        mtd_start = as_of.replace(day=1)
        realized_mtd = _summarize_events_from_db(db, plaid_account_id, mtd_start, as_of)
        mtd_realized = float(realized_mtd.get("total_dividends") or 0.0)

        projected_monthly = round(forward_total / 12.0, 3) if forward_total else 0.0

        projected_vs_received = {
            "window": {
                "label": "month_to_date",
                "start": mtd_start.isoformat(),
                "end": as_of.isoformat(),
            },
            "projected": projected_monthly,
            "received": round(mtd_realized, 3),
            "difference": round(projected_monthly - mtd_realized, 3),
            "pct_of_projection": round(
                (mtd_realized / projected_monthly) * 100.0, 2
            )
            if projected_monthly > 0
            else None,
        }

        # Additional windows from DB events: 30d / QTD / YTD
        start_30d = as_of - timedelta(days=30)
        sum_30d = _summarize_events_from_db(db, plaid_account_id, start_30d, as_of)
        start_qtd = _quarter_start(as_of)
        sum_qtd = _summarize_events_from_db(db, plaid_account_id, start_qtd, as_of)
        start_ytd = date(as_of.year, 1, 1)
        sum_ytd = _summarize_events_from_db(db, plaid_account_id, start_ytd, as_of)

        # Attach per-holding fields sourced from DB windows
        by30 = sum_30d.get("by_symbol", {})
        byq = sum_qtd.get("by_symbol", {})
        byy = sum_ytd.get("by_symbol", {})
        for h in holdings:
            s = h.get("symbol")
            if s:
                h["dividends_30d"] = round(float(by30.get(s, 0.0)), 3)
                h["dividends_qtd"] = round(float(byq.get(s, 0.0)), 3)
                h["dividends_ytd"] = round(float(byy.get(s, 0.0)), 3)

        result["dividends"] = {
            "realized_mtd": realized_mtd,
            "projected_vs_received": projected_vs_received,
            "windows": {
                "30d": sum_30d,
                "qtd": sum_qtd,
                "ytd": sum_ytd,
            },
        }
    except Exception:
        # don't let dividend math break the snapshot
        result["dividends"] = {
            "realized_mtd": None,
            "projected_vs_received": None,
        }

    snapshot_created_at = datetime.utcnow()
    result["meta"] = _build_meta("recomputed", snapshot_created_at)

    # --- persist snapshot for time-travel / history ---
    try:
        # Ensure we have an as_of for persistence (already set above)
        existing = (
            db.query(HoldingSnapshot)
            .filter(
                HoldingSnapshot.plaid_account_id == plaid_account_id,
                HoldingSnapshot.as_of_date == as_of,
            )
            .one_or_none()
        )

        # Attach goal progress & margin guidance before persisting and returning
        _inject_goal_progress(result, as_of)
        _inject_margin_guidance(result, as_of)
        # Ensure 'trend' never persists
        for h in result.get("holdings", []):
            h.pop("trend", None)

        snapshot_payload = dict(result)  # already plain JSON-ish types
        snapshot_payload["cached"] = False

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

    result["cached"] = False

    return result
