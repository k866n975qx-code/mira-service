from __future__ import annotations

import copy
import json
import os
import gzip
import time
import importlib
from datetime import date, datetime, timedelta, timezone
from math import ceil, log
from typing import Any, Dict, Optional, List
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.services.lm_client import LunchMoneyClient
from app.services.snapshot_normalizer import (
    build_and_cache_snapshot,
    cache_snapshot,
    compute_cache_key,
    load_snapshot,
    normalize_snapshot,
    validate_snapshot,
)
from app.api.deps import get_db
from app.infra.models import DividendEvent, HoldingSnapshot, LMTransaction
from app.services.holdings import reconstruct_holdings
from app.services.pricing import (
    get_latest_prices,
    _price_cache_ttl_seconds,
    get_prices_as_of,
)
from app.services.securities import resolve_symbol_from_cusip, should_overwrite_symbol
from app.services.enrich import enrich_holding
from app.services.portfolio import compute_portfolio_rollups
from app.services.enrich_fallback import ensure_minimal_ultimate
from app.services.dividends_projector import project_paydate_window, project_upcoming_exdates
from app.services.dividends import (
    extract_dividend_events,
    summarize_dividends,
    estimate_forward_dividends_for_holdings,
    _yf_cache_ttl_seconds,
)
from app.services.macro import get_macro_package


# Cache snapshots for 6 hours to reduce rebuild frequency while keeping data reasonably fresh.
SNAPSHOT_CACHE_TTL_SECONDS = 6 * 3600
# Minimum interval between expensive recomputes when refresh=true (seconds)
MIN_REFRESH_SECONDS = int(os.getenv("MIRA_SNAPSHOT_MIN_REFRESH_SECONDS", "300"))

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


def _discover_m1_investment_account_ids(client: LunchMoneyClient) -> List[int]:
    """List plaid_account_ids for M1 investment accounts from Lunch Money."""
    accounts = client.get_plaid_accounts() or []
    ids: List[int] = []
    for acc in accounts:
        inst = (acc.get("institution_name") or "").strip()
        typ = (acc.get("type") or acc.get("account_type") or "").strip().lower()
        name = (acc.get("name") or "").strip().lower()
        if inst == "M1 Finance" and (typ == "investment" or "invest" in name):
            pid = acc.get("id") or acc.get("plaid_account_id") or (acc.get("plaid_account") or {}).get("id")
            try:
                ids.append(int(pid))
            except Exception:
                continue
    return sorted(list({i for i in ids}))


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
    enrich: Optional[bool] = Query(True, description="Attach data-rich metrics under holding.ultimate"),
    perf: Optional[bool] = Query(True, description="Attach portfolio-level performance and risk rollups"),
    perf_method: Optional[str] = Query("accurate", description="Performance method: accurate|approx (default accurate)"),
    slim: bool = Query(
        True,
        description="Return a compact snapshot (drop provenance, condense holdings metadata, round decimals).",
    ),
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
        if months_to_goal is None:
            # Conservative fallback if we cannot infer growth from history.
            if target <= current:
                months_to_goal = 0
                est_date = as_of_date.isoformat()
            else:
                assumed_growth = 0.0025  # ~3% annualized growth
                if current > 0 and assumed_growth > 0:
                    months_to_goal = int(ceil(log(target / current) / log(1.0 + assumed_growth)))
                else:
                    months_to_goal = 240  # 20-year placeholder
                est_date = (as_of_date + timedelta(days=months_to_goal * 30)).isoformat()
        if months_to_goal is not None and months_to_goal > 480:
            months_to_goal = None
            est_date = None

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
            "growth_window_months": max(3, len(last3_keys)) if last3_keys else 3,
        }

        # Net of margin interest variant
        portfolio_yield_pct = income.get("portfolio_current_yield_pct")
        margin_balance = float(payload.get("margin_loan_balance") or 0.0)
        margin_balance = abs(margin_balance)
        cur_apr_val = float(apr_current_pct if apr_current_pct is not None else 4.15)
        fut_apr_val = float(apr_future_pct if apr_future_pct is not None else 5.65)
        cur_interest = round(margin_balance * (cur_apr_val / 100.0) / 12.0, 2)
        fut_interest = round(margin_balance * (fut_apr_val / 100.0) / 12.0, 2)

        def _req_value(interest: float, rate_pct: Optional[float]) -> Optional[float]:
            if rate_pct is None or rate_pct <= 0:
                return None
            return ((target + interest) * 12.0) / (rate_pct / 100.0)

        req_now = _req_value(cur_interest, portfolio_yield_pct)
        req_future = _req_value(fut_interest, portfolio_yield_pct)
        addl_now = (
            max(0.0, (req_now - portfolio_value)) if req_now is not None else None
        )
        addl_future = (
            max(0.0, (req_future - portfolio_value)) if req_future is not None else None
        )
        current_net = max(0.0, current - cur_interest)
        progress_net = (
            round((current_net / target * 100.0), 2) if target > 0 else None
        )
        progress_net_future = (
            round(((max(0.0, current - fut_interest)) / target * 100.0), 2)
            if target > 0
            else None
        )

        payload["goal_progress_net"] = {
            "target_monthly": target,
            "current_projected_monthly_net": round(current_net, 3),
            "progress_pct": progress_net,
            "portfolio_yield_pct": portfolio_yield_pct,
            "assumptions": "same_yield_structure; loan unchanged",
            "required_portfolio_value_at_goal_now": round(req_now, 2)
            if req_now is not None
            else None,
            "additional_investment_needed_now": round(addl_now, 2)
            if addl_now is not None
            else None,
            "monthly_interest_now": cur_interest,
            "future_rate_sensitivity": {
                "apr_future_pct": fut_apr_val,
                "monthly_interest_future": fut_interest,
                "required_portfolio_value_at_goal_future": round(req_future, 2)
                if req_future is not None
                else None,
                "additional_investment_needed_future": round(addl_future, 2)
                if addl_future is not None
                else None,
                "progress_pct_future": progress_net_future,
            },
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

        cur_apr = float(apr_current_pct if apr_current_pct is not None else 4.15)
        fut_apr = float(apr_future_pct if apr_future_pct is not None else 5.65)
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
                new_loan = max(0.0, current_margin - amount)
            elif borrow_capacity > 0.0:
                action = "borrow_up_to"
                amount = round(borrow_capacity, 2)
                new_loan = current_margin + amount
            else:
                action = "maintain"
                amount = 0.0
                new_loan = current_margin

            i_now = round(r_now * current_margin, 2) if r_now > 0 else 0.0
            cov_now = round((monthly_income / i_now), 3) if i_now > 0 else None
            i_future = round(r_fut * current_margin, 2) if r_fut > 0 else 0.0
            cov_future = round((monthly_income / i_future), 3) if i_future > 0 else None

            ltv_after = (
                (new_loan / portfolio_value * 100.0) if portfolio_value > 0 else None
            )
            i_now_after = round(r_now * new_loan, 2) if r_now > 0 else 0.0
            cov_now_after = (
                round((monthly_income / i_now_after), 3) if i_now_after > 0 else None
            )

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
                "after_action": {
                    "new_loan_balance": round(new_loan, 2),
                    "ltv_after_action_pct": round(ltv_after, 3) if ltv_after is not None else None,
                    "monthly_interest_now": i_now_after,
                    "income_interest_coverage_now": cov_now_after,
                },
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
        now = datetime.now(timezone.utc)
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
            # Allow snapshots to be cached/stored; honor normal TTLs unless refresh is requested
            "cache_control": {"no_store": False, "revalidate": "when-stale"},
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

    def _round_numbers(obj: Any) -> Any:
        """Recursively round floats/Decimals to 3 decimals to shrink payloads."""
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, float):
            return round(obj, 3)
        if isinstance(obj, Decimal):
            return round(float(obj), 3)
        if isinstance(obj, list):
            return [_round_numbers(v) for v in obj]
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = _round_numbers(v)
            return obj
        return obj

    def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a compact copy of the snapshot:
        - Drop heavy provenance
        - Round numeric noise
        """
        data = copy.deepcopy(snapshot)
        holdings = data.get("holdings") or []
        for h in holdings:
            h.pop("ultimate_provenance", None)
        macro_block = data.get("macro")
        if isinstance(macro_block, dict):
            macro_block.pop("provenance", None)
            snap = macro_block.get("snapshot")
            if isinstance(snap, dict):
                snap.pop("meta", None)
            hist = macro_block.get("history")
            if isinstance(hist, dict):
                hist.pop("meta", None)
            trends = macro_block.get("trends")
            if isinstance(trends, dict):
                trends.pop("meta", None)
        data = _round_numbers(data)
        return data

    def _fill_gaps(new_val: Any, existing_val: Any) -> Any:
        """Merge where None/empty in new uses existing; otherwise keep new."""
        if new_val is None:
            return existing_val
        if isinstance(new_val, dict) and isinstance(existing_val, dict):
            merged = {}
            all_keys = set(new_val.keys()) | set(existing_val.keys())
            for k in all_keys:
                merged[k] = _fill_gaps(new_val.get(k), existing_val.get(k))
            return merged
        if isinstance(new_val, list) and isinstance(existing_val, list):
            return new_val if len(new_val) > 0 else existing_val
        return new_val

    def _validate_snapshot(snap: Dict[str, Any], allow_partial: bool = False) -> None:
        missing_fields = []
        def _chk(path: str) -> Any:
            cur = snap
            for p in path.split("."):
                if not isinstance(cur, dict) or p not in cur:
                    missing_fields.append(path)
                    return None
                cur = cur[p]
            return cur

        mv = _chk("totals.market_value")
        cb = _chk("totals.cost_basis")
        inc = _chk("income.projected_monthly_income")
        vol = _chk("portfolio_rollups.risk.vol_30d_pct")
        holdings = snap.get("holdings")
        if not holdings or not isinstance(holdings, list):
            missing_fields.append("holdings")
        else:
            for h in holdings:
                if not h.get("symbol"):
                    missing_fields.append("holdings.symbol")
                    break
                if not allow_partial and h.get("market_value") is None:
                    missing_fields.append("holdings.market_value")
                    break
        if missing_fields and not allow_partial:
            raise HTTPException(status_code=500, detail=f"Snapshot validation failed; missing: {sorted(set(missing_fields))}")

    def _maybe_slim(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return _slim_snapshot(snapshot) if slim else snapshot

    def _attach_macro_block(payload: Dict[str, Any], refresh_flag: bool = False) -> None:
        """Best-effort macro attachment; non-blocking; uses as-of date when possible."""
        try:
            macro = get_macro_package(force_refresh=bool(refresh_flag), as_of=as_of)
            snap = macro.get("snapshot") if isinstance(macro, dict) else None
            if isinstance(snap, dict) and snap.get("date"):
                prov = None
                if isinstance(snap.get("meta"), dict):
                    sm = snap["meta"]
                    prov = {
                        "source": sm.get("source"),
                        "fetched_at": sm.get("fetched_at"),
                        "schema_version": sm.get("schema_version"),
                    }
                payload["macro"] = {
                    "snapshot": snap,
                    "trends": macro.get("trends"),
                    "history": macro.get("history"),
                    "provenance": prov,
                }
        except Exception:
            pass

    def _persist_snapshot_file(payload: Dict[str, Any], as_of_date: date, pid: int) -> None:
        """Write snapshot to disk (json + gzip) for comparisons; best-effort."""
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "portfolio", "snapshots"))
            os.makedirs(base_dir, exist_ok=True)
            fname = f"snapshot-{pid}-{as_of_date.isoformat()}.json"
            path = os.path.join(base_dir, fname)
            # ensure only one per day: replace any existing file for that date
            for ext in ("", ".gz"):
                try:
                    os.remove(path + ext)
                except OSError:
                    pass
            # write minified json
            with open(path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            gz_path = f"{path}.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
                json.dump(payload, gz, separators=(",", ":"))

            # Also mirror to data/snapshots/daily for weekly generator inputs.
            daily_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "snapshots", "daily"))
            os.makedirs(daily_dir, exist_ok=True)
            daily_path = os.path.join(daily_dir, f"{as_of_date.isoformat()}.json")
            existed_daily = os.path.exists(daily_path)
            with open(daily_path, "w") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            # Only trigger weekly generation when a brand-new daily snapshot was created.
            if not existed_daily:
                try:
                    wfg = importlib.import_module("scripts.weekly_fusion_generator")
                    daily_files = [f for f in os.listdir(daily_dir) if f.endswith(".json")]
                    dates: list[date] = []
                    for fname in daily_files:
                        try:
                            dates.append(date.fromisoformat(fname.replace(".json", "")))
                        except Exception:
                            continue
                    dates = sorted(set(dates))

                    # Determine last summarized weekly end_date, if any.
                    weekly_dir = os.path.abspath(os.path.join(os.path.dirname(daily_dir), "..", "summaries", "weekly"))
                    last_weekly_end: Optional[date] = None
                    if os.path.isdir(weekly_dir):
                        for wfile in os.listdir(weekly_dir):
                            if not wfile.endswith(".json"):
                                continue
                            base = wfile.replace(".json", "")
                            if base.startswith("weekly_"):
                                try:
                                    end_d = date.fromisoformat(base.replace("weekly_", "").replace("_", "-"))
                                    if last_weekly_end is None or end_d > last_weekly_end:
                                        last_weekly_end = end_d
                                except Exception:
                                    continue

                    # Require 7 unsummarized consecutive days beyond the last weekly end before generating a new weekly.
                    new_days = [d for d in dates if (last_weekly_end is None or d > last_weekly_end)]
                    if len(new_days) >= 7:
                        candidate_block = sorted(new_days)[-7:]
                        is_consecutive = all(
                            (candidate_block[i] - candidate_block[i - 1]).days == 1 for i in range(1, len(candidate_block))
                        )
                        candidate_end = candidate_block[-1]
                        weekly_fname = f"weekly_{candidate_end.strftime('%Y_%m_%d')}.json"
                        weekly_path = os.path.join(weekly_dir, weekly_fname)
                        if is_consecutive and not os.path.exists(weekly_path):
                            wfg.generate_and_write(use_cache=False)
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_short_circuit_refresh(
        plaid_account_id: int, as_of_date: date, refresh_flag: bool
    ) -> Optional[Dict[str, Any]]:
        """
        If refresh is requested but we already have a recent snapshot for this date,
        short-circuit to avoid long recompute and potential timeouts.
        """
        if not refresh_flag:
            return None
        try:
            existing = (
                db.query(HoldingSnapshot)
                .filter(
                    HoldingSnapshot.plaid_account_id == plaid_account_id,
                    HoldingSnapshot.as_of_date == as_of_date,
                )
                .one_or_none()
            )
            if existing and existing.snapshot:
                snap = dict(existing.snapshot)
                meta = snap.get("meta") or {}
                created_at_str = meta.get("snapshot_created_at")
                if created_at_str:
                    try:
                        created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                        if age < MIN_REFRESH_SECONDS:
                            snap["cached"] = True
                            return snap
                    except Exception:
                        pass
                # otherwise allow rebuild
        except Exception:
            return None
        return None

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
            as_of_cached = existing.as_of_date

            # Ensure per-holding minimal enrichment for approx calculations
            try:
                ensure_minimal_ultimate(payload.get("holdings", []), as_of_cached)
            except Exception:
                pass

            # Repair portfolio_rollups if missing/empty
            if perf:
                pr = payload.get("portfolio_rollups") or {}
                perf_block = pr.get("performance") or {}
                risk_block = pr.get("risk") or {}
                needs_rollups = (not perf_block) or (not risk_block) or ("composition" not in pr)
                if needs_rollups and compute_portfolio_rollups:
                    recomputed = None
                    try:
                        recomputed = compute_portfolio_rollups(
                            plaid_account_id,
                            as_of_cached,
                            payload.get("holdings", []),
                            perf_method=perf_method or "accurate",
                            include_mwr=True,
                        )
                    except Exception:
                        pass
                    if recomputed is None:
                        try:
                            recomputed = compute_portfolio_rollups(
                                plaid_account_id,
                                as_of_cached,
                                payload.get("holdings", []),
                                perf_method="approx",
                                include_mwr=False,
                            )
                        except Exception:
                            recomputed = None
                    if recomputed is not None:
                        recomputed.setdefault("meta", {}).setdefault(
                            "note", "Restored after cached omission."
                        )
                        payload["portfolio_rollups"] = recomputed
                # final guard: ensure block exists even if recompute failed
                if not payload.get("portfolio_rollups") or not payload["portfolio_rollups"].get("performance"):
                    payload["portfolio_rollups"] = payload.get("portfolio_rollups") or {}
                    payload["portfolio_rollups"].setdefault("performance", {})
                    payload["portfolio_rollups"].setdefault("risk", {})
                    payload["portfolio_rollups"].setdefault("benchmark", "^GSPC")
                    payload["portfolio_rollups"].setdefault(
                        "meta",
                        {"method": "approx-fallback", "note": "Restored after cached omission."},
                    )

            # Repair dividends block if missing/null
            divs = payload.get("dividends")
            if not divs or not divs.get("realized_mtd") or divs.get("projected_vs_received") is None:
                try:
                    forward_total_cached = float(
                        (payload.get("income") or {}).get("forward_12m_total") or 0.0
                    )
                    mtd_start = as_of_cached.replace(day=1)
                    realized_mtd_raw = _summarize_events_from_db(
                        db, plaid_account_id, mtd_start, as_of_cached
                    )
                    mtd_realized = float(realized_mtd_raw.get("total_dividends") or 0.0)
                    projected_monthly = round(forward_total_cached / 12.0, 3) if forward_total_cached else 0.0
                    projected_vs_received = {
                        "window": {
                            "label": "month_to_date",
                            "start": mtd_start.isoformat(),
                            "end": as_of_cached.isoformat(),
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
                    try:
                        paydate_proj = project_paydate_window(payload.get("holdings", []), mtd_start, as_of_cached)
                        if paydate_proj is not None:
                            projected_vs_received["alt"] = paydate_proj
                    except Exception:
                        pass

                    # Additional windows
                    start_30d = as_of_cached - timedelta(days=30)
                    sum_30d = _summarize_events_from_db(db, plaid_account_id, start_30d, as_of_cached)
                    start_qtd = _quarter_start(as_of_cached)
                    sum_qtd = _summarize_events_from_db(db, plaid_account_id, start_qtd, as_of_cached)
                    start_ytd = date(as_of_cached.year, 1, 1)
                    sum_ytd = _summarize_events_from_db(db, plaid_account_id, start_ytd, as_of_cached)

                    def _tag_status(window: Dict[str, Any]) -> Dict[str, Any]:
                        by_sym = window.get("by_symbol") or {}
                        tagged: Dict[str, Any] = {}
                        active_syms = {h.get("symbol") for h in payload.get("holdings", []) if h.get("symbol")}
                        for sym, amt in by_sym.items():
                            try:
                                amt_val = float(amt)
                            except Exception:
                                amt_val = amt
                            tagged[sym] = {
                                "amount": amt_val,
                                "status": "active" if sym in active_syms else "inactive",
                            }
                        window["by_symbol"] = tagged
                        return window

                    sum_30d = _tag_status(sum_30d)
                    sum_qtd = _tag_status(sum_qtd)
                    sum_ytd = _tag_status(sum_ytd)
                    sum_mtd_tagged = _tag_status(realized_mtd_raw)
                    realized_mtd_raw["by_symbol"] = sum_mtd_tagged.get("by_symbol", {})

                    payload["dividends"] = {
                        "realized_mtd": realized_mtd_raw,
                        "realized_mtd_detail": sum_mtd_tagged.get("by_symbol"),
                        "projected_vs_received": projected_vs_received,
                        "windows": {
                            "30d": sum_30d,
                            "qtd": sum_qtd,
                            "ytd": sum_ytd,
                        },
                    }
                except Exception:
                    pass

            _inject_goal_progress(payload, as_of_cached)
            _inject_margin_guidance(payload, as_of_cached)
            created_at = getattr(existing, "created_at", None) or datetime.now(timezone.utc)
            payload["meta"] = _build_meta("db", created_at)

            cache_key = compute_cache_key(payload)
            cached_snapshot = load_snapshot(cache_key)
            if cached_snapshot:
                _attach_macro_block(cached_snapshot, refresh_flag=False)
                try:
                    _persist_snapshot_file(cached_snapshot, as_of_cached, plaid_account_id)
                except Exception:
                    pass
                return _maybe_slim(cached_snapshot)

            normalized_payload = normalize_snapshot(payload)
            validate_snapshot(normalized_payload, raise_on_error=True)
            normalized_payload["cached"] = True
            _attach_macro_block(normalized_payload, refresh_flag=False)
            cache_snapshot(cache_key, normalized_payload, ttl=SNAPSHOT_CACHE_TTL_SECONDS)
            try:
                _persist_snapshot_file(normalized_payload, as_of_cached, plaid_account_id)
            except Exception:
                pass
            return _maybe_slim(normalized_payload)
    else:
        # refresh requested: short-circuit if a fresh snapshot already exists for this date
        short = _maybe_short_circuit_refresh(plaid_account_id, as_of or date.today(), refresh_flag=True)
        if short:
            _attach_macro_block(short, refresh_flag=False)
            return _maybe_slim(short)

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
    # Only include transactions up to the as_of date so backfills/time-travel are correct.
    txs = (
        db.query(LMTransaction)
        .filter(LMTransaction.plaid_account_id == plaid_account_id)
        .filter(LMTransaction.date <= as_of)
        .order_by(LMTransaction.date.asc())
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

    # fetch prices (historical if as_of in the past) with retry safeguards
    today = date.today()
    price_attempts = 0
    price_map: Dict[str, float] = {}
    while price_attempts < 3:
        if as_of < today:
            price_map = get_prices_as_of(symbols_list, as_of_date=as_of)
        else:
            price_map = get_latest_prices(symbols_list, bypass_cache=refresh)
        if symbols_list and not price_map:
            price_attempts += 1
            time.sleep(2)
            continue
        break
    price_partial = False
    if symbols_list:
        missing_syms = set(symbols_list) - set(price_map.keys())
        price_partial = len(missing_syms) > 0

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
    # Only include margin balance for "today"; for historical dates we skip to avoid leaking current balance.
    margin_loan_balance: Optional[float] = None
    margin_to_portfolio_pct: Optional[float] = None

    if as_of == date.today():
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

    # --- enrichment (data-rich metrics per holding) ---
    try:
        if enrich:
            for h in holdings:
                sym = h.get("symbol")
                if not sym:
                    continue
                last_price = float(h.get("last_price") or 0.0)
                last_ex_date = h.get("last_ex_date")
                try:
                    ultimate = enrich_holding(sym, last_price, last_ex_date, as_of)
                    if ultimate:
                        h["ultimate"] = ultimate
                except Exception:
                    # do not break snapshot if enrichment fails for one symbol
                    pass
    except Exception:
        pass

    # Minimal fallback enrichment so approx TWR can always run
    try:
        ensure_minimal_ultimate(holdings, as_of)
    except Exception:
        pass

    # --- portfolio rollups (performance, MWR, risk) ---
    if perf:
        rollups = None
        try:
            rollups = compute_portfolio_rollups(
                plaid_account_id,
                as_of,
                holdings,
                perf_method=perf_method or "accurate",
                include_mwr=True,
            )
        except Exception:
            rollups = None
        if rollups is None:
            try:
                rollups = compute_portfolio_rollups(
                    plaid_account_id,
                    as_of,
                    holdings,
                    perf_method="approx",
                    include_mwr=False,
                )
                rollups.setdefault("meta", {}).setdefault(
                    "note", "Restored after temporary omission."
                )
            except Exception:
                rollups = {
                    "performance": {},
                    "risk": {},
                    "benchmark": "^GSPC",
                    "meta": {
                        "method": "approx-fallback",
                        "note": "Restored after temporary omission.",
                    },
                }
        if not rollups.get("performance"):
            rollups.setdefault("meta", {}).setdefault(
                "note", "Restored after temporary omission."
            )
        result["portfolio_rollups"] = rollups

    # --- realized dividends + projected vs received (month-to-date) ---
    try:
        # Compute realized dividends from persisted DB events (faster & symbol-correct)
        mtd_start = as_of.replace(day=1)
        realized_mtd_raw = _summarize_events_from_db(db, plaid_account_id, mtd_start, as_of)
        mtd_realized = float(realized_mtd_raw.get("total_dividends") or 0.0)

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

        try:
            paydate_proj = project_paydate_window(holdings, mtd_start, as_of)
        except Exception:
            paydate_proj = None
        if paydate_proj is not None:
            projected_vs_received["alt"] = paydate_proj
        # upcoming ex-dates to end of month
        try:
            eom = (as_of.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            start_upcoming = as_of + timedelta(days=1)
            if start_upcoming <= eom:
                upcoming = project_upcoming_exdates(holdings, start_upcoming, eom)
                result["dividends_upcoming"] = upcoming
                if isinstance(upcoming, dict) and "meta" in upcoming:
                    result["dividends_upcoming_meta"] = upcoming["meta"]
        except Exception:
            pass

        # Additional windows from DB events: MTD breakdown + 30d / QTD / YTD
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

        # mark inactive symbols in realized windows
        def _tag_status(window: Dict[str, Any]) -> Dict[str, Any]:
            by_sym = window.get("by_symbol") or {}
            tagged: Dict[str, Any] = {}
            active_syms = {h.get("symbol") for h in holdings if h.get("symbol")}
            for sym, amt in by_sym.items():
                try:
                    amt_val = float(amt)
                except Exception:
                    amt_val = amt
                tagged[sym] = {
                    "amount": amt_val,
                    "status": "active" if sym in active_syms else "inactive",
                }
            window["by_symbol"] = tagged
            return window

        sum_30d = _tag_status(sum_30d)
        sum_qtd = _tag_status(sum_qtd)
        sum_ytd = _tag_status(sum_ytd)
        sum_mtd_tagged = _tag_status(realized_mtd_raw)
        realized_mtd_raw["by_symbol"] = sum_mtd_tagged.get("by_symbol", {})

        result["dividends"] = {
            "realized_mtd": realized_mtd_raw,
            "realized_mtd_detail": sum_mtd_tagged.get("by_symbol"),
            "projected_vs_received": projected_vs_received,
            "windows": {
                "30d": sum_30d,
                "qtd": sum_qtd,
                "ytd": sum_ytd,
            },
        }

        # Normalize by_symbol entries to {amount,status}
        try:
            active_syms = {h.get("symbol") for h in holdings if h.get("symbol")}

            def _norm(window: Dict[str, Any]) -> None:
                by_sym = window.get("by_symbol")
                if not isinstance(by_sym, dict):
                    return
                new_map: Dict[str, Any] = {}
                for sym, val in by_sym.items():
                    try:
                        amt_val = float(val if not isinstance(val, dict) else val.get("amount"))
                    except Exception:
                        amt_val = val if not isinstance(val, dict) else val.get("amount")
                    status_val = (
                        val.get("status") if isinstance(val, dict) else ("active" if sym in active_syms else "inactive")
                    )
                    new_map[sym] = {"amount": amt_val, "status": status_val}
                window["by_symbol"] = new_map

            divs_block = result.get("dividends") or {}
            mtd_block = divs_block.get("realized_mtd") or {}
            _norm(mtd_block)
            for w in (divs_block.get("windows") or {}).values():
                _norm(w)
            # mirror detail
            divs_block["realized_mtd_detail"] = mtd_block.get("by_symbol")
            result["dividends"] = divs_block
        except Exception:
            pass
    except Exception:
        # don't let dividend math break the snapshot; return zeroed shape
        mtd_start = as_of.replace(day=1)
        zero_win = {
            "label": "month_to_date",
            "start": mtd_start.isoformat(),
            "end": as_of.isoformat(),
        }
        result["dividends"] = {
            "realized_mtd": {
                "start": zero_win["start"],
                "end": zero_win["end"],
                "total_dividends": 0.0,
                "by_date": {},
                "by_month": {mtd_start.strftime("%Y-%m"): 0.0},
                "by_symbol": {},
            },
            "realized_mtd_detail": {},
            "projected_vs_received": {
                "window": zero_win,
                "projected": 0.0,
                "received": 0.0,
                "difference": 0.0,
                "pct_of_projection": None,
            },
            "windows": {},
        }

    snapshot_created_at = datetime.now(timezone.utc)
    result["meta"] = _build_meta("db", snapshot_created_at)

    # Attach goal progress & margin guidance before normalizing/caching
    _inject_goal_progress(result, as_of)
    _inject_margin_guidance(result, as_of)

    # Attach macro only for today; avoid leaking current macro into historical snapshots.
    if as_of == today:
    _attach_macro_block(result, refresh_flag=bool(refresh))

    for h in result.get("holdings", []):
        h.pop("trend", None)

    cache_key = compute_cache_key(result)
    final_snapshot = build_and_cache_snapshot(
        result, key=cache_key, ttl=SNAPSHOT_CACHE_TTL_SECONDS
    )

    # Merge gaps with existing snapshot for this date (only fill missing/None; keep new values otherwise).
    try:
        existing_for_merge = (
            db.query(HoldingSnapshot)
            .filter(
                HoldingSnapshot.plaid_account_id == plaid_account_id,
                HoldingSnapshot.as_of_date == as_of,
            )
            .one_or_none()
        )
        if existing_for_merge and existing_for_merge.snapshot:
            final_snapshot = _fill_gaps(final_snapshot, dict(existing_for_merge.snapshot))
    except Exception:
        pass

    # Validate required fields before persisting/serving
    _validate_snapshot(final_snapshot, allow_partial=price_partial)
    try:
        _persist_snapshot_file(final_snapshot, as_of, plaid_account_id)
    except Exception:
        pass

    # Attempt to generate/update weekly summary when a new daily snapshot is created.
    try:
        wfg = importlib.import_module("scripts.weekly_fusion_generator")
        # use cache when possible; generator will build if stale/missing
        wfg.generate_and_write(use_cache=True)
    except Exception:
        pass

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

        if existing:
            existing.snapshot = dict(final_snapshot)
        else:
            snap = HoldingSnapshot(
                plaid_account_id=plaid_account_id,
                as_of_date=as_of,
                snapshot=dict(final_snapshot),
            )
            db.add(snap)

        db.commit()
    except Exception:
        # don't break the API just because history write failed
        db.rollback()

    return _maybe_slim(final_snapshot)


class RefreshSnapshotsRequest(BaseModel):
    plaid_account_ids: Optional[List[int]] = None
    as_of: Optional[date] = None
    refresh: bool = True  # force rebuild (bypass price cache) on first run


@router.post("/refresh", tags=["holdings"])
def refresh_snapshots(payload: RefreshSnapshotsRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Recompute holdings snapshots for one or more plaid accounts.
    Useful for cron/health checks to keep caches warm.
    """
    as_of = payload.as_of or date.today()
    refresh_flag = bool(payload.refresh)

    try:
        client = LunchMoneyClient()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unable to init Lunch Money client: {e}")

    account_ids = payload.plaid_account_ids or []
    if not account_ids:
        try:
            account_ids = _discover_m1_investment_account_ids(client)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Unable to list plaid accounts: {e}")

    if not account_ids:
        raise HTTPException(status_code=404, detail="No plaid_account_ids provided or discovered.")

    overall_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    for pid in account_ids:
        item_start = time.perf_counter()
        try:
            snap = get_valued_holdings_for_plaid_account(
                plaid_account_id=pid,
                as_of=as_of,
                refresh=refresh_flag,
                goal_monthly=2000.0,
                symbols=None,
                apr_current_pct=None,
                apr_future_pct=None,
                apr_future_date=None,
                margin_mode=None,
                enrich=True,
                perf=True,
                perf_method="accurate",
                slim=False,
                db=db,
            )
            elapsed = time.perf_counter() - item_start
            meta = snap.get("meta") or {}
            totals = snap.get("totals") or {}
            results.append(
                {
                    "plaid_account_id": pid,
                    "elapsed_sec": round(elapsed, 3),
                    "served_from": meta.get("served_from"),
                    "cache_origin": (meta.get("cache") or {}).get("origin") or meta.get("cache_origin"),
                    "holdings": len(snap.get("holdings") or []),
                    "market_value": totals.get("market_value"),
                    "status": "ok",
                }
            )
        except Exception as e:
            results.append(
                {
                    "plaid_account_id": pid,
                    "elapsed_sec": round(time.perf_counter() - item_start, 3),
                    "error": str(e),
                    "status": "error",
                }
            )

    return {
        "as_of": as_of.isoformat(),
        "count": len(results),
        "refresh": refresh_flag,
        "total_sec": round(time.perf_counter() - overall_start, 3),
        "results": results,
    }
