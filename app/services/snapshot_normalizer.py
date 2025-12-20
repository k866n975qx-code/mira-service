from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from calendar import monthrange
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, getcontext
import numpy as np
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

from app.services.ultimate_fill import fill_ultimate_nulls
from app.services.field_registry import get_field_registry, FieldSpec

getcontext().prec = 28

# Cache lives alongside the repo to avoid network calls or external services.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".snapshot_cache")
)


def _as_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _append_warning(meta: Dict[str, Any], message: str) -> None:
    warnings = meta.setdefault("warnings", [])
    notes = meta.setdefault("notes", [])
    for collection in (warnings, notes):
        if message not in collection:
            collection.append(message)


def _ensure_dict(value: Any, default: Optional[Dict[str, Any]], meta: Dict[str, Any], label: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    _append_warning(meta, f"{label} was missing or invalid; replaced with empty object.")
    return default or {}


def _ensure_list(value: Any, meta: Dict[str, Any], label: str) -> List[Any]:
    if isinstance(value, list):
        return value
    _append_warning(meta, f"{label} was missing or invalid; replaced with empty list.")
    return []


def _ensure_meta(raw_meta: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    warnings = meta.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    meta["warnings"] = warnings
    notes = meta.get("notes")
    if not isinstance(notes, list):
        # mirror warnings into notes for downstream schemas that renamed the field
        notes = list(warnings)
    meta["notes"] = notes
    if "snapshot_created_at" not in meta:
        meta["snapshot_created_at"] = datetime.utcnow().isoformat()
        _append_warning(meta, "Added missing meta.snapshot_created_at.")
    if "snapshot_age_days" not in meta:
        meta["snapshot_age_days"] = 0
        _append_warning(meta, "Added missing meta.snapshot_age_days.")
    return meta


def _status_for(sym: str, held: set) -> str:
    return "active" if sym in held else "inactive"


def _normalize_by_symbol(by_symbol: dict, held: set) -> dict:
    out: Dict[str, Dict[str, Any]] = {}
    for s, v in (by_symbol or {}).items():
        if isinstance(v, dict):
            amt = float(v.get("amount", 0.0))
            st = v.get("status") or _status_for(s, held)
        else:
            amt = float(v or 0.0)
            st = _status_for(s, held)
        out[s] = {"amount": round(amt, 2), "status": st}
    return out


def _round2(x: float) -> float:
    from decimal import Decimal, ROUND_HALF_UP

    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(x: float) -> float:
    from decimal import Decimal, ROUND_HALF_UP

    return float(Decimal(x).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _parse_ymd(s):
    if not s:
        return None
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        return None


def _parse_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        try:
            return date.fromisoformat(val.split("T")[0])
        except Exception:
            return _parse_ymd(val)
    return None


def _fmt_ymd(d: date) -> str:
    return d.isoformat()


def _eom(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def _normalize_upcoming_window(snap: dict) -> None:
    upc = snap.setdefault("dividends_upcoming", {})
    win = upc.setdefault("window", {})
    win.setdefault("mode", "exdate")
    start = _parse_ymd(win.get("start")) or _parse_ymd(snap.get("as_of"))
    if not start:
        start = _parse_ymd(snap.get("prices_as_of")) or date.today()
    end = _parse_ymd(win.get("end"))
    if (end is None) or (end < start):
        end = _eom(start)
    win["start"] = _fmt_ymd(start)
    win["end"] = _fmt_ymd(end)


def _build_upcoming_events_from_holdings(snap: dict) -> list:
    win = snap.get("dividends_upcoming", {}).get("window", {})
    start = _parse_ymd(win.get("start"))
    end = _parse_ymd(win.get("end"))
    if not (start and end):
        return []

    events = []
    for h in snap.get("holdings", []):
        sym = h.get("symbol")
        amt = float(h.get("projected_monthly_dividend") or 0.0)
        ult = h.get("ultimate") or {}
        nxd = ult.get("next_ex_date_est") or h.get("next_ex_date_est")
        exd = _parse_ymd(nxd)
        if exd and (start <= exd <= end) and amt > 0:
            events.append({"symbol": sym, "ex_date_est": _fmt_ymd(exd), "amount_est": round(amt, 2)})
    return events


def _rebuild_dividends_upcoming_if_missing(snap: dict) -> None:
    upc = snap.setdefault("dividends_upcoming", {})
    events = upc.get("events")
    if not events:
        _normalize_upcoming_window(snap)
        synth = _build_upcoming_events_from_holdings(snap)
        upc["events"] = synth
        sum_events = round(sum(float(e.get("amount_est", 0.0)) for e in synth), 2)
        upc["projected"] = sum_events
        meta = upc.setdefault("meta", {})
        meta["sum_of_events"] = sum_events
        meta["matches_projected"] = True


def _clamp_as_of_for_cache(snap: dict) -> None:
    meta = snap.get("meta") or {}
    served = meta.get("served_from")
    if served != "cache":
        return
    p = _parse_ymd(snap.get("prices_as_of"))
    a = _parse_ymd(snap.get("as_of"))
    if p and a and a > p:
        snap["as_of"] = _fmt_ymd(p)
        meta.setdefault("warnings", []).append("as_of clamped to prices_as_of on cached snapshot.")


def _fmt_money(x):
    return float(f"{float(x):.2f}")


def _fmt_pct3(x):
    return float(f"{float(x):.3f}")


def _format_snapshot_numbers(snap: dict) -> None:
    t = snap.get("totals") or {}
    for k in ("market_value", "cost_basis", "unrealized_pnl"):
        if k in t:
            t[k] = _fmt_money(t[k])
    if "unrealized_pct" in t:
        t["unrealized_pct"] = _fmt_pct3(t["unrealized_pct"])

    inc = snap.get("income") or {}
    for k in (
        "forward_12m_total",
        "projected_monthly_income",
        "portfolio_current_yield_pct",
        "portfolio_yield_on_cost_pct",
    ):
        if k in inc:
            if k.endswith("_pct"):
                inc[k] = _fmt_pct3(inc[k])
            else:
                inc[k] = _fmt_money(inc[k])
    for margin_key in ("margin_to_portfolio_pct",):
        if margin_key in t and t.get(margin_key) is not None:
            t[margin_key] = round(float(t[margin_key]), 4)
        if margin_key in snap and snap.get(margin_key) is not None:
            snap[margin_key] = round(float(snap[margin_key]), 4)


def _round_floats(obj: Any, places: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _round_floats(v, places)
        return obj
    if isinstance(obj, list):
        for idx, v in enumerate(obj):
            obj[idx] = _round_floats(v, places)
        return obj
    return obj


def _preserve_upcoming_if_populated(original: dict, snap: dict) -> None:
    try:
        orig_upc = original.get("dividends_upcoming") or {}
        if orig_upc.get("events"):
            cur_upc = snap.setdefault("dividends_upcoming", {})
            if not cur_upc.get("events"):
                snap["dividends_upcoming"] = orig_upc
    except Exception:
        pass


def _fix_dividends_upcoming(snap: dict) -> None:
    upc = snap.setdefault("dividends_upcoming", {})
    win = upc.setdefault("window", {})
    win.setdefault("mode", "exdate")
    win.setdefault("start", snap.get("as_of"))
    win.setdefault("end", snap.get("as_of"))
    events = upc.setdefault("events", [])
    sum_events = _round2(sum(float(e.get("amount_est", 0.0)) for e in events))
    upc["projected"] = _round2(upc.get("projected", sum_events))
    meta = upc.setdefault("meta", {})
    meta["sum_of_events"] = sum_events
    meta["matches_projected"] = sum_events == upc["projected"]


def _sync_top_level_totals(snap: dict, meta: Dict[str, Any]) -> None:
    """
    Ensure legacy top-level mirrors exist for validators expecting both total_* and totals.* keys.
    """
    totals = snap.get("totals") or {}
    if "market_value" in totals:
        tmv = float(totals.get("market_value") or 0.0)
        existing = snap.get("total_market_value")
        if existing is not None and abs(float(existing) - tmv) > 0.01:
            _append_warning(meta, "total_market_value normalized to totals.market_value.")
        snap["total_market_value"] = tmv
    for field in ("margin_loan_balance", "margin_to_portfolio_pct"):
        if field in snap:
            snap.pop(field, None)
            _append_warning(meta, f"Removed redundant top-level {field}.")


def _approx_rollups_from_holdings(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight, cache-free portfolio rollups using only existing holding fields.
    """
    mv_total = sum(float(h.get("market_value") or 0.0) for h in holdings)

    def _weighted_average(keys: List[str]) -> Optional[float]:
        acc = 0.0
        weight = 0.0
        for h in holdings:
            mv = float(h.get("market_value") or 0.0)
            if mv <= 0.0:
                continue
            u = h.get("ultimate") or {}
            val = None
            for k in keys:
                if h.get(k) is not None:
                    val = h.get(k)
                    break
                if u.get(k) is not None:
                    val = u.get(k)
                    break
            if val is None:
                continue
            try:
                val_f = float(val)
            except Exception:
                continue
            acc += mv * val_f
            weight += mv
        if weight <= 0.0:
            return None
        return round(acc / weight, 3)

    perf: Dict[str, Optional[float]] = {}
    for label in ("1m", "3m", "6m", "12m"):
        key = f"twr_{label}_pct"
        avg = _weighted_average([key, f"{label}_twr_pct", f"twr{label}_pct"])
        if avg is not None:
            perf[key] = avg

    risk: Dict[str, Optional[float]] = {}
    vol30 = _weighted_average(["volatility_30d_pct", "vol_30d_pct", "vol30d_pct"])
    vol90 = _weighted_average(["volatility_90d_pct", "vol_90d_pct", "vol90d_pct"])
    mdd = _weighted_average(["max_drawdown_1y_pct", "mdd_1y_pct"])
    beta = _weighted_average(["beta_portfolio", "beta_3y", "beta"])
    ddn = _weighted_average(["downside_dev_1y_pct"])
    sharpe = _weighted_average(["sharpe_1y"])
    calmar = _weighted_average(["calmar_1y"])
    ddd = _weighted_average(["drawdown_duration_1y_days"])
    var95 = _weighted_average(["var_95_1d_pct"])
    cvar95 = _weighted_average(["cvar_95_1d_pct"])
    corr = _weighted_average(["corr_1y"])
    if vol30 is not None:
        risk["vol_30d_pct"] = vol30
    if vol90 is not None:
        risk["vol_90d_pct"] = vol90
    if mdd is not None:
        risk["max_drawdown_1y_pct"] = mdd
    if beta is not None:
        risk["beta_portfolio"] = beta
    if ddn is not None:
        risk["downside_dev_1y_pct"] = ddn
    if sharpe is not None:
        risk["sharpe_1y"] = sharpe
    if calmar is not None:
        risk["calmar_1y"] = calmar
    if ddd is not None:
        risk["drawdown_duration_1y_days"] = int(round(ddd))
    if var95 is not None:
        risk["var_95_1d_pct"] = var95
    if cvar95 is not None:
        risk["cvar_95_1d_pct"] = cvar95
    if corr is not None:
        risk["corr_1y"] = corr

    # Income stability: fallback approximation from holdings-level forward vs projected
    fwd_total = sum(float(h.get("forward_12m_dividend") or 0.0) for h in holdings)
    proj_total = sum(float(h.get("projected_monthly_dividend") or 0.0) for h in holdings)
    score = 0.0
    if fwd_total > 0:
        expected_monthly = fwd_total / 12.0
        if expected_monthly > 0:
            score = 100.0 * min(1.0, proj_total / expected_monthly)
    risk["income_stability_score"] = round(score, 2)

    return {
        "performance": perf,
        "risk": risk,
        "benchmark": "^GSPC",
        "meta": {
            "method": "approx-fallback",
            "note": "Restored after temporary omission; risk metrics derived from holdings where available.",
        },
    }


def _backfill_rollup_risk_from_holdings(
    holdings: List[Dict[str, Any]], risk: Dict[str, Any], meta: Dict[str, Any]
) -> None:
    if not isinstance(risk, dict):
        return

    mv_total = sum(float(h.get("market_value") or 0.0) for h in holdings)
    if mv_total <= 0:
        return

    def _weighted_average(keys: List[str]) -> Optional[float]:
        acc = 0.0
        weight = 0.0
        for h in holdings:
            mv = float(h.get("market_value") or 0.0)
            if mv <= 0.0:
                continue
            u = h.get("ultimate") or {}
            val = None
            for k in keys:
                if h.get(k) is not None:
                    val = h.get(k)
                    break
                if u.get(k) is not None:
                    val = u.get(k)
                    break
            if val is None:
                continue
            try:
                val_f = float(val)
            except Exception:
                continue
            acc += mv * val_f
            weight += mv
        if weight <= 0.0:
            return None
        return round(acc / weight, 3)

    mapping = {
        "vol_30d_pct": ["vol_30d_pct", "volatility_30d_pct"],
        "vol_90d_pct": ["vol_90d_pct", "volatility_90d_pct"],
        "max_drawdown_1y_pct": ["max_drawdown_1y_pct", "mdd_1y_pct"],
        "beta_portfolio": ["beta_portfolio", "beta_3y", "beta"],
        "downside_dev_1y_pct": ["downside_dev_1y_pct"],
        "sharpe_1y": ["sharpe_1y"],
        "calmar_1y": ["calmar_1y"],
        "drawdown_duration_1y_days": ["drawdown_duration_1y_days"],
        "var_95_1d_pct": ["var_95_1d_pct"],
        "cvar_95_1d_pct": ["cvar_95_1d_pct"],
        "corr_1y": ["corr_1y"],
    }

    filled: List[str] = []
    for field, keys in mapping.items():
        if risk.get(field) is None:
            v = _weighted_average(keys)
            if v is None:
                continue
            if field.endswith("_days"):
                risk[field] = int(round(v))
            else:
                risk[field] = v
            filled.append(field)
    if filled:
        _append_warning(meta, f"portfolio_rollups.risk backfilled from holdings: {', '.join(filled)}.")


def _prune_risk_aliases(rollups: Dict[str, Any], meta: Dict[str, Any]) -> None:
    risk = rollups.get("risk")
    if not isinstance(risk, dict):
        return
    if "volatility_90d_pct" in risk:
        if "vol_90d_pct" not in risk:
            risk["vol_90d_pct"] = risk.get("volatility_90d_pct")
        risk.pop("volatility_90d_pct", None)
        _append_warning(meta, "Removed duplicate volatility_90d_pct; kept vol_90d_pct.")
    if "beta_3y" in risk:
        if "beta_portfolio" not in risk:
            risk["beta_portfolio"] = risk.get("beta_3y")
        risk.pop("beta_3y", None)
        _append_warning(meta, "Removed duplicate beta_3y; kept beta_portfolio.")


def _ensure_rollups_version(snap: Dict[str, Any], meta: Dict[str, Any]) -> None:
    rollups = snap.get("portfolio_rollups") or {}
    meta_block = rollups.get("meta")
    if not isinstance(meta_block, dict):
        meta_block = {}
        rollups["meta"] = meta_block
    as_of = snap.get("as_of")
    served_from = (snap.get("meta") or {}).get("served_from")
    suffix = "cached" if served_from in ("cache", "cache_db") else "recomputed"
    desired_version = f"v{as_of}.{suffix}" if as_of else f"v.{suffix}"
    if meta_block.get("version") != desired_version:
        meta_block["version"] = desired_version
        _append_warning(meta, "portfolio_rollups.meta.version normalized.")
    snap["portfolio_rollups"] = rollups


def _clamp_goal_horizon(snap: Dict[str, Any], meta: Dict[str, Any]) -> None:
    gp = snap.get("goal_progress") or {}
    months = gp.get("months_to_goal")
    if isinstance(months, (int, float)) and months > 480:
        gp["months_to_goal"] = None
        gp["estimated_goal_date"] = None
        _append_warning(meta, "goal_progress.months_to_goal exceeded 480 months and was cleared.")
    snap["goal_progress"] = gp


def _harmonize_dividend_windows(snap: Dict[str, Any], meta: Dict[str, Any]) -> None:
    divs = snap.get("dividends") or {}
    realized = divs.get("realized_mtd") or {}
    win30 = (divs.get("windows") or {}).get("30d") or {}
    if "total_dividends" in win30:
        realized["total_dividends"] = win30.get("total_dividends")
        divs["realized_mtd"] = realized
        _append_warning(meta, "dividends.realized_mtd.total_dividends aligned to windows.30d.total_dividends.")
    snap["dividends"] = divs


def _normalize_meta_fields(snap: Dict[str, Any]) -> None:
    meta = snap.setdefault("meta", {})
    served_from = meta.get("served_from")
    if served_from and not meta.get("source"):
        meta["source"] = served_from
    if "cache_origin" in meta:
        # honor explicit cache origin, otherwise set a canonical name
        if not meta.get("cache_origin"):
            meta["cache_origin"] = "mira.snapshot_cache"
    elif meta.get("served_from") in ("cache", "cache_db"):
        meta["cache_origin"] = "mira.snapshot_cache"
    sca = meta.get("snapshot_created_at")
    if isinstance(sca, str):
        if "." in sca and not sca.endswith("Z"):
            meta["snapshot_created_at"] = sca.split(".")[0] + "Z"
        elif "T" in sca and not sca.endswith("Z"):
            meta["snapshot_created_at"] = sca + "Z"
    meta.setdefault("schema_version", "2.4")


def _recompute_totals_and_weights(snap: dict) -> None:
    hs = snap.get("holdings", [])
    mv = _round2(sum(float(h.get("market_value", 0.0)) for h in hs))
    cb = _round2(sum(float(h.get("cost_basis", 0.0)) for h in hs))
    pnl = _round2(mv - cb)
    upct = 0.0 if cb == 0 else _round3((mv - cb) / cb * 100.0)
    t = snap.setdefault("totals", {})
    t["market_value"] = mv
    t["cost_basis"] = cb
    t["unrealized_pnl"] = pnl
    t["unrealized_pct"] = upct
    if mv > 0:
        for h in hs:
            h["weight_pct"] = _round3(float(h.get("market_value", 0.0)) / mv * 100.0)
    snap["count"] = len(hs)


def _recompute_income(snap: dict) -> None:
    fwd = sum(float(h.get("forward_12m_dividend", 0.0)) for h in snap.get("holdings", []))
    monthly = fwd / 12.0
    mv = float(snap.get("totals", {}).get("market_value", 0.0))
    cb = float(snap.get("totals", {}).get("cost_basis", 0.0))
    curr = 0.0 if mv == 0 else (fwd / mv * 100.0)
    yoc = 0.0 if cb == 0 else (fwd / cb * 100.0)
    snap["income"] = {
        "forward_12m_total": _round2(fwd),
        "projected_monthly_income": _round3(monthly),
        "portfolio_current_yield_pct": _round3(curr),
        "portfolio_yield_on_cost_pct": _round3(yoc),
    }


def _compare_prices_vs_snapshot(snap: dict) -> None:
    import datetime as _dt

    meta = snap.setdefault("meta", {})
    p = snap.get("prices_as_of")
    sc = snap.get("meta", {}).get("snapshot_created_at")
    try:
        p_d = _dt.datetime.strptime(p, "%Y-%m-%d").date() if p else None
    except Exception:
        _append_warning(meta, "prices_as_of not parseable.")
        return

    def _parse_dt(s):
        if not s:
            return None
        fmts = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")
        for f in fmts:
            try:
                return _dt.datetime.strptime(s, f)
            except Exception:
                pass
        return None

    sc_dt = _parse_dt(sc)
    if sc_dt and p_d and p_d > sc_dt.date():
        _append_warning(meta, "prices_as_of is after snapshot_created_at; price snapshot may precede price time.")


def _validate_div_windows(snap: dict) -> None:
    meta = snap.setdefault("meta", {})
    wins = (snap.get("dividends") or {}).get("windows") or {}

    def sum_dict(d):
        return round(sum(float(v) for v in (d or {}).values()), 2)

    for key in ("30d", "qtd", "ytd"):
        w = wins.get(key) or {}
        by_date = sum_dict(w.get("by_date"))
        by_month = sum_dict(w.get("by_month"))
        by_symbol = round(
            sum(
                (float(v.get("amount", 0.0)) if isinstance(v, dict) else float(v or 0.0))
                for v in (w.get("by_symbol") or {}).values()
            ),
            2,
        )
        total = float(w.get("total_dividends", by_symbol))
        if abs(by_date - total) > 0.01:
            _append_warning(meta, f"dividends.windows.{key} by_date sum {by_date} != total {total}")
        if abs(by_month - total) > 0.01:
            _append_warning(meta, f"dividends.windows.{key} by_month sum {by_month} != total {total}")
        if abs(by_symbol - total) > 0.01:
            _append_warning(meta, f"dividends.windows.{key} by_symbol sum {by_symbol} != total {total}")


def _set_cached_flag(snap: dict) -> None:
    meta = snap.setdefault("meta", {})
    served = meta.get("served_from")
    caches = meta.get("cache", {})
    bypass = any((v or {}).get("bypassed", False) for v in caches.values())
    snap["cached"] = served in ("cache", "cache_db") and not bypass


def _sanity(snap: dict) -> None:
    meta = snap.setdefault("meta", {})
    wsum = round(sum(float(h.get("weight_pct", 0.0)) for h in snap.get("holdings", [])), 3)
    if abs(wsum - 100.0) > 0.1:
        _append_warning(meta, f"holding weight sum {wsum}% != 100% (±0.1)")
    fwd_h = round(sum(float(h.get("forward_12m_dividend", 0.0)) for h in snap.get("holdings", [])), 2)
    fwd_i = round(float(snap.get("income", {}).get("forward_12m_total", 0.0)), 2)
    if abs(fwd_h - fwd_i) > 0.01:
        _append_warning(meta, f"income.forward_12m_total {fwd_i} != sum holdings {fwd_h}")


def _iter_path_values(obj: Any, spec_path: str):
    """
    Yield (concrete_path, value) pairs for a FieldSpec path that may include [] to denote list items.
    Example: "holdings[].ultimate.beta_3y" -> yields holdings[0].ultimate.beta_3y etc.
    """
    parts = spec_path.split(".")

    def _recurse(current, idx, prefix):
        if idx >= len(parts):
            yield prefix, current
            return
        part = parts[idx]
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part
        next_val = None
        if isinstance(current, dict):
            next_val = current.get(key)
        if is_list:
            if isinstance(next_val, list):
                for i, item in enumerate(next_val):
                    new_prefix = f"{prefix}.{key}[{i}]" if prefix else f"{key}[{i}]"
                    yield from _recurse(item, idx + 1, new_prefix)
            else:
                new_prefix = f"{prefix}.{key}[]" if prefix else f"{key}[]"
                yield new_prefix, None
        else:
            new_prefix = f"{prefix}.{key}" if prefix else key
            if next_val is None:
                yield new_prefix, None
            else:
                yield from _recurse(next_val, idx + 1, new_prefix)

    yield from _recurse(obj, 0, "")


def _compute_coverage(snap: Dict[str, Any], extra_conflicts: Optional[List[str]] = None) -> Dict[str, Any]:
    specs = get_field_registry()
    total = filled = pulled = derived = validated = missing = 0
    missing_paths: List[str] = []
    for spec in specs:
        for concrete_path, value in _iter_path_values(snap, spec.path):
            total += 1
            if value is None:
                missing += 1
                missing_paths.append(concrete_path)
                continue
            filled += 1
            if spec.source_type == "pulled":
                pulled += 1
            elif spec.source_type == "derived":
                derived += 1
            elif spec.source_type == "validated":
                validated += 1
    denom = total or 1
    return {
        "filled_pct": round(filled / denom * 100.0, 2),
        "pulled_pct": round(pulled / denom * 100.0, 2),
        "derived_pct": round(derived / denom * 100.0, 2),
        "validated_pct": round(validated / denom * 100.0, 2),
        "missing_pct": round(missing / denom * 100.0, 2),
        "missing_paths": missing_paths,
        "conflict_paths": sorted(set(extra_conflicts or [])),
    }


def _prov_entry(provider: str, source_type: str, method: str, inputs: list[str], fetched_at: Optional[str] = None, note: Optional[str] = None) -> dict:
    return {
        "source_type": source_type,
        "provider": provider,
        "method": method,
        "inputs": inputs,
        "validated_by": [],
        "fetched_at": fetched_at or datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "note": note,
    }


def _canonicalize_by_symbol(
    by_symbol: Dict[str, Any],
    holdings_symbols: List[str],
    meta: Dict[str, Any],
    label: str,
) -> Dict[str, Dict[str, Any]]:
    canonical: Dict[str, Dict[str, Any]] = {}
    for symbol, value in (by_symbol or {}).items():
        amount = value
        status = None
        if isinstance(value, dict):
            amount = value.get("amount")
            status = value.get("status")
        dec_amount = _as_decimal(amount)
        if status not in ("active", "inactive"):
            status = "inactive" if symbol not in holdings_symbols else "active"
            _append_warning(meta, f"{label}.{symbol} status normalized to {status}.")
        if not isinstance(value, dict):
            _append_warning(meta, f"{label}.{symbol} converted from number to object.")
        canonical[symbol] = {"amount": float(dec_amount), "status": status}
    return canonical


def _sum_dict_values(d: Dict[str, Any]) -> Decimal:
    total = Decimal("0")
    for v in (d or {}).values():
        total += _as_decimal(v)
    return total


def _canonicalize_dividend_window(
    window: Dict[str, Any],
    holdings_symbols: List[str],
    meta: Dict[str, Any],
    label: str,
    start_fallback: Optional[str],
    end_fallback: Optional[str],
) -> Dict[str, Any]:
    win = _ensure_dict(window, {}, meta, f"dividends.windows.{label}")
    if "start" not in win and start_fallback is not None:
        win["start"] = start_fallback
        _append_warning(meta, f"dividends.windows.{label}.start added.")
    if "end" not in win and end_fallback is not None:
        win["end"] = end_fallback
        _append_warning(meta, f"dividends.windows.{label}.end added.")
    win["by_date"] = _ensure_dict(win.get("by_date"), {}, meta, f"dividends.windows.{label}.by_date")
    win["by_month"] = _ensure_dict(win.get("by_month"), {}, meta, f"dividends.windows.{label}.by_month")
    win["by_symbol"] = _canonicalize_by_symbol(
        _ensure_dict(win.get("by_symbol"), {}, meta, f"dividends.windows.{label}.by_symbol"),
        holdings_symbols,
        meta,
        f"dividends.windows.{label}.by_symbol",
    )
    sum_by_symbol = sum(_as_decimal(v.get("amount")) for v in win["by_symbol"].values())
    sum_by_date = _sum_dict_values(win["by_date"])
    if "total_dividends" in win:
        existing = _as_decimal(win.get("total_dividends"))
    else:
        existing = Decimal("0")
    desired = sum_by_symbol
    if abs(sum_by_date - desired) > Decimal("0.01"):
        _append_warning(meta, f"dividends.windows.{label} by_date and by_symbol were misaligned; using by_symbol total.")
    if abs(existing - desired) > Decimal("0.01"):
        _append_warning(meta, f"dividends.windows.{label}.total_dividends recomputed.")
    win["total_dividends"] = float(desired)
    return win


HOLDING_ULTIMATE_FIELDS = [
    # dividends
    "trailing_12m_div_ps",
    "trailing_12m_yield_pct",
    "forward_yield_pct",
    "forward_yield_method",
    "div_growth_1y_pct",
    "div_growth_3y_cagr_pct",
    "div_growth_5y_cagr_pct",
    "div_cv_12p",
    "div_cut_3m_vs_12m_pct",
    "derived_fields",
    # risk
    "vol_30d_pct",
    "vol_90d_pct",
    "beta_3y",
    "downside_dev_1y_pct",
    "sharpe_1y",
    "sortino_1y",
    "calmar_1y",
    "max_drawdown_1y_pct",
    "drawdown_duration_1y_days",
    "var_95_1d_pct",
    "cvar_95_1d_pct",
    "corr_1y",
    # performance
    "twr_1m_pct",
    "twr_3m_pct",
    "twr_6m_pct",
    "twr_12m_pct",
]


def _ensure_holding_shapes(holdings: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    derived_any = False
    for h in holdings:
        if not isinstance(h, dict):
            continue
        ult = h.setdefault("ultimate", {})
        if not isinstance(ult, dict):
            ult = {}
            h["ultimate"] = ult
            _append_warning(meta, f"ultimate block added for holding {h.get('symbol')}.")
        # normalize derived_fields container
        if "derived_fields" in ult and not isinstance(ult.get("derived_fields"), list):
            ult["derived_fields"] = []
        # Fallback yields if missing
        derived_fields: List[str] = []
        try:
            lp = float(h.get("last_price")) if h.get("last_price") is not None else None
        except Exception:
            lp = None
        try:
            div_ps = float(ult.get("trailing_12m_div_ps")) if ult.get("trailing_12m_div_ps") is not None else None
        except Exception:
            div_ps = None
        if ult.get("trailing_12m_yield_pct") is None and div_ps is not None and lp:
            try:
                ult["trailing_12m_yield_pct"] = _round3((div_ps / lp) * 100.0)
                derived_fields.append("trailing_12m_yield_pct")
            except Exception:
                pass
        fwd_total = h.get("forward_12m_dividend")
        mv = h.get("market_value")
        try:
            fwd_total_f = float(fwd_total) if fwd_total is not None else None
            mv_f = float(mv) if mv is not None else None
        except Exception:
            fwd_total_f = mv_f = None
        if ult.get("forward_yield_pct") is None and fwd_total_f and mv_f and mv_f != 0:
            try:
                ult["forward_yield_pct"] = _round3((fwd_total_f / mv_f) * 100.0)
                derived_fields.append("forward_yield_pct")
            except Exception:
                pass
        if derived_fields:
            ultd = ult.get("derived_fields")
            if not isinstance(ultd, list):
                ultd = []
                ult["derived_fields"] = ultd
            for f in sorted(derived_fields):
                if f not in ultd:
                    ultd.append(f)
            derived_any = True
    if derived_any:
        _append_warning(meta, "Filled yield fields with derived fallbacks where missing.")


def _normalize_dividends(snap: Dict[str, Any], meta: Dict[str, Any], holdings_symbols: List[str]) -> Dict[str, Any]:
    dividends = _ensure_dict(snap.get("dividends"), {}, meta, "dividends")
    as_of = snap.get("as_of")
    realized = _ensure_dict(dividends.get("realized_mtd"), {}, meta, "dividends.realized_mtd")
    realized.setdefault("start", as_of)
    realized.setdefault("end", as_of)
    realized["by_date"] = _ensure_dict(realized.get("by_date"), {}, meta, "dividends.realized_mtd.by_date")
    realized["by_month"] = _ensure_dict(realized.get("by_month"), {}, meta, "dividends.realized_mtd.by_month")
    realized["by_symbol"] = _canonicalize_by_symbol(
        _ensure_dict(realized.get("by_symbol"), {}, meta, "dividends.realized_mtd.by_symbol"),
        holdings_symbols,
        meta,
        "dividends.realized_mtd.by_symbol",
    )
    sum_by_symbol = sum(_as_decimal(v.get("amount")) for v in realized["by_symbol"].values())
    realized["total_dividends"] = float(sum_by_symbol)
    dividends["realized_mtd"] = realized

    windows = _ensure_dict(dividends.get("windows"), {}, meta, "dividends.windows")
    for label in ("30d", "qtd", "ytd"):
        windows[label] = _canonicalize_dividend_window(
            windows.get(label) or {},
            holdings_symbols,
            meta,
            label,
            as_of,
            as_of,
        )
    dividends["windows"] = windows
    dividends["projected_vs_received"] = _ensure_dict(
        dividends.get("projected_vs_received"), {}, meta, "dividends.projected_vs_received"
    )
    snap["dividends"] = dividends
    return dividends


def _normalize_dividends_upcoming(snap: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    divu = _ensure_dict(snap.get("dividends_upcoming"), {}, meta, "dividends_upcoming")
    as_of = snap.get("as_of")
    divu["window"] = _ensure_dict(divu.get("window"), {}, meta, "dividends_upcoming.window")
    divu["window"].setdefault("mode", "exdate")
    divu["window"].setdefault("start", as_of)
    divu["window"].setdefault("end", as_of)
    divu["events"] = _ensure_list(divu.get("events"), meta, "dividends_upcoming.events")
    meta_block = _ensure_dict(divu.get("meta"), {}, meta, "dividends_upcoming.meta")

    sum_events = Decimal("0")
    for ev in divu["events"]:
        if not isinstance(ev, dict):
            continue
        amt = _as_decimal(ev.get("amount_est"))
        ev["amount_est"] = float(amt)
        sum_events += amt
    divu["meta"] = meta_block
    sum_events_q = sum_events.quantize(Decimal("0.01"))
    divu["meta"]["sum_of_events"] = float(sum_events_q)
    projected = _as_decimal(divu.get("projected"))
    projected_q = projected.quantize(Decimal("0.01")) if projected.is_finite() else Decimal("0.00")
    matches = sum_events_q == projected_q
    divu["meta"]["matches_projected"] = bool(matches)
    if "projected" not in divu:
        divu["projected"] = float(sum_events_q)
        _append_warning(meta, "dividends_upcoming.projected filled from events.")
    if "dividends_upcoming_meta" in snap:
        _append_warning(meta, "Removed dividends_upcoming_meta in favor of dividends_upcoming.meta.")
        snap.pop("dividends_upcoming_meta", None)
    return divu


def _normalize_portfolio_rollups(snap: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    pr = _ensure_dict(snap.get("portfolio_rollups"), {}, meta, "portfolio_rollups")
    pr["performance"] = _ensure_dict(pr.get("performance"), {}, meta, "portfolio_rollups.performance")
    pr["risk"] = _ensure_dict(pr.get("risk"), {}, meta, "portfolio_rollups.risk")
    snap["portfolio_rollups"] = pr
    return pr


ROLLUP_RISK_FIELDS = [
    "vol_30d_pct",
    "vol_90d_pct",
    "max_drawdown_1y_pct",
    "beta_portfolio",
    "downside_dev_1y_pct",
    "sharpe_1y",
    "calmar_1y",
    "drawdown_duration_1y_days",
    "var_95_1d_pct",
    "cvar_95_1d_pct",
    "corr_1y",
    "income_stability_score",
]


def _ensure_rollup_risk_fields(risk: Dict[str, Any], meta: Dict[str, Any]) -> None:
    missing: List[str] = []
    for field in ROLLUP_RISK_FIELDS:
        if field not in risk:
            risk[field] = None
            missing.append(field)
    if missing:
        _append_warning(meta, f"Filled missing portfolio_rollups.risk fields: {', '.join(missing)}.")


def normalize_snapshot(raw: Dict[str, Any]) -> Dict[str, Any]:
    original_raw = copy.deepcopy(raw or {})
    snap = copy.deepcopy(original_raw)
    snap_meta = _ensure_meta(snap.get("meta"))
    snap["meta"] = snap_meta
    holdings = _ensure_list(snap.get("holdings"), snap_meta, "holdings")
    snap["holdings"] = holdings
    _ensure_holding_shapes(holdings, snap_meta)
    holdings_symbols = [h.get("symbol") for h in holdings if isinstance(h, dict) and h.get("symbol")]

    totals = _ensure_dict(snap.get("totals"), {}, snap_meta, "totals")
    snap["totals"] = totals
    income = _ensure_dict(snap.get("income"), {}, snap_meta, "income")
    snap["income"] = income
    snap["missing_prices"] = _ensure_list(snap.get("missing_prices"), snap_meta, "missing_prices")
    snap.setdefault("cached", False)

    _normalize_dividends(snap, snap_meta, holdings_symbols)
    held_syms = {h.get("symbol") for h in snap.get("holdings", []) if isinstance(h, dict)}
    wins = snap.setdefault("dividends", {}).setdefault("windows", {})
    for k in ("30d", "qtd", "ytd"):
        w = wins.setdefault(k, {})
        w["by_symbol"] = _normalize_by_symbol(w.get("by_symbol", {}), held_syms)
    rm = snap["dividends"].setdefault("realized_mtd", {})
    rm["by_symbol"] = _normalize_by_symbol(rm.get("by_symbol", {}), held_syms)
    _normalize_portfolio_rollups(snap, snap_meta)
    _ensure_rollup_risk_fields(snap.get("portfolio_rollups", {}).get("risk", {}), snap_meta)

    snap["goal_progress"] = _ensure_dict(snap.get("goal_progress"), {}, snap_meta, "goal_progress")
    snap["goal_progress_net"] = _ensure_dict(snap.get("goal_progress_net"), {}, snap_meta, "goal_progress_net")
    snap["margin_guidance"] = _ensure_dict(snap.get("margin_guidance"), {}, snap_meta, "margin_guidance")

    # Track initial provenance for metadata pulled from upstream snapshot
    snap.setdefault("holdings_provenance", _prov_entry("db", "pulled", "snapshot_source", ["db_snapshot"]))
    snap.setdefault("income_provenance", snap.get("income_provenance") or {})
    snap.setdefault("totals_provenance", snap.get("totals_provenance") or {})

    if "cost_basis" not in totals:
        totals["cost_basis"] = 0.0
        _append_warning(snap_meta, "totals.cost_basis added as 0.0.")
    ml_balance = snap.get("margin_loan_balance", totals.get("margin_loan_balance"))
    if ml_balance is None:
        ml_balance = 0.0
        _append_warning(snap_meta, "margin_loan_balance added as 0.0.")
    totals["margin_loan_balance"] = float(_as_decimal(ml_balance))
    snap["margin_loan_balance"] = totals["margin_loan_balance"]

    # -------- Recompute totals --------
    mv_total = sum(_as_decimal(h.get("market_value")) for h in holdings if isinstance(h, dict))
    pnl_total = sum(_as_decimal(h.get("unrealized_pnl")) for h in holdings if isinstance(h, dict))
    totals["market_value"] = float(mv_total)
    totals["unrealized_pnl"] = float(pnl_total)
    cb_dec = _as_decimal(totals.get("cost_basis"))
    totals["unrealized_pct"] = float(((pnl_total / cb_dec) * Decimal("100")) if cb_dec != 0 else Decimal("0"))
    totals["margin_to_portfolio_pct"] = round(
        float(
            ((Decimal(str(totals.get("margin_loan_balance", 0))) / mv_total) * Decimal("100"))
            if mv_total != 0
            else Decimal("0")
        ),
        4,
    )
    _append_warning(snap_meta, "Totals recomputed from holdings.")

    # -------- Recompute weights if needed --------
    if holdings and mv_total != 0:
        current_sum = sum(_as_decimal(h.get("weight_pct")) for h in holdings)
        if abs(current_sum - Decimal("100.000")) > Decimal("0.05"):
            weights: List[Decimal] = []
            mv_total_dec = mv_total
            for h in holdings:
                mv = _as_decimal(h.get("market_value"))
                pct = (mv / mv_total_dec) * Decimal("100")
                pct = pct.quantize(Decimal("0.001"))
                weights.append(pct)
            if weights:
                diff = Decimal("100.000") - sum(weights)
                weights[-1] = (weights[-1] + diff).quantize(Decimal("0.001"))
                for h, pct in zip(holdings, weights):
                    if isinstance(h, dict):
                        h["weight_pct"] = float(pct)
                _append_warning(snap_meta, "Recomputed holding weights.")

    # -------- Recompute income --------
    fwd_total = sum(_as_decimal(h.get("forward_12m_dividend")) for h in holdings if isinstance(h, dict))
    proj_monthly = sum(_as_decimal(h.get("projected_monthly_dividend")) for h in holdings if isinstance(h, dict))
    income["forward_12m_total"] = float(fwd_total)
    income["projected_monthly_income"] = float(proj_monthly)
    income["portfolio_current_yield_pct"] = float(
        ((fwd_total / mv_total) * Decimal("100")) if mv_total != 0 else Decimal("0")
    )
    income["portfolio_yield_on_cost_pct"] = float(
        ((fwd_total / cb_dec) * Decimal("100")) if cb_dec != 0 else Decimal("0")
    )
    _append_warning(snap_meta, "Income recomputed from holdings.")

    # -------- Dividends upcoming meta --------
    _normalize_dividends_upcoming(snap, snap_meta)

    # -------- Count --------
    snap["count"] = len(holdings)

    # -------- Deduplicate --------
    if "dividends_upcoming_meta" in snap:
        snap.pop("dividends_upcoming_meta", None)
        _append_warning(snap_meta, "Removed dividends_upcoming_meta duplicate.")
    if "realized_mtd_detail" in snap.get("dividends", {}):
        snap["dividends"].pop("realized_mtd_detail", None)
        _append_warning(snap_meta, "Removed dividends.realized_mtd_detail duplicate.")

    _normalize_portfolio_rollups(snap, snap_meta)

    _recompute_totals_and_weights(snap)
    _recompute_income(snap)
    rollups = snap.get("portfolio_rollups") or {}
    perf_block = rollups.get("performance") or {}
    risk_block = rollups.get("risk") or {}
    needs_rollups = (not perf_block) or (not risk_block) or ("benchmark" not in rollups)
    if needs_rollups:
        approx_rollups = _approx_rollups_from_holdings(holdings)
        rollups.setdefault("performance", {})
        rollups.setdefault("risk", {})
        for k, v in (approx_rollups.get("performance") or {}).items():
            rollups["performance"].setdefault(k, v)
        for k, v in (approx_rollups.get("risk") or {}).items():
            rollups["risk"].setdefault(k, v)
        rollups.setdefault("benchmark", approx_rollups.get("benchmark", "^GSPC"))
        meta_block = rollups.setdefault("meta", {})
        meta_block.setdefault("method", (approx_rollups.get("meta") or {}).get("method", "approx-fallback"))
        meta_block.setdefault("note", (approx_rollups.get("meta") or {}).get("note", "Restored after temporary omission."))
        snap["portfolio_rollups"] = rollups
        _append_warning(snap_meta, "portfolio_rollups restored with approximate rollups.")
    _prune_risk_aliases(rollups, snap_meta)
    _backfill_rollup_risk_from_holdings(holdings, rollups.get("risk") or {}, snap_meta)
    _ensure_rollup_risk_fields(rollups.get("risk") or {}, snap_meta)
    _ensure_rollups_version(snap, snap_meta)
    _clamp_goal_horizon(snap, snap_meta)
    _fix_dividends_upcoming(snap)
    _harmonize_dividend_windows(snap, snap_meta)
    _preserve_upcoming_if_populated(original_raw, snap)
    _normalize_upcoming_window(snap)
    _rebuild_dividends_upcoming_if_missing(snap)
    _compare_prices_vs_snapshot(snap)
    _validate_div_windows(snap)
    _clamp_as_of_for_cache(snap)

    # Fill remaining nulls in holdings[*].ultimate using fallback providers + price-derived metrics.
    # This is best-effort and should not override non-null fields.
    as_of_date = _parse_date(snap.get("as_of")) or date.today()
    conflict_paths: List[str] = []
    for h in holdings:
        ult = h.get("ultimate")
        if isinstance(ult, dict):
            filled, prov, conflicts = fill_ultimate_nulls(
                h.get("symbol") or "",
                ult,
                benchmark_symbol=rollups.get("benchmark") or "^GSPC",
                as_of=as_of_date,
            )
            h["ultimate"] = filled
            if conflicts:
                conflict_paths.extend(conflicts)
            # optional provenance for debugging/validation
            if prov:
                if h.get("last_price") is None and prov.get("_last_price_value") is not None:
                    h["last_price"] = prov["_last_price_value"]
                h.setdefault("ultimate_provenance", {})
                for k, v in prov.items():
                    if k.startswith("_"):
                        continue
                    if hasattr(v, "__dict__"):
                        v = v.__dict__
                    h["ultimate_provenance"][k] = v

    _format_snapshot_numbers(snap)
    _sync_top_level_totals(snap, snap_meta)
    # Reset cache bypass flags after recompute
    for entry in (snap_meta.get("cache") or {}).values():
        if isinstance(entry, dict):
            entry["bypassed"] = False
    snap = _round_floats(snap, places=4)
    _normalize_meta_fields(snap)
    _set_cached_flag(snap)
    _sanity(snap)
    # Provenance for top-level aggregates (best-effort, derived internally)
    snap["totals_provenance"] = {
        k: _prov_entry("internal", "derived", "recompute_totals", ["holdings"])
        for k in (snap.get("totals") or {}).keys()
    }
    snap["income_provenance"] = {
        k: _prov_entry("internal", "derived", "income_from_holdings", ["holdings"])
        for k in (snap.get("income") or {}).keys()
    }
    perf_prov = {}
    for k in (snap.get("portfolio_rollups", {}).get("performance") or {}).keys():
        perf_prov[k] = _prov_entry("internal", "derived", "compute_portfolio_rollups", ["holdings"])
    risk_prov = {}
    for k in (snap.get("portfolio_rollups", {}).get("risk") or {}).keys():
        risk_prov[k] = _prov_entry("internal", "derived", "compute_portfolio_rollups", ["holdings"])
    snap["portfolio_rollups_provenance"] = {
        "performance": perf_prov,
        "risk": risk_prov,
    }
    snap["dividends_provenance"] = _prov_entry("internal", "derived", "normalize_dividends", ["transactions"])
    snap["dividends_upcoming_provenance"] = _prov_entry("internal", "derived", "project_upcoming_dividends", ["holdings"])
    snap["goal_progress_provenance"] = _prov_entry("internal", "derived", "goal_progress_calculator", ["income", "goal_settings"])
    snap["margin_guidance_provenance"] = _prov_entry("internal", "derived", "margin_guidance_calculator", ["totals", "income"])
    snap_meta["notes"] = list(dict.fromkeys((snap_meta.get("notes") or []) + (snap_meta.get("warnings") or [])))
    snap_meta.pop("warnings", None)
    # Coverage accounting
    snap["coverage"] = _compute_coverage(snap, extra_conflicts=conflict_paths)
    # Validation warnings (non-fatal): appended to meta.notes and recorded as conflicts
    warns = validate_snapshot(snap, raise_on_error=False)
    if warns:
        snap.setdefault("meta", {}).setdefault("notes", []).extend([f"validation_warning: {w}" for w in warns])
        if isinstance(snap.get("coverage"), dict):
            merged_conflicts = list(dict.fromkeys((snap["coverage"].get("conflict_paths") or []) + warns))
            snap["coverage"]["conflict_paths"] = merged_conflicts

    # NOTE: Do NOT drop nulls. We keep them so clients can see what is truly missing.
    # Nulls should be filled upstream (ultimate enrichment / dividends / etc) when possible.


    return snap


def validate_snapshot(snapshot: Dict[str, Any], raise_on_error: bool = True) -> List[str]:
    errors: List[str] = []
    holdings = snapshot.get("holdings") or []
    totals = snapshot.get("totals") or {}
    income = snapshot.get("income") or {}
    divu = snapshot.get("dividends_upcoming") or {}

    if snapshot.get("count") != len(holdings):
        errors.append("count does not match holdings length.")

    mv_total = sum(_as_decimal(h.get("market_value")) for h in holdings if isinstance(h, dict))
    totals_mv = _as_decimal(totals.get("market_value"))
    if abs(mv_total - totals_mv) > Decimal("0.01"):
        errors.append("totals.market_value does not match summed holdings.market_value.")

    cb_total = sum(_as_decimal(h.get("cost_basis")) for h in holdings if isinstance(h, dict))
    totals_cb = _as_decimal(totals.get("cost_basis"))
    if abs(cb_total - totals_cb) > Decimal("0.01"):
        errors.append("totals.cost_basis does not match summed holdings.cost_basis.")

    recomputed_upnl = mv_total - cb_total
    if abs(recomputed_upnl - _as_decimal(totals.get("unrealized_pnl"))) > Decimal("0.01"):
        errors.append("totals.unrealized_pnl does not match derived value.")
    recomputed_upct = (recomputed_upnl / cb_total * Decimal("100")) if cb_total != 0 else Decimal("0")
    if abs(recomputed_upct - _as_decimal(totals.get("unrealized_pct"))) > Decimal("0.01"):
        errors.append("totals.unrealized_pct does not match derived value.")

    fwd_sum = sum(_as_decimal(h.get("forward_12m_dividend")) for h in holdings if isinstance(h, dict))
    if abs(fwd_sum - _as_decimal(income.get("forward_12m_total"))) > Decimal("0.01"):
        errors.append("income.forward_12m_total does not match holdings.")

    proj_sum = sum(_as_decimal(h.get("projected_monthly_dividend")) for h in holdings if isinstance(h, dict))
    if abs(proj_sum - _as_decimal(income.get("projected_monthly_income"))) > Decimal("0.02"):
        errors.append("income.projected_monthly_income does not match holdings.")

    cyield = (fwd_sum / mv_total * Decimal("100")) if mv_total != 0 else Decimal("0")
    if abs(cyield - _as_decimal(income.get("portfolio_current_yield_pct"))) > Decimal("0.01"):
        errors.append("income.portfolio_current_yield_pct does not match derived value.")
    yoc = (fwd_sum / cb_total * Decimal("100")) if cb_total != 0 else Decimal("0")
    if abs(yoc - _as_decimal(income.get("portfolio_yield_on_cost_pct"))) > Decimal("0.01"):
        errors.append("income.portfolio_yield_on_cost_pct does not match derived value.")

    weight_sum = sum(_as_decimal(h.get("weight_pct")) for h in holdings if isinstance(h, dict))
    if abs(weight_sum - Decimal("100.000")) > Decimal("0.05"):
        errors.append("Holding weights do not sum to 100% within tolerance.")

    margin_pct = ( _as_decimal(totals.get("margin_loan_balance")) / mv_total * Decimal("100") ) if mv_total != 0 else Decimal("0")
    if abs(margin_pct - _as_decimal(totals.get("margin_to_portfolio_pct"))) > Decimal("0.01"):
        errors.append("totals.margin_to_portfolio_pct does not match derived value.")

    meta = snapshot.setdefault("meta", {})
    gp = snapshot.get("goal_progress") or {}
    if isinstance(gp.get("months_to_goal"), (int, float)) and gp["months_to_goal"] > 480:
        gp["months_to_goal"] = None
        gp["estimated_goal_date"] = None
        _append_warning(meta, "goal_progress.months_to_goal exceeded 480 months and was cleared during validation.")
    snapshot["goal_progress"] = gp

    # Ensure legacy top-level mirrors exist and match totals for downstream validators
    tmv_top = snapshot.get("total_market_value")
    if tmv_top is None:
        snapshot["total_market_value"] = float(totals_mv)
        _append_warning(meta, "total_market_value filled from totals.market_value during validation.")
    elif abs(_as_decimal(tmv_top) - totals_mv) > Decimal("0.01"):
        snapshot["total_market_value"] = float(totals_mv)
        _append_warning(meta, "total_market_value normalized to totals.market_value during validation.")

    meta_block = divu.get("meta") or {}
    _prune_risk_aliases(snapshot.get("portfolio_rollups") or {}, meta)
    projected = _as_decimal(divu.get("projected"))
    sum_events = _as_decimal(meta_block.get("sum_of_events"))
    projected_q = projected.quantize(Decimal("0.01"))
    sum_events_q = sum_events.quantize(Decimal("0.01"))

    def _coerce_bool(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in ("true", "1", "yes", "y", "on"):
                return True
            if lv in ("false", "0", "no", "n", "off"):
                return False
        return None

    matches = _coerce_bool(meta_block.get("matches_projected"))
    matches_correct = sum_events_q == projected_q
    if matches is None or matches != matches_correct:
        meta_block["matches_projected"] = matches_correct
        meta_block["sum_of_events"] = float(sum_events_q)
        try:
            _append_warning(
                snapshot.setdefault("meta", {}),
                "dividends_upcoming.meta.matches_projected corrected during validation.",
            )
        except Exception:
            pass

    if "dividends_upcoming_meta" in snapshot:
        errors.append("Duplicate field dividends_upcoming_meta remains.")
    if "realized_mtd_detail" in (snapshot.get("dividends") or {}):
        errors.append("Duplicate field dividends.realized_mtd_detail remains.")

    meta_notes = meta.get("notes") or []
    meta_warnings = meta.get("warnings") or []
    if meta_warnings and not meta_notes:
        meta_notes = list(meta_warnings)
    meta["notes"] = list(dict.fromkeys(meta_notes + meta_warnings))
    meta.pop("warnings", None)

    if errors and raise_on_error:
        raise ValueError("; ".join(errors))
    return errors


def compute_cache_key(snapshot: Dict[str, Any]) -> str:
    holdings = snapshot.get("holdings") or []
    holdings_list: List[List[Any]] = []
    for h in holdings:
        if not isinstance(h, dict):
            continue
        holdings_list.append(
            [
                h.get("symbol"),
                h.get("shares"),
                h.get("last_price"),
                h.get("market_value"),
            ]
        )
    holdings_hash = hashlib.sha1(json.dumps(holdings_list, sort_keys=True).encode()).hexdigest()
    meta = snapshot.get("meta") or {}
    last_sync = snapshot.get("last_transaction_sync_at") or meta.get("last_transaction_sync_at") or ""
    combined = f"{last_sync}|{holdings_hash}"
    final_hash = hashlib.sha1(combined.encode()).hexdigest()
    plaid = snapshot.get("plaid_account_id", meta.get("plaid_account_id", "na"))
    as_of = snapshot.get("as_of", "na")
    return f"snapshot-{plaid}-{as_of}-{final_hash}"


def _cache_path(key: str, cache_dir: Optional[str] = None) -> str:
    base = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(base, exist_ok=True)
    safe_key = key.replace(os.sep, "_")
    return os.path.join(base, f"{safe_key}.json")


def cache_snapshot(key: str, snapshot: Dict[str, Any], ttl: int = 3600, cache_dir: Optional[str] = None) -> None:
    path = _cache_path(key, cache_dir)
    payload = {"key": key, "expires_at": time.time() + ttl, "data": snapshot}
    tmp: Optional[NamedTemporaryFile] = None
    try:
        tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path))
        json.dump(payload, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if tmp and not tmp.closed:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def load_snapshot(key: str, cache_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = _cache_path(key, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("expires_at", 0) < time.time():
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    snap = normalize_snapshot(copy.deepcopy(data))
    snap_meta = _ensure_meta(snap.get("meta"))
    snap["meta"] = snap_meta
    snap_meta["served_from"] = "cache_db"
    snap_meta["cache_origin"] = "mira.snapshot_cache"
    now_ts = time.time()
    snap_meta["cache_age_seconds"] = max(0, int(now_ts - payload.get("expires_at", now_ts)))
    _clamp_as_of_for_cache(snap)
    _set_cached_flag(snap)
    validate_snapshot(snap, raise_on_error=True)
    # Validation warnings (non-fatal): appended to meta.notes
    warns = validate_snapshot(snap)
    if warns:
        snap.setdefault('meta', {}).setdefault('notes', []).extend([f'validation_warning: {w}' for w in warns])

    return snap


def build_and_cache_snapshot(
    raw: Dict[str, Any],
    key: Optional[str] = None,
    ttl: int = 3600,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snap = normalize_snapshot(raw)
    validate_snapshot(snap, raise_on_error=True)
    snap_meta = snap.get("meta", {})
    snap_meta["served_from"] = snap_meta.get("served_from") or "recomputed"
    snap_meta.setdefault("cache_control", {"no_store": False, "revalidate": "when-stale"})
    _set_cached_flag(snap)
    cache_key = key or compute_cache_key(snap)
    cache_snapshot(cache_key, snap, ttl=ttl, cache_dir=cache_dir)
    # Validation warnings (non-fatal): appended to meta.notes
    warns = validate_snapshot(snap)
    if warns:
        snap.setdefault('meta', {}).setdefault('notes', []).extend([f'validation_warning: {w}' for w in warns])

    return snap


RAW_INPUT_JSON = r"""
{
  "as_of": "2025-12-18",
  "count": 12,
  "holdings": [
    {
      "symbol": "AMLP",
      "shares": 10.021,
      "cost_basis": 471.71,
      "avg_cost": 47.073,
      "trades": 13,
      "last_price": 46.82,
      "market_value": 469.173,
      "unrealized_pnl": -2.537,
      "unrealized_pct": -0.538,
      "weight_pct": 2.976,
      "forward_12m_dividend": 39.38,
      "last_ex_date": "2025-11-12",
      "forward_method": "t12m",
      "projected_monthly_dividend": 3.28,
      "current_yield_pct": 8.393,
      "yield_on_cost_pct": 8.348,
      "ultimate": {
        "name": "Alerian MLP ETF",
        "exchange": "PCX",
        "currency": "USD",
        "category": "Energy Limited Partnership",
        "vol_30d_pct": 12.939,
        "vol_90d_pct": 12.057,
        "beta_3y": 0.567,
        "max_drawdown_1y_pct": -14.267,
        "sortino_1y": 0.362,
        "trailing_12m_div_ps": 3.93,
        "distribution_frequency": "quarterly",
        "next_ex_date_est": "2026-02-10",
        "twr_1m_pct": 0.515,
        "twr_3m_pct": 0.037,
        "twr_6m_pct": 0.154,
        "twr_12m_pct": 3.334,
        "top_holdings": [
          {
            "symbol": "MPLX",
            "name": "MPLX LP Partnership Units",
            "weight_pct": 13.44
          },
          {
            "symbol": "SUN",
            "name": "Sunoco LP",
            "weight_pct": 12.918
          },
          {
            "symbol": "WES",
            "name": "Western Midstream Partners LP",
            "weight_pct": 12.896
          },
          {
            "symbol": "EPD",
            "name": "Enterprise Products Partners LP",
            "weight_pct": 12.888
          },
          {
            "symbol": "PAA",
            "name": "Plains All American Pipeline LP",
            "weight_pct": 12.626
          },
          {
            "symbol": "ET",
            "name": "Energy Transfer LP",
            "weight_pct": 12
          },
          {
            "symbol": "HESM",
            "name": "Hess Midstream LP Class A",
            "weight_pct": 9.294
          },
          {
            "symbol": "CQP",
            "name": "Cheniere Energy Partners LP",
            "weight_pct": 4.677
          },
          {
            "symbol": "USAC",
            "name": "USA Compression Partners LP",
            "weight_pct": 4.12
          },
          {
            "symbol": "GEL",
            "name": "Genesis Energy LP",
            "weight_pct": 3.545
          }
        ],
        "sector_weights": {
          "realestate": 0,
          "consumer_cyclical": 0,
          "basic_materials": 0,
          "consumer_defensive": 0,
          "technology": 0,
          "communication_services": 0,
          "financial_services": 0,
          "utilities": 2.54,
          "industrials": 0,
          "energy": 97.46,
          "healthcare": 0
        }
      },
      "dividends_30d": 0,
      "dividends_qtd": 4.52,
      "dividends_ytd": 4.52
    },
    {
      "symbol": "ARCC",
      "shares": 70.125,
      "cost_basis": 1414.59,
      "avg_cost": 20.172,
      "trades": 14,
      "last_price": 20.14,
      "market_value": 1412.319,
      "unrealized_pnl": -2.271,
      "unrealized_pct": -0.161,
      "weight_pct": 8.957,
      "forward_12m_dividend": 134.64,
      "last_ex_date": "2025-12-15",
      "forward_method": "t12m",
      "projected_monthly_dividend": 11.22,
      "current_yield_pct": 9.533,
      "yield_on_cost_pct": 9.518,
      "dividends_30d": 0,
      "dividends_qtd": 0,
      "dividends_ytd": 0
    },
    {
      "symbol": "FSK",
      "shares": 115.181,
      "cost_basis": 1776.63,
      "avg_cost": 15.425,
      "trades": 13,
      "last_price": 14.97,
      "market_value": 1724.264,
      "unrealized_pnl": -52.366,
      "unrealized_pct": -2.947,
      "weight_pct": 10.936,
      "forward_12m_dividend": 322.51,
      "last_ex_date": "2025-12-03",
      "forward_method": "t12m",
      "projected_monthly_dividend": 26.88,
      "current_yield_pct": 18.704,
      "yield_on_cost_pct": 18.153,
      "dividends_30d": 40.41,
      "dividends_qtd": 40.41,
      "dividends_ytd": 40.41
    },
    {
      "symbol": "JEPI",
      "shares": 38.379,
      "cost_basis": 2179.39,
      "avg_cost": 56.787,
      "trades": 12,
      "last_price": 57.41,
      "market_value": 2203.317,
      "unrealized_pnl": 23.927,
      "unrealized_pct": 1.098,
      "weight_pct": 13.974,
      "forward_12m_dividend": 179.96,
      "last_ex_date": "2025-12-01",
      "forward_method": "t12m",
      "projected_monthly_dividend": 15,
      "current_yield_pct": 8.168,
      "yield_on_cost_pct": 8.257,
      "ultimate": {
        "name": "JPMorgan Equity Premium Income ETF",
        "exchange": "PCX",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 9.108,
        "vol_90d_pct": 7.626,
        "beta_3y": 0.59,
        "max_drawdown_1y_pct": -13.258,
        "sortino_1y": 0.556,
        "trailing_12m_div_ps": 4.689,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-31",
        "twr_1m_pct": 2.647,
        "twr_3m_pct": 2.891,
        "twr_6m_pct": 7.046,
        "twr_12m_pct": 5.245,
        "top_holdings": [
          {
            "symbol": "GOOGL",
            "name": "Alphabet Inc Class A",
            "weight_pct": 1.863
          },
          {
            "symbol": "JNJ",
            "name": "Johnson & Johnson",
            "weight_pct": 1.644
          },
          {
            "symbol": "ABBV",
            "name": "AbbVie Inc",
            "weight_pct": 1.618
          },
          {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "weight_pct": 1.56
          },
          {
            "symbol": "ROST",
            "name": "Ross Stores Inc",
            "weight_pct": 1.541
          },
          {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "weight_pct": 1.515
          },
          {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "weight_pct": 1.514
          },
          {
            "symbol": "NEE",
            "name": "NextEra Energy Inc",
            "weight_pct": 1.512
          },
          {
            "symbol": "ADI",
            "name": "Analog Devices Inc",
            "weight_pct": 1.501
          },
          {
            "symbol": "MA",
            "name": "Mastercard Inc Class A",
            "weight_pct": 1.479
          }
        ],
        "sector_weights": {
          "realestate": 3.3,
          "consumer_cyclical": 11.84,
          "basic_materials": 2.03,
          "consumer_defensive": 8.2,
          "technology": 19.32,
          "communication_services": 7.19,
          "financial_services": 13.03,
          "utilities": 5.17,
          "industrials": 13.11,
          "energy": 2.14,
          "healthcare": 14.66
        }
      },
      "dividends_30d": 12.56,
      "dividends_qtd": 17.27,
      "dividends_ytd": 17.27
    },
    {
      "symbol": "JEPQ",
      "shares": 38.06,
      "cost_basis": 2209.99,
      "avg_cost": 58.066,
      "trades": 15,
      "last_price": 57.91,
      "market_value": 2204.052,
      "unrealized_pnl": -5.938,
      "unrealized_pct": -0.269,
      "weight_pct": 13.979,
      "forward_12m_dividend": 228.47,
      "last_ex_date": "2025-12-01",
      "forward_method": "t12m",
      "projected_monthly_dividend": 19.04,
      "current_yield_pct": 10.366,
      "yield_on_cost_pct": 10.338,
      "ultimate": {
        "name": "JPMorgan Nasdaq Equity Premium Income ETF",
        "exchange": "NGM",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 16.422,
        "vol_90d_pct": 12.361,
        "beta_3y": 0.923,
        "max_drawdown_1y_pct": -20.07,
        "sortino_1y": 0.806,
        "trailing_12m_div_ps": 6.003,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-31",
        "twr_1m_pct": 2.976,
        "twr_3m_pct": 3.91,
        "twr_6m_pct": 15.468,
        "twr_12m_pct": 11.239,
        "top_holdings": [
          {
            "symbol": "NVDA",
            "name": "NVIDIA Corp",
            "weight_pct": 7.888
          },
          {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "weight_pct": 7.448
          },
          {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "weight_pct": 6.561
          },
          {
            "symbol": "GOOG",
            "name": "Alphabet Inc Class C",
            "weight_pct": 6.004
          },
          {
            "symbol": "AVGO",
            "name": "Broadcom Inc",
            "weight_pct": 5.269
          },
          {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "weight_pct": 4.582
          },
          {
            "symbol": "META",
            "name": "Meta Platforms Inc Class A",
            "weight_pct": 2.765
          },
          {
            "symbol": "TSLA",
            "name": "Tesla Inc",
            "weight_pct": 2.643
          },
          {
            "symbol": "NFLX",
            "name": "Netflix Inc",
            "weight_pct": 2.213
          },
          {
            "symbol": "AMD",
            "name": "Advanced Micro Devices Inc",
            "weight_pct": 1.706
          }
        ],
        "sector_weights": {
          "realestate": 0.22,
          "consumer_cyclical": 13.6,
          "basic_materials": 0.84,
          "consumer_defensive": 4.21,
          "technology": 54.71,
          "communication_services": 16.41,
          "financial_services": 0.51,
          "utilities": 1.29,
          "industrials": 2.95,
          "energy": 0.29,
          "healthcare": 4.96
        }
      },
      "dividends_30d": 18.83,
      "dividends_qtd": 18.83,
      "dividends_ytd": 18.83
    },
    {
      "symbol": "NLY",
      "shares": 28.142,
      "cost_basis": 616.7,
      "avg_cost": 21.914,
      "trades": 12,
      "last_price": 22.6,
      "market_value": 636.004,
      "unrealized_pnl": 19.304,
      "unrealized_pct": 3.13,
      "weight_pct": 4.034,
      "forward_12m_dividend": 77.39,
      "last_ex_date": "2025-09-30",
      "forward_method": "t12m",
      "projected_monthly_dividend": 6.45,
      "current_yield_pct": 12.168,
      "yield_on_cost_pct": 12.549,
      "dividends_30d": 0,
      "dividends_qtd": 0,
      "dividends_ytd": 0
    },
    {
      "symbol": "QYLD",
      "shares": 62.79,
      "cost_basis": 1096.06,
      "avg_cost": 17.456,
      "trades": 14,
      "last_price": 17.58,
      "market_value": 1103.853,
      "unrealized_pnl": 7.793,
      "unrealized_pct": 0.711,
      "weight_pct": 7.001,
      "forward_12m_dividend": 138.26,
      "last_ex_date": "2025-11-24",
      "forward_method": "t12m",
      "projected_monthly_dividend": 11.52,
      "current_yield_pct": 12.525,
      "yield_on_cost_pct": 12.614,
      "ultimate": {
        "name": "Global X NASDAQ 100 Covered Call ETF",
        "exchange": "NGM",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 6.934,
        "vol_90d_pct": 7.221,
        "beta_3y": 0.725,
        "max_drawdown_1y_pct": -19.059,
        "sortino_1y": 0.711,
        "trailing_12m_div_ps": 2.202,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-24",
        "twr_1m_pct": 2.526,
        "twr_3m_pct": 5.948,
        "twr_6m_pct": 12.596,
        "twr_12m_pct": 8.67,
        "top_holdings": [
          {
            "symbol": "NVDA",
            "name": "NVIDIA Corp",
            "weight_pct": 9.624
          },
          {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "weight_pct": 9.259
          },
          {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "weight_pct": 8.183
          },
          {
            "symbol": "AVGO",
            "name": "Broadcom Inc",
            "weight_pct": 7.014
          },
          {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "weight_pct": 5.565
          },
          {
            "symbol": "GOOGL",
            "name": "Alphabet Inc Class A",
            "weight_pct": 4.167
          },
          {
            "symbol": "GOOG",
            "name": "Alphabet Inc Class C",
            "weight_pct": 3.889
          },
          {
            "symbol": "TSLA",
            "name": "Tesla Inc",
            "weight_pct": 3.508
          },
          {
            "symbol": "META",
            "name": "Meta Platforms Inc Class A",
            "weight_pct": 3.145
          }
        ],
        "sector_weights": {
          "realestate": 0.15,
          "consumer_cyclical": 12.83,
          "basic_materials": 1,
          "consumer_defensive": 4.51,
          "technology": 54.7,
          "communication_services": 16.72,
          "financial_services": 0.31,
          "utilities": 1.44,
          "industrials": 2.95,
          "energy": 0.49,
          "healthcare": 4.9
        }
      },
      "dividends_30d": 8.94,
      "dividends_qtd": 8.94,
      "dividends_ytd": 8.94
    },
    {
      "symbol": "RYLD",
      "shares": 51.199,
      "cost_basis": 782.99,
      "avg_cost": 15.293,
      "trades": 13,
      "last_price": 15.41,
      "market_value": 788.98,
      "unrealized_pnl": 5.99,
      "unrealized_pct": 0.765,
      "weight_pct": 5.004,
      "forward_12m_dividend": 94.67,
      "last_ex_date": "2025-11-24",
      "forward_method": "t12m",
      "projected_monthly_dividend": 7.89,
      "current_yield_pct": 11.999,
      "yield_on_cost_pct": 12.091,
      "ultimate": {
        "name": "Global X Russell 2000 Covered Call ETF",
        "exchange": "PCX",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 12.747,
        "vol_90d_pct": 10.431,
        "beta_3y": 0.697,
        "max_drawdown_1y_pct": -19.045,
        "sortino_1y": 0.314,
        "trailing_12m_div_ps": 1.849,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-24",
        "twr_1m_pct": 3.427,
        "twr_3m_pct": 4.761,
        "twr_6m_pct": 10.544,
        "twr_12m_pct": 3.576,
        "top_holdings": [
          {
            "symbol": "RSSL",
            "name": "Global X Russell 2000 ETF",
            "weight_pct": 106.58
          }
        ],
        "sector_weights": {
          "realestate": 6.81,
          "consumer_cyclical": 9.5,
          "basic_materials": 4.39,
          "consumer_defensive": 2.5,
          "technology": 15.49,
          "communication_services": 2.5,
          "financial_services": 16.68,
          "utilities": 3.3,
          "industrials": 15.52,
          "energy": 4.66,
          "healthcare": 18.64
        }
      },
      "dividends_30d": 7.68,
      "dividends_qtd": 7.68,
      "dividends_ytd": 7.68
    },
    {
      "symbol": "SPYI",
      "shares": 9.006,
      "cost_basis": 467.14,
      "avg_cost": 51.872,
      "trades": 15,
      "last_price": 52.42,
      "market_value": 472.072,
      "unrealized_pnl": 4.932,
      "unrealized_pct": 1.056,
      "weight_pct": 2.994,
      "forward_12m_dividend": 55.21,
      "last_ex_date": "2025-11-26",
      "forward_method": "t12m",
      "projected_monthly_dividend": 4.6,
      "current_yield_pct": 11.695,
      "yield_on_cost_pct": 11.819,
      "ultimate": {
        "name": "Neos S&P 500(R) High Income ETF",
        "exchange": "BTS",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 11.47,
        "vol_90d_pct": 9.898,
        "beta_3y": 0.788,
        "max_drawdown_1y_pct": -16.475,
        "sortino_1y": 1.033,
        "trailing_12m_div_ps": 6.131,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-26",
        "twr_1m_pct": 2.847,
        "twr_3m_pct": 2.722,
        "twr_6m_pct": 12.233,
        "twr_12m_pct": 12.397,
        "top_holdings": [
          {
            "symbol": "NVDA",
            "name": "NVIDIA Corp",
            "weight_pct": 7.492
          },
          {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "weight_pct": 7.157
          },
          {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "weight_pct": 6.334
          },
          {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "weight_pct": 3.895
          },
          {
            "symbol": "AVGO",
            "name": "Broadcom Inc",
            "weight_pct": 3.268
          },
          {
            "symbol": "GOOGL",
            "name": "Alphabet Inc Class A",
            "weight_pct": 3.198
          },
          {
            "symbol": "GOOG",
            "name": "Alphabet Inc Class C",
            "weight_pct": 2.567
          },
          {
            "symbol": "META",
            "name": "Meta Platforms Inc Class A",
            "weight_pct": 2.441
          },
          {
            "symbol": "TSLA",
            "name": "Tesla Inc",
            "weight_pct": 2.091
          },
          {
            "symbol": "BRK-B",
            "name": "Berkshire Hathaway Inc Class B",
            "weight_pct": 1.64
          }
        ],
        "sector_weights": {
          "realestate": 1.86,
          "consumer_cyclical": 10.22,
          "basic_materials": 1.45,
          "consumer_defensive": 5,
          "technology": 35.35,
          "communication_services": 11.03,
          "financial_services": 12.95,
          "utilities": 2.54,
          "industrials": 7.04,
          "energy": 2.96,
          "healthcare": 9.61
        }
      },
      "dividends_30d": 7.78,
      "dividends_qtd": 7.78,
      "dividends_ytd": 7.78
    },
    {
      "symbol": "SVOL",
      "shares": 144.126,
      "cost_basis": 2529.59,
      "avg_cost": 17.551,
      "trades": 13,
      "last_price": 17.64,
      "market_value": 2542.39,
      "unrealized_pnl": 12.8,
      "unrealized_pct": 0.506,
      "weight_pct": 16.125,
      "forward_12m_dividend": 495.79,
      "last_ex_date": "2025-11-21",
      "forward_method": "t12m",
      "projected_monthly_dividend": 41.32,
      "current_yield_pct": 19.501,
      "yield_on_cost_pct": 19.6,
      "ultimate": {
        "name": "Simplify Volatility Premium ETF",
        "exchange": "PCX",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 18.114,
        "vol_90d_pct": 21.23,
        "beta_3y": 1.236,
        "max_drawdown_1y_pct": -33.502,
        "sortino_1y": -0.056,
        "trailing_12m_div_ps": 3.44,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-21",
        "twr_1m_pct": 5.474,
        "twr_3m_pct": 1.46,
        "twr_6m_pct": 17.412,
        "twr_12m_pct": -2.013,
        "top_holdings": [
          {
            "symbol": "SVOL",
            "name": "Simplify Volatility Premium ETF",
            "weight_pct": 50.652
          },
          {
            "symbol": "SPUC",
            "name": "Simplify US Equity PLUS Upsd Cnvxty ETF",
            "weight_pct": 12.096
          },
          {
            "symbol": "AGGH",
            "name": "Simplify Aggregate Bond ETF",
            "weight_pct": 11.149
          },
          {
            "symbol": "QIS",
            "name": "Simplify Multi-QIS Alternative ETF",
            "weight_pct": 10.538
          },
          {
            "symbol": "NMB",
            "name": "Simplify National Muni Bond ETF",
            "weight_pct": 5.787
          },
          {
            "symbol": "NXTI",
            "name": "Simplify Next Intangible Core Index ETF",
            "weight_pct": 3.908
          },
          {
            "symbol": "XV",
            "name": "Simplify Target 15 Distribution ETF",
            "weight_pct": 2.417
          }
        ],
        "sector_weights": {
          "realestate": 1.49,
          "consumer_cyclical": 9.5,
          "basic_materials": 1.39,
          "consumer_defensive": 5.68,
          "technology": 38.31,
          "communication_services": 9.52,
          "financial_services": 12.18,
          "utilities": 1.98,
          "industrials": 7.47,
          "energy": 3.08,
          "healthcare": 9.39
        }
      },
      "dividends_30d": 12.85,
      "dividends_qtd": 12.85,
      "dividends_ytd": 12.85
    },
    {
      "symbol": "TRIN",
      "shares": 114.493,
      "cost_basis": 1690.82,
      "avg_cost": 14.768,
      "trades": 10,
      "last_price": 15.18,
      "market_value": 1737.997,
      "unrealized_pnl": 47.177,
      "unrealized_pct": 2.79,
      "weight_pct": 11.023,
      "forward_12m_dividend": 233.56,
      "last_ex_date": "2025-09-30",
      "forward_method": "t12m",
      "projected_monthly_dividend": 19.46,
      "current_yield_pct": 13.438,
      "yield_on_cost_pct": 13.813,
      "dividends_30d": 0,
      "dividends_qtd": 0,
      "dividends_ytd": 0
    },
    {
      "symbol": "XYLD",
      "shares": 11.66,
      "cost_basis": 468.23,
      "avg_cost": 40.157,
      "trades": 14,
      "last_price": 40.53,
      "market_value": 472.575,
      "unrealized_pnl": 4.345,
      "unrealized_pct": 0.928,
      "weight_pct": 2.997,
      "forward_12m_dividend": 59.89,
      "last_ex_date": "2025-11-24",
      "forward_method": "t12m",
      "projected_monthly_dividend": 4.99,
      "current_yield_pct": 12.673,
      "yield_on_cost_pct": 12.791,
      "ultimate": {
        "name": "Global X S&P 500 Covered Call ETF",
        "exchange": "PCX",
        "currency": "USD",
        "category": "Derivative Income",
        "vol_30d_pct": 4.855,
        "vol_90d_pct": 5.513,
        "beta_3y": 0.581,
        "max_drawdown_1y_pct": -15.528,
        "sortino_1y": 0.859,
        "trailing_12m_div_ps": 5.136,
        "distribution_frequency": "monthly",
        "next_ex_date_est": "2025-12-24",
        "twr_1m_pct": 2.835,
        "twr_3m_pct": 5.79,
        "twr_6m_pct": 10.337,
        "twr_12m_pct": 8.718,
        "top_holdings": [
          {
            "symbol": "NVDA",
            "name": "NVIDIA Corp",
            "weight_pct": 7.691
          },
          {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "weight_pct": 7.37
          },
          {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "weight_pct": 6.513
          },
          {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "weight_pct": 4.031
          },
          {
            "symbol": "AVGO",
            "name": "Broadcom Inc",
            "weight_pct": 3.375
          },
          {
            "symbol": "GOOGL",
            "name": "Alphabet Inc Class A",
            "weight_pct": 3.317
          },
          {
            "symbol": "GOOG",
            "name": "Alphabet Inc Class C",
            "weight_pct": 2.662
          },
          {
            "symbol": "META",
            "name": "Meta Platforms Inc Class A",
            "weight_pct": 2.503
          },
          {
            "symbol": "TSLA",
            "name": "Tesla Inc",
            "weight_pct": 2.15
          }
        ],
        "sector_weights": {
          "realestate": 1.86,
          "consumer_cyclical": 10.41,
          "basic_materials": 1.49,
          "consumer_defensive": 4.85,
          "technology": 35.33,
          "communication_services": 11,
          "financial_services": 12.77,
          "utilities": 2.33,
          "industrials": 7.35,
          "energy": 2.82,
          "healthcare": 9.79
        }
      },
      "dividends_30d": 3.83,
      "dividends_qtd": 3.83,
      "dividends_ytd": 3.83
    }
  ],
  "plaid_account_id": 317631,
  "total_market_value": 15766.996,
  "prices_as_of": "2025-12-18",
  "missing_prices": [],
  "margin_loan_balance": 7155.74,
  "margin_to_portfolio_pct": 45.384,
  "totals": {
    "cost_basis": 15703.84,
    "market_value": 15766.996,
    "unrealized_pnl": 63.156,
    "unrealized_pct": 0.402,
    "margin_loan_balance": 7155.74,
    "margin_to_portfolio_pct": 45.384
  },
  "income": {
    "forward_12m_total": 2059.73,
    "projected_monthly_income": 171.644,
    "portfolio_current_yield_pct": 13.064,
    "portfolio_yield_on_cost_pct": 13.116
  },
  "portfolio_rollups": {
    "performance": {},
    "risk": {},
    "benchmark": "^GSPC",
    "meta": {
      "method": "approx-fallback",
      "note": "Restored after temporary omission."
    }
  },
  "dividends_upcoming": {
    "window": {
      "mode": "exdate",
      "start": "2025-12-19",
      "end": "2025-12-31"
    },
    "projected": 104.36,
    "events": [
      {
        "symbol": "SVOL",
        "ex_date_est": "2025-12-21",
        "amount_est": 41.32
      },
      {
        "symbol": "QYLD",
        "ex_date_est": "2025-12-24",
        "amount_est": 11.52
      },
      {
        "symbol": "RYLD",
        "ex_date_est": "2025-12-24",
        "amount_est": 7.89
      },
      {
        "symbol": "XYLD",
        "ex_date_est": "2025-12-24",
        "amount_est": 4.99
      },
      {
        "symbol": "SPYI",
        "ex_date_est": "2025-12-26",
        "amount_est": 4.6
      },
      {
        "symbol": "JEPI",
        "ex_date_est": "2025-12-31",
        "amount_est": 15
      },
      {
        "symbol": "JEPQ",
        "ex_date_est": "2025-12-31",
        "amount_est": 19.04
      }
    ],
    "meta": {
      "sum_of_events": 104.36,
      "matches_projected": true
    }
  },
  "dividends_upcoming_meta": {
    "sum_of_events": 104.36,
    "matches_projected": true
  },
  "dividends": {
    "realized_mtd": {
      "start": "2025-12-01",
      "end": "2025-12-18",
      "total_dividends": 95.29,
      "by_date": {
        "2025-12-02": 20.45,
        "2025-12-03": 31.39,
        "2025-12-15": 3.04,
        "2025-12-17": 40.41
      },
      "by_month": {
        "2025-12": 95.29
      },
      "by_symbol": {
        "RYLD": {
          "amount": 7.68,
          "status": "active"
        },
        "XYLD": {
          "amount": 3.83,
          "status": "active"
        },
        "QYLD": {
          "amount": 8.94,
          "status": "active"
        },
        "JEPQ": {
          "amount": 18.83,
          "status": "active"
        },
        "JEPI": {
          "amount": 12.56,
          "status": "active"
        },
        "O": {
          "amount": 3.04,
          "status": "inactive"
        },
        "FSK": {
          "amount": 40.41,
          "status": "active"
        }
      }
    },
    "realized_mtd_detail": {
      "RYLD": {
        "amount": 7.68,
        "status": "active"
      },
      "XYLD": {
        "amount": 3.83,
        "status": "active"
      },
      "QYLD": {
        "amount": 8.94,
        "status": "active"
      },
      "JEPQ": {
        "amount": 18.83,
        "status": "active"
      },
      "JEPI": {
        "amount": 12.56,
        "status": "active"
      },
      "O": {
        "amount": 3.04,
        "status": "inactive"
      },
      "FSK": {
        "amount": 40.41,
        "status": "active"
      }
    },
    "projected_vs_received": {
      "window": {
        "label": "month_to_date",
        "start": "2025-12-01",
        "end": "2025-12-18"
      },
      "projected": 171.644,
      "received": 95.29,
      "difference": 76.354,
      "pct_of_projection": 55.52,
      "alt": {
        "mode": "paydate",
        "window": {
          "start": "2025-12-01",
          "end": "2025-12-18"
        },
        "projected": 0,
        "expected_events": []
      }
    },
    "windows": {
      "30d": {
        "start": "2025-11-18",
        "end": "2025-12-18",
        "total_dividends": 115.92,
        "by_date": {
          "2025-11-28": 20.63,
          "2025-12-02": 20.45,
          "2025-12-03": 31.39,
          "2025-12-15": 3.04,
          "2025-12-17": 40.41
        },
        "by_month": {
          "2025-11": 20.63,
          "2025-12": 95.29
        },
        "by_symbol": {
          "SVOL": {
            "amount": 12.85,
            "status": "active"
          },
          "SPYI": {
            "amount": 7.78,
            "status": "active"
          },
          "RYLD": {
            "amount": 7.68,
            "status": "active"
          },
          "XYLD": {
            "amount": 3.83,
            "status": "active"
          },
          "QYLD": {
            "amount": 8.94,
            "status": "active"
          },
          "JEPQ": {
            "amount": 18.83,
            "status": "active"
          },
          "JEPI": {
            "amount": 12.56,
            "status": "active"
          },
          "O": {
            "amount": 3.04,
            "status": "inactive"
          },
          "FSK": {
            "amount": 40.41,
            "status": "active"
          }
        }
      },
      "qtd": {
        "start": "2025-10-01",
        "end": "2025-12-18",
        "total_dividends": 125.53,
        "by_date": {
          "2025-11-05": 4.71,
          "2025-11-14": 0.38,
          "2025-11-17": 4.52,
          "2025-11-28": 20.63,
          "2025-12-02": 20.45,
          "2025-12-03": 31.39,
          "2025-12-15": 3.04,
          "2025-12-17": 40.41
        },
        "by_month": {
          "2025-11": 30.24,
          "2025-12": 95.29
        },
        "by_symbol": {
          "JEPI": {
            "amount": 17.27,
            "status": "active"
          },
          "MAIN": {
            "amount": 0.38,
            "status": "inactive"
          },
          "AMLP": {
            "amount": 4.52,
            "status": "active"
          },
          "SVOL": {
            "amount": 12.85,
            "status": "active"
          },
          "SPYI": {
            "amount": 7.78,
            "status": "active"
          },
          "RYLD": {
            "amount": 7.68,
            "status": "active"
          },
          "XYLD": {
            "amount": 3.83,
            "status": "active"
          },
          "QYLD": {
            "amount": 8.94,
            "status": "active"
          },
          "JEPQ": {
            "amount": 18.83,
            "status": "active"
          },
          "O": {
            "amount": 3.04,
            "status": "inactive"
          },
          "FSK": {
            "amount": 40.41,
            "status": "active"
          }
        }
      },
      "ytd": {
        "start": "2025-01-01",
        "end": "2025-12-18",
        "total_dividends": 125.53,
        "by_date": {
          "2025-11-05": 4.71,
          "2025-11-14": 0.38,
          "2025-11-17": 4.52,
          "2025-11-28": 20.63,
          "2025-12-02": 20.45,
          "2025-12-03": 31.39,
          "2025-12-15": 3.04,
          "2025-12-17": 40.41
        },
        "by_month": {
          "2025-11": 30.24,
          "2025-12": 95.29
        },
        "by_symbol": {
          "JEPI": {
            "amount": 17.27,
            "status": "active"
          },
          "MAIN": {
            "amount": 0.38,
            "status": "inactive"
          },
          "AMLP": {
            "amount": 4.52,
            "status": "active"
          },
          "SVOL": {
            "amount": 12.85,
            "status": "active"
          },
          "SPYI": {
            "amount": 7.78,
            "status": "active"
          },
          "RYLD": {
            "amount": 7.68,
            "status": "active"
          },
          "XYLD": {
            "amount": 3.83,
            "status": "active"
          },
          "QYLD": {
            "amount": 8.94,
            "status": "active"
          },
          "JEPQ": {
            "amount": 18.83,
            "status": "active"
          },
          "O": {
            "amount": 3.04,
            "status": "inactive"
          },
          "FSK": {
            "amount": 40.41,
            "status": "active"
          }
        }
      }
    }
  },
  "meta": {
    "served_from": "recomputed",
    "snapshot_created_at": "2025-12-19T06:06:31.247866",
    "snapshot_age_days": 0,
    "last_transaction_sync_at": "2025-12-19T01:23:45.785652-08:00",
    "cache_control": {
      "no_store": false,
      "revalidate": "when-stale"
    },
    "cache": {
      "yf_dividends": {
        "ttl_seconds": 3600,
        "bypassed": true
      },
      "pricing": {
        "ttl_seconds": 3600,
        "bypassed": true
      }
    }
  },
  "goal_progress": {
    "target_monthly": 2000,
    "current_projected_monthly": 171.64,
    "shortfall": 1828.36,
    "progress_pct": 8.58,
    "portfolio_yield_pct": 13.064,
    "months_to_goal": null,
    "estimated_goal_date": null,
    "assumptions": "based_on_last_3_full_months_realized",
    "growth_window_months": 3,
    "required_portfolio_value_at_goal": 183710.96,
    "additional_investment_needed": 167943.97
  },
  "goal_progress_net": {
    "target_monthly": 2000,
    "current_projected_monthly_net": 146.894,
    "progress_pct": 7.34,
    "portfolio_yield_pct": 13.064,
    "assumptions": "same_yield_structure; loan unchanged",
    "required_portfolio_value_at_goal_now": 185984.38,
    "additional_investment_needed_now": 170217.39,
    "monthly_interest_now": 24.75,
    "future_rate_sensitivity": {
      "apr_future_pct": 5.65,
      "monthly_interest_future": 33.69,
      "required_portfolio_value_at_goal_future": 186805.57,
      "additional_investment_needed_future": 171038.58,
      "progress_pct_future": 6.9
    }
  },
  "margin_guidance": {
    "selected_mode": "balanced",
    "rates": {
      "apr_current_pct": 4.15,
      "apr_future_pct": 5.65,
      "apr_future_date": "2026-11-01"
    },
    "modes": [
      {
        "mode": "conservative",
        "action": "repay",
        "amount": 5263.7,
        "ltv_now_pct": 45.384,
        "ltv_stress_pct": 75.64,
        "monthly_interest_now": 24.75,
        "income_interest_coverage_now": 6.935,
        "monthly_interest_future": 33.69,
        "income_interest_coverage_future": 5.095,
        "after_action": {
          "new_loan_balance": 1892.04,
          "ltv_after_action_pct": 12,
          "monthly_interest_now": 6.54,
          "income_interest_coverage_now": 26.245
        },
        "constraints": {
          "max_margin_pct": 20,
          "stress_drawdown_pct": 40,
          "min_income_coverage": 2
        }
      },
      {
        "mode": "balanced",
        "action": "repay",
        "amount": 4396.52,
        "ltv_now_pct": 45.384,
        "ltv_stress_pct": 64.835,
        "monthly_interest_now": 24.75,
        "income_interest_coverage_now": 6.935,
        "monthly_interest_future": 33.69,
        "income_interest_coverage_future": 5.095,
        "after_action": {
          "new_loan_balance": 2759.22,
          "ltv_after_action_pct": 17.5,
          "monthly_interest_now": 9.54,
          "income_interest_coverage_now": 17.992
        },
        "constraints": {
          "max_margin_pct": 25,
          "stress_drawdown_pct": 30,
          "min_income_coverage": 1.5
        }
      },
      {
        "mode": "aggressive",
        "action": "repay",
        "amount": 3371.66,
        "ltv_now_pct": 45.384,
        "ltv_stress_pct": 56.73,
        "monthly_interest_now": 24.75,
        "income_interest_coverage_now": 6.935,
        "monthly_interest_future": 33.69,
        "income_interest_coverage_future": 5.095,
        "after_action": {
          "new_loan_balance": 3784.08,
          "ltv_after_action_pct": 24,
          "monthly_interest_now": 13.09,
          "income_interest_coverage_now": 13.113
        },
        "constraints": {
          "max_margin_pct": 30,
          "stress_drawdown_pct": 20,
          "min_income_coverage": 1.2
        }
      }
    ]
  },
  "cached": false
}
"""


def _by_symbol_is_canonical(dividends: Dict[str, Any]) -> bool:
    def check_map(m: Dict[str, Any]) -> bool:
        if not isinstance(m, dict):
            return False
        for v in m.values():
            if not isinstance(v, dict):
                return False
            if "amount" not in v or "status" not in v:
                return False
        return True

    realized = dividends.get("realized_mtd", {})
    windows = dividends.get("windows", {})
    if not check_map(realized.get("by_symbol", {})):
        return False
    for label in ("30d", "qtd", "ytd"):
        if not check_map((windows.get(label) or {}).get("by_symbol", {})):
            return False
    return True


def _print_self_check(normalized: Dict[str, Any], cache_dir: Optional[str] = None) -> None:
    key = compute_cache_key(normalized)
    cached = load_snapshot(key, cache_dir=cache_dir)
    checks: List[Tuple[str, bool]] = []
    checks.append(("count === 12", normalized.get("count") == 12))
    checks.append(("totals.market_value === 15766.996", normalized.get("totals", {}).get("market_value") == 15766.996))
    checks.append(
        ("income.forward_12m_total === 2059.73", abs(normalized.get("income", {}).get("forward_12m_total", 0) - 2059.73) <= 0.01)
    )
    checks.append(
        ("income.projected_monthly_income === 171.644", abs(normalized.get("income", {}).get("projected_monthly_income", 0) - 171.644) <= 0.01)
    )
    checks.append(("dividends.realized_mtd.total_dividends === 95.29", normalized.get("dividends", {}).get("realized_mtd", {}).get("total_dividends") == 95.29))
    checks.append(
        (
            'dividends.windows["30d"].total_dividends === 115.92',
            ((normalized.get("dividends", {}).get("windows", {}) or {}).get("30d", {}) or {}).get("total_dividends") == 115.92,
        )
    )
    du = normalized.get("dividends_upcoming", {})
    checks.append(
        (
            "dividends_upcoming.meta.sum_of_events === 104.36",
            abs((du.get("meta") or {}).get("sum_of_events", 0) - 104.36) < 1e-6,
        )
    )
    checks.append(
        (
            "dividends_upcoming.meta.matches_projected === true",
            (du.get("meta") or {}).get("matches_projected") is True,
        )
    )
    checks.append(("All by_symbol entries canonical", _by_symbol_is_canonical(normalized.get("dividends", {}))))
    checks.append(("cache round-trip served_from cache", cached is not None and (cached.get("meta") or {}).get("served_from") == "cache"))
    checks.append(("cache round-trip cached flag true", cached is not None and cached.get("cached") is True))
    for label, ok in checks:
        prefix = "OK" if ok else "FAIL"
        print(f"{prefix} {label}")


if __name__ == "__main__":
    raw = json.loads(RAW_INPUT_JSON)
    normalized_snapshot = build_and_cache_snapshot(raw, ttl=3600, cache_dir=DEFAULT_CACHE_DIR)
    print(json.dumps(normalized_snapshot, indent=2))
    _print_self_check(normalized_snapshot, cache_dir=DEFAULT_CACHE_DIR)


def drop_nulls(obj: Any) -> Any:
    """Recursively drop keys with value None from dicts.

    Lists are preserved; their elements are cleaned recursively.
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if v is None:
                continue
            cleaned = drop_nulls(v)
            # if cleaned is an empty dict after dropping, keep it only if original wasn't None and key is meaningful
            if cleaned is None:
                continue
            new[k] = cleaned
        return new
    if isinstance(obj, list):
        return [drop_nulls(v) for v in obj if v is not None]
    return obj


def count_nulls(obj: Any) -> int:
    if obj is None:
        return 1
    if isinstance(obj, dict):
        return sum(count_nulls(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_nulls(v) for v in obj)
    return 0
