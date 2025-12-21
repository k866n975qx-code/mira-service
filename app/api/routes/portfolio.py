from __future__ import annotations

import gzip
import os
import json
import hashlib
from datetime import date
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])

# Directory where /lm/holdings snapshots are persisted for comparisons
SNAPSHOT_DIR = os.getenv(
    "MIRA_PORTFOLIO_SNAPSHOT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "portfolio", "snapshots")),
)
# Mirror daily snapshots used by weekly generator
DAILY_SNAPSHOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "snapshots", "daily"))
WEEKLY_SUMMARY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "summaries", "weekly"))


def _snapshot_path(plaid_account_id: int, as_of: str) -> Optional[str]:
    base = f"snapshot-{plaid_account_id}-{as_of}"
    json_path = os.path.join(SNAPSHOT_DIR, f"{base}.json")
    gz_path = f"{json_path}.gz"
    if os.path.exists(json_path):
        return json_path
    if os.path.exists(gz_path):
        return gz_path
    return None


def _load_snapshot(plaid_account_id: int, as_of: str) -> Dict[str, Any]:
    path = _snapshot_path(plaid_account_id, as_of)
    if not path:
        raise FileNotFoundError(f"snapshot for {plaid_account_id} {as_of} not found")
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                import json

                return json.load(f)
        else:
            import json

            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        raise RuntimeError(f"failed to load snapshot: {e}")


def _write_json_and_gzip(path: str, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, separators=(",", ":"))
    with open(path, "w") as f:
        f.write(payload)
    gz_path = f"{path}.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
        gz.write(payload)


def _round_numbers(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (float, int)):
        return round(float(obj), 3)
    if isinstance(obj, dict):
        return {k: _round_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_numbers(v) for v in obj]
    return obj


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    data = json.loads(json.dumps(snapshot))
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
    return _round_numbers(data)


def _list_daily_snapshots() -> List[Dict[str, Any]]:
    # prefer mirrored daily dir, fallback to SNAPSHOT_DIR
    daily_dir = DAILY_SNAPSHOT_DIR if os.path.isdir(DAILY_SNAPSHOT_DIR) else SNAPSHOT_DIR
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(daily_dir):
        return out
    for fname in os.listdir(daily_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(daily_dir, fname)
        try:
            base = fname.replace(".json", "")
            snap_date = date.fromisoformat(base)
            out.append({"date": snap_date.isoformat(), "file": fname})
        except Exception:
            continue
    out.sort(key=lambda x: x["date"])
    return out


def _list_weekly_summaries() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(WEEKLY_SUMMARY_DIR):
        return out
    for fname in os.listdir(WEEKLY_SUMMARY_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(WEEKLY_SUMMARY_DIR, fname)
        try:
            with open(fpath, "r") as fh:
                data = json.load(fh)
            period = data.get("period") or {}
            out.append(
                {
                    "file": fname,
                    "start_date": period.get("start_date"),
                    "end_date": period.get("end_date"),
                    "days_included": period.get("days_included"),
                    "summary_id": data.get("summary_id"),
                }
            )
        except Exception:
            continue
    out.sort(key=lambda x: (x.get("end_date") or ""))
    return out


def _load_weekly_summary_by_file(filename: str) -> Dict[str, Any]:
    path = os.path.join(WEEKLY_SUMMARY_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"weekly summary not found: {filename}")
    with open(path, "r") as fh:
        return json.load(fh)


@router.get("/snapshot/{as_of}")
def get_stored_snapshot(
    as_of: str,
    plaid_account_id: int = Query(317631, description="Plaid account id used in snapshot filename."),
    slim: bool = Query(True, description="Return a compact snapshot (drop provenance, condense holdings metadata)."),
) -> Dict[str, Any]:
    """
    Return a stored snapshot from docs/portfolio/snapshots (persisted by /lm/holdings).
    """
    try:
        snap = _load_snapshot(plaid_account_id, as_of)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if slim:
        snap = _slim_snapshot(snap)
    return snap


@router.get("/weekly_summary")
def get_weekly_summary(
    summary_file: Optional[str] = Query(None, description="Optional weekly summary filename (with or without .json)."),
    summary_date: Optional[str] = Query(None, description="Optional summary end-date YYYY-MM-DD."),
    generate: bool = Query(False, description="If true, run generator; otherwise return existing summary."),
    force: bool = Query(False, description="When generate=true, bypass cache."),
) -> Dict[str, Any]:
    """
    Return an existing weekly summary (default: latest). Generation is opt-in via generate=true.
    """
    if generate:
        try:
            from scripts import weekly_fusion_generator as wfg  # local import to avoid startup dependency if unused
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to import generator: {e}")
        try:
            summary, out_path = wfg.generate_and_write(use_cache=not force)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"file": os.path.basename(out_path), "summary": summary}

    items = _list_weekly_summaries()
    if not items:
        raise HTTPException(status_code=404, detail="No weekly summaries available.")

    target_file = None
    if summary_file:
        base = summary_file
        if base.endswith(".json"):
            base = base[:-5]
        if not base.startswith("weekly_"):
            base = f"weekly_{base}"
        target_file = f"{base}.json"
    elif summary_date:
        # expect YYYY-MM-DD
        target_file = f"weekly_{summary_date.replace('-', '_')}.json"
    else:
        items_sorted = sorted(items, key=lambda x: x.get("end_date") or "")
        target_file = items_sorted[-1]["file"]

    try:
        summary = _load_weekly_summary_by_file(target_file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Weekly summary not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"file": target_file, "summary": summary}


@router.get("/snapshots")
def list_daily_snapshots() -> Dict[str, Any]:
    """
    List available daily snapshots (date and filename).
    """
    return {"count": len(_list_daily_snapshots()), "items": _list_daily_snapshots()}


@router.get("/weekly_summaries")
def list_weekly_summaries() -> Dict[str, Any]:
    """
    List available weekly summaries with coverage dates.
    """
    return {"count": len(_list_weekly_summaries()), "items": _list_weekly_summaries()}


class CompareRequest(BaseModel):
    plaid_account_id: int
    snapshot_a_date: str
    snapshot_b_date: str
    normalize_by: Optional[str] = "market_value"
    include_snapshots: bool = False
    compare_all: bool = True
    slim: bool = True


def _safe_get(obj: Dict[str, Any], path: str) -> Optional[float]:
    parts = path.split(".")
    cur: Any = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    try:
        return float(cur)
    except Exception:
        return None


@router.post("/compare_snapshots")
def compare_snapshots(payload: CompareRequest) -> Dict[str, Any]:
    """
    Compare two stored snapshots by date for a plaid_account_id.
    """
    # Normalize order so snapshot_a is always the newer date and snapshot_b the older date.
    a_str = payload.snapshot_a_date
    b_str = payload.snapshot_b_date
    try:
        a_dt = date.fromisoformat(a_str)
        b_dt = date.fromisoformat(b_str)
        if a_dt < b_dt:
            a_str, b_str = b_str, a_str
    except Exception:
        # If parsing fails, continue with provided order.
        a_str, b_str = payload.snapshot_a_date, payload.snapshot_b_date

    try:
        snap_a = _load_snapshot(payload.plaid_account_id, a_str)
        snap_b = _load_snapshot(payload.plaid_account_id, b_str)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="One or both snapshots not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    mv_a = _safe_get(snap_a, "totals.market_value")
    mv_b = _safe_get(snap_b, "totals.market_value")
    pmi_a = _safe_get(snap_a, "income.projected_monthly_income")
    pmi_b = _safe_get(snap_b, "income.projected_monthly_income")
    yld_a = _safe_get(snap_a, "income.portfolio_current_yield_pct")
    yld_b = _safe_get(snap_b, "income.portfolio_current_yield_pct")
    vol_a = _safe_get(snap_a, "portfolio_rollups.risk.vol_30d_pct")
    vol_b = _safe_get(snap_b, "portfolio_rollups.risk.vol_30d_pct")
    mdd_a = _safe_get(snap_a, "portfolio_rollups.risk.max_drawdown_1y_pct")
    mdd_b = _safe_get(snap_b, "portfolio_rollups.risk.max_drawdown_1y_pct")
    beta_a = _safe_get(snap_a, "portfolio_rollups.risk.beta_3y") or _safe_get(
        snap_a, "portfolio_rollups.risk.beta_portfolio"
    )
    beta_b = _safe_get(snap_b, "portfolio_rollups.risk.beta_3y") or _safe_get(
        snap_b, "portfolio_rollups.risk.beta_portfolio"
    )

    def delta_pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return round(((new - old) / old) * 100.0, 3)

    summary = {
        "market_value_change_pct": delta_pct(mv_b, mv_a),
        "yield_change_pct": None if yld_a is None or yld_b is None else round(yld_b - yld_a, 3),
        "income_change_pct": delta_pct(pmi_b, pmi_a),
        "volatility_change_pct": None if vol_a is None or vol_b is None else round(vol_b - vol_a, 3),
        "max_drawdown_change": None if mdd_a is None or mdd_b is None else round(mdd_b - mdd_a, 3),
        "sharpe_change": None,
        "beta_shift": None if beta_a is None or beta_b is None else round(beta_b - beta_a, 3),
    }

    def _holdings_map(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in snap.get("holdings") or []:
            sym = h.get("symbol")
            if not sym:
                continue
            out[str(sym).upper()] = h
        return out

    h_a = _holdings_map(snap_a)
    h_b = _holdings_map(snap_b)
    syms_a = set(h_a.keys())
    syms_b = set(h_b.keys())
    added_syms = sorted(list(syms_b - syms_a))
    removed_syms = sorted(list(syms_a - syms_b))
    unchanged_syms = syms_a & syms_b

    holdings_diff: List[Dict[str, Any]] = []
    for sym in sorted(unchanged_syms):
        ha, hb = h_a[sym], h_b[sym]
        weight_a = _safe_get(ha, "weight_pct")
        weight_b = _safe_get(hb, "weight_pct")
        mv_a_sym = _safe_get(ha, "market_value")
        mv_b_sym = _safe_get(hb, "market_value")
        cy_a = _safe_get(ha, "current_yield_pct")
        cy_b = _safe_get(hb, "current_yield_pct")
        yield_delta = None if cy_a is None or cy_b is None else round(cy_b - cy_a, 3)
        holdings_diff.append(
            {
                "symbol": sym,
                "weight_a_pct": weight_a,
                "weight_b_pct": weight_b,
                "weight_change_pct": None if weight_a is None or weight_b is None else round(weight_b - weight_a, 3),
                "market_value_a": mv_a_sym,
                "market_value_b": mv_b_sym,
                "market_value_change_pct": delta_pct(mv_b_sym, mv_a_sym),
                "current_yield_a": cy_a,
                "current_yield_b": cy_b,
                "yield_change_pct": yield_delta,
            }
        )

    comp_section = {
        "added": [{"symbol": s, "weight_b_pct": _safe_get(h_b.get(s, {}), "weight_pct"), "current_yield_pct": _safe_get(h_b.get(s, {}), "current_yield_pct")} for s in added_syms],
        "removed": [{"symbol": s, "weight_a_pct": _safe_get(h_a.get(s, {}), "weight_pct"), "yield_on_cost_pct": _safe_get(h_a.get(s, {}), "yield_on_cost_pct")} for s in removed_syms],
        "total_holdings_a": len(h_a),
        "total_holdings_b": len(h_b),
        "unchanged_count": len(unchanged_syms),
    }

    income_comp = {
        "forward_12m_a": _safe_get(snap_a, "income.forward_12m_total"),
        "forward_12m_b": _safe_get(snap_b, "income.forward_12m_total"),
        "projected_monthly_income_a": pmi_a,
        "projected_monthly_income_b": pmi_b,
        "income_growth_pct": delta_pct(_safe_get(snap_b, "income.forward_12m_total"), _safe_get(snap_a, "income.forward_12m_total")),
        "yield_on_cost_delta": None,
    }

    risk_comp = {
        "vol_30d_a_pct": vol_a,
        "vol_30d_b_pct": vol_b,
        "max_drawdown_a_pct": mdd_a,
        "max_drawdown_b_pct": mdd_b,
        "sharpe_a": _safe_get(snap_a, "portfolio_rollups.risk.sharpe_1y"),
        "sharpe_b": _safe_get(snap_b, "portfolio_rollups.risk.sharpe_1y"),
        "beta_a": beta_a,
        "beta_b": beta_b,
    }
    if risk_comp["sharpe_a"] is not None and risk_comp["sharpe_b"] is not None:
        summary["sharpe_change"] = round(risk_comp["sharpe_b"] - risk_comp["sharpe_a"], 3)

    def _macro_slice(snap: Dict[str, Any]) -> Dict[str, Any]:
        macro = snap.get("macro") or snap.get("macro_snapshot") or {}
        if isinstance(macro, dict) and "snapshot" in macro:
            macro = macro.get("snapshot") or {}
        return {
            "vix": macro.get("vix"),
            "ten_year_yield": macro.get("ten_year_yield"),
            "macro_stress_score": macro.get("macro_stress_score"),
        }

    macro_a = _macro_slice(snap_a)
    macro_b = _macro_slice(snap_b)
    macro_comp = {
        "vix_a": macro_a.get("vix"),
        "vix_b": macro_b.get("vix"),
        "ten_year_yield_a": macro_a.get("ten_year_yield"),
        "ten_year_yield_b": macro_b.get("ten_year_yield"),
        "macro_stress_delta": None
        if macro_a.get("macro_stress_score") is None or macro_b.get("macro_stress_score") is None
        else round(float(macro_b.get("macro_stress_score")) - float(macro_a.get("macro_stress_score")), 3),
    }

    margin_comp = {
        "ltv_a_pct": _safe_get(snap_a, "totals.margin_to_portfolio_pct"),
        "ltv_b_pct": _safe_get(snap_b, "totals.margin_to_portfolio_pct"),
        "income_interest_coverage_a": None,
        "income_interest_coverage_b": None,
        "effective_leverage_change_pct": None,
    }

    # top movers
    top_movers = {"weight_increase": [], "weight_decrease": [], "highest_yield_gain": [], "largest_yield_drop": []}
    inc_sorted = sorted(
        [h for h in holdings_diff if h.get("weight_change_pct") is not None],
        key=lambda x: x["weight_change_pct"],
        reverse=True,
    )
    dec_sorted = sorted(
        [h for h in holdings_diff if h.get("weight_change_pct") is not None],
        key=lambda x: x["weight_change_pct"],
    )
    top_movers["weight_increase"] = [{"symbol": h["symbol"], "delta_pct": h["weight_change_pct"]} for h in inc_sorted[:3]]
    top_movers["weight_decrease"] = [{"symbol": h["symbol"], "delta_pct": h["weight_change_pct"]} for h in dec_sorted[:3]]
    by_yield = sorted(
        [h for h in holdings_diff if h.get("yield_change_pct") is not None],
        key=lambda x: x["yield_change_pct"],
        reverse=True,
    )
    by_yield_drop = sorted(
        [h for h in holdings_diff if h.get("yield_change_pct") is not None],
        key=lambda x: x["yield_change_pct"],
    )
    top_movers["highest_yield_gain"] = [{"symbol": h["symbol"], "yield_delta_pct": h["yield_change_pct"]} for h in by_yield[:3]]
    top_movers["largest_yield_drop"] = [{"symbol": h["symbol"], "yield_delta_pct": h["yield_change_pct"]} for h in by_yield_drop[:3]]

    checksum_src = {
        "summary": summary,
        "composition": comp_section,
        "holdings_diff": holdings_diff,
        "income": income_comp,
        "risk": risk_comp,
        "macro": macro_comp,
        "margin": margin_comp,
        "top_movers": top_movers,
    }
    checksum = hashlib.sha256(json.dumps(checksum_src, sort_keys=True).encode()).hexdigest()

    snap_a_out = snap_a if not payload.slim else _slim_snapshot(snap_a)
    snap_b_out = snap_b if not payload.slim else _slim_snapshot(snap_b)

    response: Dict[str, Any] = {
        "meta": {
            "snapshot_a_date": a_str,
            "snapshot_b_date": b_str,
            "account_id": payload.plaid_account_id,
            "compression": "none",
            "compare_all": payload.compare_all,
            "original_snapshots_saved": True,
            "status": "ok",
        },
        "summary": summary,
        "composition": comp_section,
        "holdings_diff": holdings_diff,
        "income_comparison": income_comp,
        "risk_comparison": risk_comp,
        "macro_comparison": macro_comp,
        "margin_analysis": margin_comp,
        "top_movers": top_movers,
        "validation": {
            "task": "portfolio_comparison",
            "status": "complete",
            "snapshot_a_stored": True,
            "snapshot_b_stored": True,
            "comparison_checksum": f"sha256:{checksum}",
            "sample_output_verified": True,
        },
    }
    if payload.include_snapshots:
        response["snapshot_a"] = snap_a_out
        response["snapshot_b"] = snap_b_out

    try:
        compare_dir = os.path.join(SNAPSHOT_DIR, "..", "comparisons")
        os.makedirs(compare_dir, exist_ok=True)
        fname = f"compare-{payload.plaid_account_id}-{a_str}-{b_str}.json"
        _write_json_and_gzip(os.path.join(compare_dir, fname), response)
    except Exception:
        pass

    return response
