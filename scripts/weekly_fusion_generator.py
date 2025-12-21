#!/usr/bin/env python3
"""
Weekly Fusion Summary generator.

Reads the 7 most recent daily portfolio snapshots from ./data/snapshots/daily/,
aggregates key metrics, validates against the embedded schema, caches for 24h,
and writes a weekly summary JSON to ./data/summaries/weekly/.
"""

import glob
import json
import os
import time
from datetime import datetime, date

from jsonschema import validate, ValidationError

# ---------------------------- paths & constants ---------------------------- #
DAILY_DIR = os.path.abspath(os.path.join(".", "data", "snapshots", "daily"))
WEEKLY_DIR = os.path.abspath(os.path.join(".", "data", "summaries", "weekly"))
CACHE_DIR = os.path.abspath(os.path.join(".", "cache", "weekly"))
CACHE_TTL_SECONDS = 24 * 60 * 60
MIN_DAYS = 7

SCHEMA = {
    "type": "object",
    "required": [
        "summary_id",
        "period",
        "aggregates",
        "income",
        "holdings_deltas",
        "comparative_deltas",
        "macro_context",
        "end_of_week_portfolio",
        "meta",
    ],
    "properties": {
        "summary_id": {"type": "string"},
        "period": {
            "type": "object",
            "required": ["start_date", "end_date", "days_included"],
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "days_included": {"type": "integer"},
            },
        },
        "aggregates": {
            "type": "object",
            "properties": {
                "market_value": {
                    "type": "object",
                    "properties": {
                        "avg": {"type": "number"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "delta_pct": {"type": "number"},
                    },
                },
                "forward_yield_pct": {
                    "type": "object",
                    "properties": {"avg": {"type": "number"}, "delta": {"type": "number"}},
                },
                "sharpe_ratio": {
                    "type": "object",
                    "properties": {"avg": {"type": "number"}, "delta": {"type": "number"}},
                },
                "beta": {
                    "type": "object",
                    "properties": {"avg": {"type": "number"}, "delta": {"type": "number"}},
                },
                "vol_30d_pct": {"type": "object", "properties": {"avg": {"type": "number"}}},
                "max_drawdown_pct": {"type": "object", "properties": {"avg": {"type": "number"}}},
            },
        },
        "income": {
            "type": "object",
            "properties": {
                "avg_monthly_income": {"type": "number"},
                "total_weekly_income": {"type": "number"},
                "yield_on_cost_pct": {"type": "number"},
                "drip_reinvested": {"type": "number"},
            },
        },
        "holdings_deltas": {
            "type": "object",
            "properties": {
                "positions_added": {"type": "array", "items": {"type": "string"}},
                "positions_removed": {"type": "array", "items": {"type": "string"}},
                "weight_changes": {"type": "object", "patternProperties": {".*": {"type": "number"}}},
                "sector_rotation": {"type": "object", "patternProperties": {".*": {"type": "number"}}},
                "diversification_index": {
                    "type": "object",
                    "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "delta": {"type": "number"}},
                },
                "positions_count": {
                    "type": "object",
                    "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                },
            },
        },
        "comparative_deltas": {
            "type": "object",
            "properties": {
                "avg_market_value_change_pct": {"type": "number"},
                "avg_income_change_pct": {"type": "number"},
                "avg_yield_change_pct": {"type": "number"},
            },
        },
        "macro_context": {
            "type": "object",
            "properties": {
                "mean_vix": {"type": "number"},
                "mean_ten_year_yield": {"type": "number"},
                "mean_cpi": {"type": "number"},
                "macro_regime": {"type": "string"},
            },
        },
        "end_of_week_portfolio": {
            "type": "object",
            "properties": {
                "market_value": {"type": "number"},
                "cost_basis": {"type": "number"},
                "unrealized_pnl_pct": {"type": "number"},
                "forward_yield_pct": {"type": "number"},
                "projected_monthly_income": {"type": "number"},
                "sharpe_ratio": {"type": "number"},
                "beta": {"type": "number"},
                "max_drawdown_pct": {"type": "number"},
                "vol_30d_pct": {"type": "number"},
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "weight_pct": {"type": "number"},
                            "yield_pct": {"type": "number"},
                            "market_value": {"type": "number"},
                        },
                    },
                },
            },
        },
        "meta": {
            "type": "object",
            "properties": {
                "data_sources": {"type": "array", "items": {"type": "string"}},
                "aggregation_method": {"type": "string"},
                "system_version": {"type": "string"},
                "generated_at": {"type": "string"},
                "valid": {"type": "boolean"},
            },
        },
    },
}


# ---------------------------- utility helpers ------------------------------ #
def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {msg}")


def safe_get(obj, path, default=None):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur.get(part)
    try:
        return float(cur)
    except Exception:
        return default


def delta_pct(new, old):
    if new is None or old is None or old == 0:
        return None
    return round(((new - old) / old) * 100.0, 3)


def load_daily_snapshots(n: int = MIN_DAYS):
    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.json")))
    if not files:
        return []
    # sort by date parsed from filename tail
    def _parse_date(path):
        base = os.path.basename(path).replace(".json", "")
        try:
            return date.fromisoformat(base)
        except Exception:
            return None

    dated = [(f, _parse_date(f)) for f in files]
    dated = [(f, d) for f, d in dated if d is not None]
    dated.sort(key=lambda x: x[1])
    latest = dated[-n:]
    snapshots = []
    for path, d in latest:
        with open(path, "r") as fh:
            snapshots.append((d, json.load(fh)))
    log(f"[Load] Loaded {len(snapshots)} daily snapshots")
    return snapshots


def compute_aggregates(snapshots):
    vals = [safe_get(s, "totals.market_value") for _, s in snapshots if safe_get(s, "totals.market_value") is not None]
    fyields = [safe_get(s, "income.portfolio_current_yield_pct") for _, s in snapshots if safe_get(s, "income.portfolio_current_yield_pct") is not None]
    sharpes = [safe_get(s, "portfolio_rollups.risk.sharpe_1y") for _, s in snapshots if safe_get(s, "portfolio_rollups.risk.sharpe_1y") is not None]
    betas = []
    for _, snap in snapshots:
        b = safe_get(snap, "portfolio_rollups.risk.beta_portfolio")
        if b is None:
            b = safe_get(snap, "portfolio_rollups.risk.beta_3y")
        if b is not None:
            betas.append(b)
    vols = [safe_get(s, "portfolio_rollups.risk.vol_30d_pct") for _, s in snapshots if safe_get(s, "portfolio_rollups.risk.vol_30d_pct") is not None]
    mdds = [safe_get(s, "portfolio_rollups.risk.max_drawdown_1y_pct") for _, s in snapshots if safe_get(s, "portfolio_rollups.risk.max_drawdown_1y_pct") is not None]

    mv_start = vals[0] if vals else None
    mv_end = vals[-1] if vals else None

    aggs = {
        "market_value": {
            "avg": round(sum(vals) / len(vals), 3) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "start": mv_start,
            "end": mv_end,
            "delta_pct": delta_pct(mv_end, mv_start),
        },
        "forward_yield_pct": {
            "avg": round(sum(fyields) / len(fyields), 3) if fyields else None,
            "delta": delta_pct(fyields[-1], fyields[0]) if fyields else None,
        },
        "sharpe_ratio": {
            "avg": round(sum(sharpes) / len(sharpes), 3) if sharpes else None,
            "delta": delta_pct(sharpes[-1], sharpes[0]) if sharpes else None,
        },
        "beta": {
            "avg": round(sum(betas) / len(betas), 3) if betas else None,
            "delta": delta_pct(betas[-1], betas[0]) if len(betas) >= 2 else None,
        },
        "vol_30d_pct": {"avg": round(sum(vols) / len(vols), 3) if vols else None},
        "max_drawdown_pct": {"avg": round(sum(mdds) / len(mdds), 3) if mdds else None},
    }
    return aggs


def detect_holdings_changes(start_snap, end_snap):
    def _map_holdings(snap):
        out = {}
        for h in snap.get("holdings") or []:
            sym = h.get("symbol")
            if sym:
                out[str(sym).upper()] = h
        return out

    start_map = _map_holdings(start_snap)
    end_map = _map_holdings(end_snap)
    start_syms = set(start_map.keys())
    end_syms = set(end_map.keys())

    added = sorted(list(end_syms - start_syms))
    removed = sorted(list(start_syms - end_syms))
    weight_changes = {}
    for sym in start_syms & end_syms:
        w0 = safe_get(start_map[sym], "weight_pct")
        w1 = safe_get(end_map[sym], "weight_pct")
        if w0 is not None and w1 is not None:
            weight_changes[sym] = round(w1 - w0, 3)

    def _diversification_index(weights_dict):
        weights = [v for v in weights_dict.values() if isinstance(v, (int, float))]
        if not weights:
            return None
        w = [x / 100.0 for x in weights if x is not None]
        denom = sum([x * x for x in w])
        return round(1 / denom, 3) if denom > 0 else None

    div_start = _diversification_index({k: safe_get(v, "weight_pct") for k, v in start_map.items()})
    div_end = _diversification_index({k: safe_get(v, "weight_pct") for k, v in end_map.items()})

    return {
        "positions_added": added,
        "positions_removed": removed,
        "weight_changes": weight_changes,
        "sector_rotation": {},
        "diversification_index": {
            "start": div_start,
            "end": div_end,
            "delta": round(div_end - div_start, 3) if div_start is not None and div_end is not None else None,
        },
        "positions_count": {"start": len(start_syms), "end": len(end_syms)},
    }


def compute_deltas(snapshots):
    if len(snapshots) < 2:
        return {
            "avg_market_value_change_pct": None,
            "avg_income_change_pct": None,
            "avg_yield_change_pct": None,
        }
    mv_deltas = []
    inc_deltas = []
    yld_deltas = []
    for idx in range(1, len(snapshots)):
        _, prev = snapshots[idx - 1]
        _, curr = snapshots[idx]
        mv_deltas.append(delta_pct(safe_get(curr, "totals.market_value"), safe_get(prev, "totals.market_value")))
        inc_deltas.append(delta_pct(safe_get(curr, "income.projected_monthly_income"), safe_get(prev, "income.projected_monthly_income")))
        yld_deltas.append(delta_pct(safe_get(curr, "income.portfolio_current_yield_pct"), safe_get(prev, "income.portfolio_current_yield_pct")))

    def _avg(items):
        vals = [x for x in items if x is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 3)

    return {
        "avg_market_value_change_pct": _avg(mv_deltas),
        "avg_income_change_pct": _avg(inc_deltas),
        "avg_yield_change_pct": _avg(yld_deltas),
    }


def merge_macro_context(snapshots):
    vix = []
    ten_y = []
    cpi = []
    stress = []
    for _, snap in snapshots:
        macro = snap.get("macro") or snap.get("macro_snapshot") or {}
        if isinstance(macro, dict) and "snapshot" in macro:
            macro = macro.get("snapshot") or {}
        if isinstance(macro, dict):
            if macro.get("vix") is not None:
                vix.append(float(macro.get("vix")))
            if macro.get("ten_year_yield") is not None:
                ten_y.append(float(macro.get("ten_year_yield")))
            if macro.get("cpi_yoy") is not None:
                cpi.append(float(macro.get("cpi_yoy")))
            if macro.get("macro_stress_score") is not None:
                stress.append(float(macro.get("macro_stress_score")))

    def _mean(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    mean_stress = _mean(stress)
    if mean_stress is not None and mean_stress > 1.0:
        regime = "stress"
    elif _mean(vix) is not None and _mean(vix) > 25:
        regime = "caution"
    else:
        regime = "neutral"

    return {
        "mean_vix": _mean(vix),
        "mean_ten_year_yield": _mean(ten_y),
        "mean_cpi": _mean(cpi),
        "macro_regime": regime,
    }


def compute_income_block(snapshots):
    pmi = [safe_get(s, "income.projected_monthly_income") for _, s in snapshots if safe_get(s, "income.projected_monthly_income") is not None]
    yoc = [safe_get(s, "income.portfolio_yield_on_cost_pct") for _, s in snapshots if safe_get(s, "income.portfolio_yield_on_cost_pct") is not None]
    avg_monthly_income = round(sum(pmi) / len(pmi), 3) if pmi else None
    total_weekly_income = round((avg_monthly_income or 0.0) * (7.0 / 30.0), 3) if avg_monthly_income is not None else None
    return {
        "avg_monthly_income": avg_monthly_income,
        "total_weekly_income": total_weekly_income,
        "yield_on_cost_pct": round(sum(yoc) / len(yoc), 3) if yoc else None,
        "drip_reinvested": 0.0,
    }


def end_of_week_portfolio(snapshot):
    holdings_out = []
    for h in snapshot.get("holdings") or []:
        holdings_out.append(
            {
                "symbol": h.get("symbol"),
                "weight_pct": safe_get(h, "weight_pct"),
                "yield_pct": safe_get(h, "current_yield_pct"),
                "market_value": safe_get(h, "market_value"),
            }
        )
    return {
        "market_value": safe_get(snapshot, "totals.market_value"),
        "cost_basis": safe_get(snapshot, "totals.cost_basis"),
        "unrealized_pnl_pct": safe_get(snapshot, "totals.unrealized_pct"),
        "forward_yield_pct": safe_get(snapshot, "income.portfolio_current_yield_pct"),
        "projected_monthly_income": safe_get(snapshot, "income.projected_monthly_income"),
        "sharpe_ratio": safe_get(snapshot, "portfolio_rollups.risk.sharpe_1y"),
        "beta": safe_get(snapshot, "portfolio_rollups.risk.beta_portfolio") or safe_get(snapshot, "portfolio_rollups.risk.beta_3y"),
        "max_drawdown_pct": safe_get(snapshot, "portfolio_rollups.risk.max_drawdown_1y_pct"),
        "vol_30d_pct": safe_get(snapshot, "portfolio_rollups.risk.vol_30d_pct"),
        "holdings": holdings_out,
    }


def generate_weekly_summary(snapshots):
    if len(snapshots) < 1:
        raise RuntimeError("No snapshots available to generate weekly summary.")
    snapshots = sorted(snapshots, key=lambda x: x[0])
    start_date = snapshots[0][0]
    end_date = snapshots[-1][0]
    start_snap = snapshots[0][1]
    end_snap = snapshots[-1][1]

    summary = {
        "summary_id": f"weekly_{end_date.isoformat()}",
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_included": len(snapshots),
        },
        "aggregates": compute_aggregates(snapshots),
        "income": compute_income_block(snapshots),
        "holdings_deltas": detect_holdings_changes(start_snap, end_snap),
        "comparative_deltas": compute_deltas(snapshots),
        "macro_context": merge_macro_context(snapshots),
        "end_of_week_portfolio": end_of_week_portfolio(end_snap),
        "meta": {
            "data_sources": ["daily_snapshots"],
            "aggregation_method": "7-day rolling",
            "system_version": "weekly_fusion_v1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "valid": True,
        },
    }
    return summary


def validate_schema(summary):
    try:
        validate(instance=summary, schema=SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)


def cache_summary(summary, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[Cache] Written to {cache_path}")


def load_from_cache(week_end_date):
    cache_path = os.path.join(CACHE_DIR, f"weekly_{week_end_date.strftime('%Y_%m_%d')}.json")
    if not os.path.exists(cache_path):
        return None, cache_path
    age = time.time() - os.path.getmtime(cache_path)
    if age < CACHE_TTL_SECONDS:
        with open(cache_path, "r") as fh:
            summary = json.load(fh)
        log(f"[Load] Reusing cache {cache_path}")
        return summary, cache_path
    return None, cache_path


def generate_and_write(use_cache: bool = True):
    snapshots = load_daily_snapshots(n=MIN_DAYS)
    if len(snapshots) < MIN_DAYS:
        log(f"[Skip] Only {len(snapshots)} daily snapshots; need {MIN_DAYS} to generate weekly.")
        return None, None
    week_end = snapshots[-1][0]
    cached, cache_path = load_from_cache(week_end)
    if use_cache and cached:
        valid, err = validate_schema(cached)
        if valid:
            log("[Validate] Schema OK (cache)")
            return cached, cache_path
        log(f"[Validate] Cache invalid: {err}")

    log("[Compute] Aggregates starting")
    summary = generate_weekly_summary(snapshots)
    valid, err = validate_schema(summary)
    if not valid:
        log(f"[Validate] Failed: {err}")
        raise RuntimeError(err)
    log("[Validate] Schema OK")

    # ensure directories
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    fname = f"weekly_{week_end.strftime('%Y_%m_%d')}.json"
    out_path = os.path.join(WEEKLY_DIR, fname)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[Write] Summary saved to {out_path}")

    cache_summary(summary, cache_path)
    return summary, out_path


def main():
    try:
        summary, out_path = generate_and_write(use_cache=True)
    except Exception as e:
        raise SystemExit(str(e))
    print(f"✅ Weekly Fusion Summary generated successfully: {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
