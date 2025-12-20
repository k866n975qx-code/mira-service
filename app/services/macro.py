from __future__ import annotations

import json
import os
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import gzip
import time

from app.services.cache_utils import load_ttl_cache, store_ttl_cache

# Default location for persisted macro artifacts
DEFAULT_MACRO_DIR = os.getenv(
    "MIRA_MACRO_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "macro")),
)

_TTL_SECONDS = 12 * 3600  # half-day freshness


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _round_safe(val: Optional[float], places: int = 3) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), places)
    except Exception:
        return None


def _write_json_and_gzip(path: str, data: Dict[str, Any]) -> None:
    """
    Persist JSON in a compact form plus a .gz alongside to save space.
    """
    payload = json.dumps(data, separators=(",", ":"))
    with open(path, "w") as f:
        f.write(payload)
    gz_path = f"{path}.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
        gz.write(payload)


def _cleanup_old_files(base_dir: str, days: int = 30) -> None:
    cutoff = time.time() - days * 86400
    for fname in os.listdir(base_dir):
        if not fname.endswith(".json") and not fname.endswith(".json.gz"):
            continue
        fpath = os.path.join(base_dir, fname)
        try:
            if os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
        except Exception:
            continue


def _fred(series_id: str, days_back: int = 400) -> Optional[pd.Series]:
    end = date.today()
    start = end - timedelta(days=days_back)
    try:
        df = pdr.DataReader(series_id, "fred", start, end)
        if isinstance(df, pd.DataFrame):
            return df.squeeze()
        return df
    except Exception:
        return None


def _first_nonempty(*series_list: Optional[pd.Series]) -> Optional[pd.Series]:
    for s in series_list:
        if s is None:
            continue
        try:
            if len(s.dropna()) > 0:
                return s
        except Exception:
            continue
    return None


def _latest(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    try:
        return _safe_float(series.dropna().iloc[-1])
    except Exception:
        return None


def _cpi_yoy(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or len(series) < 13:
        return None
    try:
        s = series.dropna()
        if len(s) < 13:
            return None
        latest = float(s.iloc[-1])
        prior = float(s.iloc[-13])
        if prior == 0:
            return None
        return round(((latest - prior) / prior) * 100.0, 4)
    except Exception:
        return None


def _vix_latest() -> Optional[float]:
    try:
        hist = yf.Ticker("^VIX").history(period="1mo")
        if not hist.empty and "Close" in hist.columns:
            return _safe_float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


def _last_history_value(history: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    recs = (history or {}).get("records") or []
    for rec in reversed(recs):
        val = rec.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return None


def fetch_macro_snapshot(prev_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch macro snapshot using public FRED + yfinance (no API key required).
    Falls back gracefully if some inputs are missing.
    """
    now_utc = datetime.utcnow().isoformat() + "Z"
    # Rates / spreads
    d10 = _fred("DGS10")
    d2 = _fred("DGS2")
    dff = _first_nonempty(_fred("DFF"), _fred("FEDFUNDS"))
    hy = _fred("BAMLH0A0HYM2EY")  # ICE BofA US High Yield Index Option-Adjusted Spread (%)
    cpi = _fred("CPIAUCSL", days_back=2000)  # longer window for YoY calc
    unemp = _fred("UNRATE")
    m2 = _fred("WM2NS")
    gdp = _fred("GDPC1")  # quarterly, best effort

    ten_year_yield = _latest(d10)
    two_year_yield = _latest(d2)
    yield_spread = None
    if ten_year_yield is not None and two_year_yield is not None:
        yield_spread = round(ten_year_yield - two_year_yield, 4)

    fed_funds = _latest(dff)
    cpi_yoy = _cpi_yoy(cpi)
    if cpi_yoy is None:
        cpi_yoy = _last_history_value(prev_history, "cpi_yoy")
        cpi_yoy_source = "history" if cpi_yoy is not None else None
    else:
        cpi_yoy_source = "fred"
    vix = _vix_latest()
    hy_spread_pct = _latest(hy)
    hy_spread_bps = round(hy_spread_pct * 100.0, 4) if hy_spread_pct is not None else None
    m2_supply = _latest(m2)
    unemp_rate = _latest(unemp)
    gdp_level = _latest(gdp)

    snap: Dict[str, Any] = {
        "date": date.today().isoformat(),
        "ten_year_yield": ten_year_yield,
        "two_year_yield": two_year_yield,
        "yield_spread_10y_2y": yield_spread,
        "fed_funds_rate": fed_funds,
        "cpi_yoy": cpi_yoy,
        "vix": vix,
        "hy_spread_bps": hy_spread_bps,
        "m2_money_supply_bn": m2_supply,
        "unemployment_rate": unemp_rate,
        "gdp_nowcast": gdp_level,
        "meta": {
            "source": "fred+yfinance",
            "fetched_at": now_utc,
            "schema_version": "1.2",
        },
    }
    if cpi_yoy_source:
        snap["meta"]["cpi_yoy_source"] = cpi_yoy_source

    # Derived metrics (best-effort)
    if ten_year_yield is not None and cpi_yoy is not None:
        snap["real_10y_yield"] = round(ten_year_yield - cpi_yoy, 4)
    if vix is not None and hy_spread_bps is not None and unemp_rate is not None and yield_spread is not None:
        snap["macro_stress_score"] = round(
            0.3 * (vix / 20.0)
            + 0.3 * (hy_spread_bps / 400.0)
            + 0.2 * (unemp_rate / 5.0)
            + 0.2 * (abs(yield_spread) / 1.0),
            4,
        )
    if ten_year_yield is not None and vix is not None and hy_spread_bps is not None:
        snap["valuation_heat_index"] = round(
            0.4 * (20.0 / 22.0)  # forward PE proxy fallback
            + 0.3 * (ten_year_yield / 4.0)
            + 0.2 * (vix / 20.0)
            + 0.1 * (hy_spread_bps / 400.0),
            4,
        )
    if yield_spread is not None:
        snap["curve_inversion_depth"] = yield_spread
    if "real_10y_yield" in snap and hy_spread_bps is not None:
        snap["liquidity_index"] = round(1 - (snap["real_10y_yield"] + hy_spread_bps / 4000.0), 4)
    elif hy_spread_bps is not None:
        last_real = _last_history_value(prev_history, "real_10y_yield")
        if last_real is not None:
            snap["liquidity_index"] = round(1 - (last_real + hy_spread_bps / 4000.0), 4)

    return snap


def _history_path() -> str:
    _ensure_dir(DEFAULT_MACRO_DIR)
    return os.path.join(DEFAULT_MACRO_DIR, "macro_history.json")


def _snapshot_path() -> str:
    _ensure_dir(DEFAULT_MACRO_DIR)
    return os.path.join(DEFAULT_MACRO_DIR, "macro_snapshot.json")


def _trends_path() -> str:
    _ensure_dir(DEFAULT_MACRO_DIR)
    return os.path.join(DEFAULT_MACRO_DIR, "macro_trends.json")


def update_history(snapshot: Dict[str, Any], days: int = 30) -> Dict[str, Any]:
    path = _history_path()
    _cleanup_old_files(os.path.dirname(path), days=days)
    if os.path.exists(path):
        try:
            history = json.load(open(path))
        except Exception:
            history = {"records": [], "meta": {}}
    else:
        history = {"records": [], "meta": {}}

    records = history.get("records") or []
    records = [r for r in records if r.get("date") != snapshot.get("date")]
    records.append(
        {
            "date": snapshot.get("date"),
            "valuation_heat_index": snapshot.get("valuation_heat_index"),
            "real_10y_yield": snapshot.get("real_10y_yield"),
            "curve_inversion_depth": snapshot.get("curve_inversion_depth"),
            "macro_stress_score": snapshot.get("macro_stress_score"),
            "liquidity_index": snapshot.get("liquidity_index"),
        }
    )
    records = sorted(records, key=lambda x: x.get("date") or "")[-days:]
    history["records"] = records
    history["meta"] = {
        "count": len(records),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json_and_gzip(path, history)
    return history


def compute_trends(history: Dict[str, Any]) -> Dict[str, Any]:
    records = history.get("records") or []
    if not records:
        return {
            "date": None,
            "trends": {},
            "meta": {"window_days": 0, "computed_at": datetime.utcnow().isoformat() + "Z", "schema_version": "1.2"},
        }
    df = pd.DataFrame(records).set_index("date")
    trends: Dict[str, Dict[str, Optional[float]]] = {}
    for col in ["valuation_heat_index", "macro_stress_score", "liquidity_index"]:
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.dropna()
        if series.empty:
            trends[col] = {"delta_7d": None, "delta_30d": None, "sma_7d": None, "sma_30d": None}
            continue
        last = series.iloc[-1]
        # Use available window for deltas; fallback to earliest if shorter than 7/30.
        idx_7 = max(0, len(series) - 7)
        idx_30 = max(0, len(series) - 30)
        delta_7d = _round_safe(last - series.iloc[idx_7], 3) if len(series) >= 1 else None
        delta_30d = _round_safe(last - series.iloc[idx_30], 3) if len(series) >= 1 else None
        trends[col] = {
            "delta_7d": delta_7d,
            "delta_30d": delta_30d,
            "sma_7d": _round_safe(series.tail(7).mean(), 3),
            "sma_30d": _round_safe(series.tail(30).mean(), 3) if len(series) >= 30 else _round_safe(series.mean(), 3),
        }
    macro_trends = {
        "date": records[-1].get("date"),
        "trends": trends,
        "meta": {
            "window_days": len(records),
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "schema_version": "1.2",
        },
    }
    _write_json_and_gzip(_trends_path(), macro_trends)
    return macro_trends


def _load_existing() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    snap = hist = trends = None
    try:
        if os.path.exists(_snapshot_path()):
            snap = json.load(open(_snapshot_path()))
    except Exception:
        snap = None
    try:
        if os.path.exists(_history_path()):
            hist = json.load(open(_history_path()))
    except Exception:
        hist = None
    try:
        if os.path.exists(_trends_path()):
            trends = json.load(open(_trends_path()))
    except Exception:
        trends = None
    return snap, hist, trends


def get_macro_package(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Return a macro package: snapshot + history + trends.
    Tries to use cached files; recomputes if stale or missing.
    """
    snap, hist, trends = _load_existing()
    today = date.today().isoformat()
    fresh = (snap is not None) and snap.get("date") == today and not force_refresh
    if fresh and hist and trends:
        return {"snapshot": snap, "history": hist, "trends": trends}

    # To avoid repeated heavy calls, use a TTL disk cache guard
    guard_key = "macro_guard"
    guard_ok = load_ttl_cache("macro", guard_key) is None or force_refresh
    if guard_ok:
        try:
            store_ttl_cache("macro", guard_key, {"started_at": datetime.utcnow().isoformat() + "Z"}, 300)
        except Exception:
            pass
        snap = fetch_macro_snapshot()
        try:
            _write_json_and_gzip(_snapshot_path(), snap)
        except Exception:
            pass
        hist = update_history(snap)
        trends = compute_trends(hist)
    # Re-load to ensure consistent shapes
    snap2, hist2, trends2 = _load_existing()
    return {
        "snapshot": snap2 or snap or {},
        "history": hist2 or hist or {},
        "trends": trends2 or trends or {},
    }
