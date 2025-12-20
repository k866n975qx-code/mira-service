from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from app.services.rollups import compute_category_exposure, compute_lookthrough

# -------- Helpers --------
def _drange(as_of: date, days: int) -> Tuple[date, date]:
    start = as_of - timedelta(days=days)
    return start, as_of


def _daterange_days(as_of: date, days: int) -> Tuple[date, date]:
    # Backward-compatible alias
    return _drange(as_of, days)


def _adj_close(symbol: str, start: date, end: date) -> pd.Series:
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return pd.Series(dtype=float)
    if "Close" not in df.columns and "Adj Close" not in df.columns:
        return pd.Series(dtype=float)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).date
    s = s[(s.index >= start) & (s.index <= end)]
    return s


def _adj_close_series(symbol: str, start: date, end: date) -> pd.Series:
    # Backward-compatible alias
    return _adj_close(symbol, start, end)


def _portfolio_value(holdings: List[Dict], start: date, end: date) -> pd.Series:
    """
    Approximate daily portfolio value assuming constant shares: V_t = sum_i shares_i * AdjClose_i(t).
    """
    series = []
    for h in holdings:
        sym = h.get("symbol")
        if not sym:
            continue
        shares = float(h.get("shares") or 0.0)
        if shares <= 0:
            continue
        s = _adj_close(sym, start, end)
        if s.empty:
            continue
        series.append(s * shares)
    if not series:
        return pd.Series(dtype=float)
    df = pd.concat(series, axis=1).ffill().bfill()
    v = df.sum(axis=1)
    v = v.sort_index()
    return v


def _portfolio_value_series(holdings: List[Dict], start: date, end: date) -> pd.Series:
    # Backward-compatible alias
    return _portfolio_value(holdings, start, end)


def _daily_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


def _compound_twr(r: pd.Series) -> Optional[float]:
    if r is None or len(r) == 0:
        return None
    prod = float(np.prod(1.0 + r.values) - 1.0)
    return prod


def _vol_annualized(r: pd.Series, window: int) -> Optional[float]:
    if len(r) < window:
        return None
    v = r.rolling(window).std().iloc[-1] * math.sqrt(252.0)
    return float(v) if pd.notnull(v) else None


def _max_drawdown(s: pd.Series) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    return float(dd.min()) if len(dd) else None


def _sortino(r: pd.Series) -> Optional[float]:
    if r is None or len(r) == 0:
        return None
    mean = r.mean()
    downside = r[r < 0]
    if len(downside) == 0:
        return None
    ds = downside.std()
    if ds == 0 or pd.isna(ds):
        return None
    sortino_daily = mean / ds
    return float(sortino_daily * math.sqrt(252.0))


def _beta_ols(asset_r: pd.Series, bench_r: pd.Series) -> Optional[float]:
    df = pd.concat([asset_r, bench_r], axis=1).dropna()
    if len(df) < 60:
        return None
    a = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = np.var(b)
    if var_b == 0:
        return None
    cov = np.cov(a, b, ddof=0)[0, 1]
    return float(cov / var_b)


def _first_nonempty_series(
    symbols: List[str], start: date, end: date
) -> Tuple[Optional[str], pd.Series]:
    for sym in symbols:
        s = _adj_close(sym, start, end)
        if not s.empty and len(s) > 2:
            return sym, s
    return None, pd.Series(dtype=float)


def _benchmark_twr_windows(
    as_of: date, windows: Dict[str, int], symbols: Optional[List[str]] = None
) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    symbols = symbols or ["^GSPC", "SPY", "IVV", "VOO", "^SPX", "^INX"]
    out: Dict[str, Optional[float]] = {}
    chosen: Optional[str] = None
    for label, days in windows.items():
        start, end = _drange(as_of, days)
        if chosen is None:
            chosen, series = _first_nonempty_series(symbols, start, end)
        else:
            _, series = _first_nonempty_series([chosen], start, end)
        r = series.pct_change().dropna()
        twr = float(np.prod(1.0 + r.values) - 1.0) if len(r) else None
        out[f"benchmark_twr_{label}_pct"] = None if twr is None else round(
            twr * 100.0, 3
        )
    return out, chosen


def _approx_twr_from_holdings(
    holdings: List[Dict], windows: Dict[str, int]
) -> Dict[str, Optional[float]]:
    """
    Approximate portfolio TWR by weighting per-holding ultimate.twr_* by market value.
    """
    mv_total = sum(float(h.get("market_value") or 0.0) for h in holdings)
    out: Dict[str, Optional[float]] = {}
    for label in windows.keys():
        if mv_total <= 0:
            out[f"twr_{label}_approx_pct"] = None
            continue
        acc = 0.0
        has_any = False
        for h in holdings:
            mv = float(h.get("market_value") or 0.0)
            if mv <= 0:
                continue
            w = 0.0 if mv_total == 0 else mv / mv_total
            u = h.get("ultimate") or {}
            hv = _get_twr_from_ultimate(u, label)
            if hv is None:
                continue
            acc += w * float(hv)
            has_any = True
        out[f"twr_{label}_approx_pct"] = round(acc, 3) if has_any else None
    return out


def _robust_get(d: Dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return None


def _get_twr_from_ultimate(u: Dict, label: str) -> Optional[float]:
    return _robust_get(
        u,
        [
            f"twr_{label}_pct",
            f"twr_{label}",
            f"{label}_twr_pct",
            f"twr{label}_pct",
        ],
    )


def _approx_risk_from_holdings(holdings: List[Dict]) -> Dict[str, Optional[float]]:
    mv_total = sum(float(h.get("market_value") or 0.0) for h in holdings)
    keys = {
        "vol_30d_pct": ["vol_30d_pct", "vol30d_pct"],
        "vol_90d_pct": ["vol_90d_pct", "vol90d_pct"],
        "max_drawdown_1y_pct": ["max_drawdown_1y_pct", "mdd_1y_pct"],
        "sortino_1y": ["sortino_1y", "sortino"],
        "beta_3y": ["beta_3y", "beta"],
    }
    out: Dict[str, Optional[float]] = {k: None for k in keys}
    if mv_total <= 0:
        return out
    for k, variants in keys.items():
        acc = 0.0
        has_any = False
        for h in holdings:
            mv = float(h.get("market_value") or 0.0)
            if mv <= 0:
                continue
            u = h.get("ultimate") or {}
            v = _robust_get(u, variants)
            if v is None:
                continue
            acc += (mv / mv_total) * v
            has_any = True
        out[k] = round(acc, 3) if has_any else None
    return out


# -------- Public API --------
def compute_portfolio_rollups(
    account_id: int,
    as_of: date,
    holdings: List[Dict],
    perf_method: str = "accurate",
    include_mwr: bool = True,
) -> Dict:
    """
    Portfolio-level income, performance (TWR & MWR), risk, and benchmark.
    TWR: constant-share daily portfolio value from yfinance Adj Close, with an approx fallback.
    MWR: requires optional db_cashflows fetcher; otherwise returns nulls with a note.
    """
    windows_days = {"1m": 30, "3m": 90, "6m": 180, "12m": 365}
    perf: Dict[str, Optional[float]] = {}
    approx_perf = _approx_twr_from_holdings(holdings, windows_days)
    requested = (perf_method or "accurate").lower()
    accurate_count = 0

    if requested != "approx":
        for label, days in windows_days.items():
            try:
                start, end = _drange(as_of, days)
                pv = _portfolio_value(holdings, start, end)
                if pv.empty or len(pv) < 2:
                    perf[f"twr_{label}_pct"] = None
                    continue
                r = _daily_returns(pv)
                twr = _compound_twr(r)
                if twr is None:
                    perf[f"twr_{label}_pct"] = None
                    continue
                perf[f"twr_{label}_pct"] = round(twr * 100.0, 3)
                accurate_count += 1
            except Exception:
                perf[f"twr_{label}_pct"] = None

    # always include approx
    perf.update(approx_perf)

    bench, bench_symbol = _benchmark_twr_windows(as_of, windows_days, None)
    for label in windows_days.keys():
        b = bench.get(f"benchmark_twr_{label}_pct")
        p = perf.get(f"twr_{label}_pct")
        if p is None:
            p = perf.get(f"twr_{label}_approx_pct")
        perf[f"alpha_{label}_pct"] = None if p is None or b is None else round(
            p - b, 3
        )

    # Risk metrics
    risk = {
        "vol_30d_pct": None,
        "vol_90d_pct": None,
        "max_drawdown_1y_pct": None,
        "sortino_1y": None,
        "beta_3y": None,
        "downside_dev_1y_pct": None,
        "sharpe_1y": None,
        "calmar_1y": None,
        "drawdown_duration_1y_days": None,
        "var_95_1d_pct": None,
        "cvar_95_1d_pct": None,
        "corr_1y": None,
    }
    try:
        start_1y, end_1y = _drange(as_of, 365)
        pv_1y = _portfolio_value(holdings, start_1y, end_1y)
        if not pv_1y.empty and len(pv_1y) > 2:
            r_1y = _daily_returns(pv_1y)
            v30 = _vol_annualized(r_1y, 30)
            v90 = _vol_annualized(r_1y, 90)
            mdd = _max_drawdown(pv_1y)
            sort = _sortino(r_1y)
            risk["vol_30d_pct"] = None if v30 is None else round(v30 * 100.0, 3)
            risk["vol_90d_pct"] = None if v90 is None else round(v90 * 100.0, 3)
            risk["max_drawdown_1y_pct"] = None if mdd is None else round(
                mdd * 100.0, 3
            )
            risk["sortino_1y"] = None if sort is None else round(sort, 3)
            # Additional risk metrics derived from r_1y / pv_1y (no external data)
            try:
                dn = r_1y[r_1y < 0]
                if len(dn) >= 10:
                    dd = dn.std() * math.sqrt(252.0)
                    risk["downside_dev_1y_pct"] = round(float(dd) * 100.0, 3)
            except Exception:
                pass
            try:
                mu = r_1y.mean() * 252.0
                sig = r_1y.std() * math.sqrt(252.0)
                if sig and float(sig) > 0:
                    risk["sharpe_1y"] = round(float(mu / sig), 3)  # rf=0 by default
            except Exception:
                pass
            try:
                start_v = float(pv_1y.iloc[0])
                end_v = float(pv_1y.iloc[-1])
                n = max(1, len(pv_1y) - 1)
                if start_v > 0:
                    cagr = (end_v / start_v) ** (252.0 / n) - 1.0
                    if mdd is not None and float(mdd) != 0:
                        risk["calmar_1y"] = round((cagr * 100.0) / abs(float(mdd) * 100.0), 3)
            except Exception:
                pass
            try:
                running_max = pv_1y.cummax()
                underwater = pv_1y < running_max
                max_streak = 0
                streak = 0
                for v in underwater.values:
                    if bool(v):
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                risk["drawdown_duration_1y_days"] = int(max_streak)
            except Exception:
                pass
            try:
                if len(r_1y) >= 30:
                    q = float(np.quantile(r_1y.values, 0.05))
                    tail = r_1y[r_1y <= q]
                    cvar = float(tail.mean()) if len(tail) else None
                    risk["var_95_1d_pct"] = round(q * 100.0, 3)
                    risk["cvar_95_1d_pct"] = round(cvar * 100.0, 3) if cvar is not None else None
            except Exception:
                pass
            try:
                bench = yf.Ticker("^GSPC").history(start=start_1y, end=end_1y, auto_adjust=True)
                if not bench.empty and "Close" in bench.columns:
                    br = bench["Close"].pct_change().dropna()
                    df = pd.concat([r_1y.rename("p"), br.rename("b")], axis=1).dropna()
                    if len(df) >= 20:
                        risk["corr_1y"] = round(float(df["p"].corr(df["b"])), 3)
            except Exception:
                pass
    except Exception:
        pass

    # Beta over 3y vs ^GSPC
    try:
        start_3y = as_of - timedelta(days=3 * 365)
        pv_3y = _portfolio_value(holdings, start_3y, as_of)
        if not pv_3y.empty and len(pv_3y) > 2:
            bench_syms = [s for s in [bench_symbol, "^GSPC", "SPY", "IVV", "VOO"] if s]
            _, bench3_series = _first_nonempty_series(bench_syms, start_3y, as_of)
            asset_r = _daily_returns(pv_3y)
            bench_r = _daily_returns(bench3_series)
            beta = _beta_ols(asset_r, bench_r)
            risk["beta_3y"] = None if beta is None else round(beta, 3)
    except Exception:
        pass

    approx_risk = _approx_risk_from_holdings(holdings)
    for k, v in risk.items():
        if v is None and approx_risk.get(k) is not None:
            risk[k] = approx_risk[k]

    # Income rollups from current snapshot fields
    total_mv = sum(float(h.get("market_value") or 0.0) for h in holdings)
    total_cb = sum(float(h.get("cost_basis") or 0.0) for h in holdings)
    fwd12_total = sum(float(h.get("forward_12m_dividend") or 0.0) for h in holdings)
    income = {
        "forward_12m_total": round(fwd12_total, 2),
        "projected_monthly_income": round(fwd12_total / 12.0, 3),
        "portfolio_current_yield_pct": round((fwd12_total / total_mv) * 100.0, 3)
        if total_mv > 0
        else None,
        "portfolio_yield_on_cost_pct": round((fwd12_total / total_cb) * 100.0, 3)
        if total_cb > 0
        else None,
    }

    # MWR/XIRR (if cashflows available)
    mwr: Dict[str, Optional[float]] = {}
    meta_note = None
    if include_mwr:
        try:
            from app.services.db_cashflows import fetch_account_cashflows  # optional user-implemented

            for label, days in windows_days.items():
                start, end = _daterange_days(as_of, days)
                flows = fetch_account_cashflows(
                    account_id, start, end
                )  # [{'date': date, 'amount': float}, ...]
                pv = _portfolio_value_series(holdings, start, end)
                if pv.empty or len(pv) < 2:
                    mwr[f"mwr_xirr_{label}_pct"] = None
                    continue
                v0, vT = float(pv.iloc[0]), float(pv.iloc[-1])
                legs = [(start, -v0)] + [
                    (cf["date"], float(cf["amount"])) for cf in flows
                ] + [(end, vT)]
                irr = _xirr(legs)
                mwr[f"mwr_xirr_{label}_pct"] = None if irr is None else round(
                    irr * 100.0, 3
                )
        except Exception:
            meta_note = "MWR unavailable: no cashflow fetcher or error; returning nulls."

    lookthrough = compute_lookthrough(holdings)
    by_category = compute_category_exposure(holdings)

    method = (
        "approx"
        if requested == "approx"
        else ("accurate" if accurate_count > 0 else "approx-fallback")
    )

    return {
        "income": income,
        "performance": {**perf, **mwr, **bench},
        "risk": risk,
        "benchmark": bench_symbol or "^GSPC",
        "composition": {"lookthrough": lookthrough, "by_category": by_category},
        "meta": {"method": method, "mwr_note": meta_note},
    }


# -------- XIRR --------
def _xnpv(rate: float, cashflows: List[Tuple[date, float]]) -> float:
    if rate <= -1.0:
        return float("inf")
    t0 = cashflows[0][0]
    total = 0.0
    for d, amt in cashflows:
        dt_days = (d - t0).days
        total += amt / ((1.0 + rate) ** (dt_days / 365.0))
    return total


def _xirr(
    cashflows: List[Tuple[date, float]], tol: float = 1e-6, max_iter: int = 100
) -> Optional[float]:
    """
    Robust bisection between -0.999 and 10 (1000%); returns annualized rate.
    """
    if not cashflows:
        return None
    has_pos = any(amt > 0 for _, amt in cashflows)
    has_neg = any(amt < 0 for _, amt in cashflows)
    if not (has_pos and has_neg):
        return None
    lo, hi = -0.999, 10.0
    npv_lo = _xnpv(lo, cashflows)
    npv_hi = _xnpv(hi, cashflows)
    if npv_lo * npv_hi > 0:
        for k in (20.0, 50.0, 100.0):
            hi = k
            npv_hi = _xnpv(hi, cashflows)
            if npv_lo * npv_hi <= 0:
                break
        else:
            return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        npv_mid = _xnpv(mid, cashflows)
        if abs(npv_mid) < tol:
            return mid
        if npv_lo * npv_mid <= 0:
            hi = mid
            npv_hi = npv_mid
        else:
            lo = mid
            npv_lo = npv_mid
    return mid
