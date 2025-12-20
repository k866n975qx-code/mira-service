from __future__ import annotations

from datetime import date, timedelta
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services import market_data, metrics
from app.services.provenance import conflict_entry, missing_entry, prov_entry
from app.services.cache_utils import load_ttl_cache, store_ttl_cache

try:  # type: ignore
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore

try:  # type: ignore
    from yahooquery import Ticker as YQ
except Exception:  # pragma: no cover
    YQ = None  # type: ignore

try:  # type: ignore
    import QuantLib as ql
except Exception:  # pragma: no cover
    ql = None  # type: ignore

try:  # type: ignore
    import financetoolkit as ftk
    from financetoolkit import risk as ftk_risk
except Exception:  # pragma: no cover
    ftk = None  # type: ignore
    ftk_risk = None  # type: ignore

try:  # type: ignore
    import vectorbt as vbt
except Exception:  # pragma: no cover
    vbt = None  # type: ignore


def _series_from_history(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in ("Adj Close", "close", "Close"):
        if col in df.columns:
            s = df[col]
            break
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        s = df[num_cols[0]] if num_cols else pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def _history_cache_ttl_seconds() -> int:
    try:
        return int(os.getenv("MIRA_HISTORY_CACHE_TTL_SECONDS", "3600"))
    except Exception:
        return 3600


def _series_to_cache_payload(s: pd.Series) -> List[List[Any]]:
    return [[pd.to_datetime(idx).isoformat(), float(val)] for idx, val in s.dropna().items()]


def _series_from_cache_payload(payload: Any) -> pd.Series:
    if not isinstance(payload, list):
        return pd.Series(dtype=float)
    data = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        ts, val = row
        try:
            dt = pd.to_datetime(ts).tz_localize(None)
            fv = float(val)
            data.append((dt, fv))
        except Exception:
            continue
    if not data:
        return pd.Series(dtype=float)
    idx, vals = zip(*data)
    return pd.Series(list(vals), index=pd.to_datetime(list(idx))).sort_index()


def fetch_price_history(symbol: str, start: date, end: date) -> Tuple[pd.Series, dict]:
    cache_key = f"price:{symbol}:{start.isoformat()}:{end.isoformat()}"
    cached = load_ttl_cache("history_series", cache_key)
    if cached is not None:
        s = _series_from_cache_payload(cached)
        if not s.empty:
            return s, prov_entry("cache", "served", "price_history_cache", [symbol], note="ttl_cache_hit")

    if yf is not None:
        try:
            df = yf.download(symbol, start=start, end=end + timedelta(days=1), auto_adjust=False, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                s = _series_from_history(df)
                try:
                    store_ttl_cache("history_series", cache_key, _series_to_cache_payload(s), _history_cache_ttl_seconds())
                except Exception:
                    pass
                return s, prov_entry("yfinance", "pulled", "price_history", [symbol])
        except Exception:
            pass

    if YQ is not None:
        try:
            yq = YQ(symbol)
            hist = yq.history(start=start.isoformat(), end=end.isoformat())
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                # yahooquery returns multi-index with symbol level; drop it if present
                if isinstance(hist.index, pd.MultiIndex):
                    hist = hist.reset_index(level=0, drop=True)
                s = _series_from_history(hist)
                try:
                    store_ttl_cache("history_series", cache_key, _series_to_cache_payload(s), _history_cache_ttl_seconds())
                except Exception:
                    pass
                return s, prov_entry("yahooquery", "pulled", "price_history", [symbol])
        except Exception:
            pass

    df, prov = market_data.get_history(symbol, start=start, end=end)
    s = _series_from_history(df)
    try:
        store_ttl_cache("history_series", cache_key, _series_to_cache_payload(s), _history_cache_ttl_seconds())
    except Exception:
        pass
    provider = getattr(prov, "provider", "unknown")
    fetched_at = getattr(prov, "fetched_at", None)
    note = getattr(prov, "note", None)
    return s, prov_entry(provider, "pulled", "price_history", [symbol], fetched_at=fetched_at, note=note)


def fetch_dividend_history(symbol: str, start: date, end: date) -> Tuple[pd.Series, dict]:
    cache_key = f"div:{symbol}:{start.isoformat()}:{end.isoformat()}"
    cached = load_ttl_cache("history_series", cache_key)
    if cached is not None:
        s = _series_from_cache_payload(cached)
        if not s.empty:
            return s, prov_entry("cache", "served", "dividend_history_cache", [symbol], note="ttl_cache_hit")

    if yf is not None:
        try:
            div = yf.Ticker(symbol).dividends
            if div is not None and len(div) > 0:
                div = div[(div.index.date >= start) & (div.index.date <= end)]
                div.index = pd.to_datetime(div.index).tz_localize(None)
                try:
                    store_ttl_cache("history_series", cache_key, _series_to_cache_payload(div), _history_cache_ttl_seconds())
                except Exception:
                    pass
                return div, prov_entry("yfinance", "pulled", "dividend_history", [symbol])
        except Exception:
            pass

    if YQ is not None:
        try:
            t = YQ(symbol)
            dh = t.dividend_history
            if isinstance(dh, pd.DataFrame) and len(dh) > 0:
                col = None
                for candidate in ("dividend", "Dividends", "amount", "value"):
                    if candidate in dh.columns:
                        col = candidate
                        break
                if col:
                    s = pd.to_numeric(dh[col], errors="coerce")
                    s.index = pd.to_datetime(dh.index).tz_localize(None)
                    s = s[(s.index.date >= start) & (s.index.date <= end)]
                    s = s.dropna()
                    try:
                        store_ttl_cache("history_series", cache_key, _series_to_cache_payload(s), _history_cache_ttl_seconds())
                    except Exception:
                        pass
                    return s, prov_entry("yahooquery", "pulled", "dividend_history", [symbol])
        except Exception:
            pass

    # openbb is optional and often unavailable on newer Python versions
    try:
        from openbb import obb  # type: ignore

        try:
            res = obb.equity.dividends(symbol=symbol, start_date=start.isoformat(), end_date=end.isoformat())
            df = None
            if hasattr(res, "results") and res.results is not None:
                df = res.results if isinstance(res.results, pd.DataFrame) else None
            elif hasattr(res, "to_df"):
                df = res.to_df()
            if df is not None and len(df) > 0:
                col = None
                for candidate in ("dividend", "Dividends", "amount", "value", "cash_dividend"):
                    if candidate in df.columns:
                        col = candidate
                        break
                if col:
                    s = pd.to_numeric(df[col], errors="coerce")
                    s.index = pd.to_datetime(df.index if df.index.name else df.get("date", df.index)).tz_localize(None)
                    s = s[(s.index.date >= start) & (s.index.date <= end)]
                    s = s.dropna()
                    try:
                        store_ttl_cache("history_series", cache_key, _series_to_cache_payload(s), _history_cache_ttl_seconds())
                    except Exception:
                        pass
                    return s, prov_entry("openbb", "pulled", "dividend_history", [symbol])
        except Exception:
            pass
    except Exception:
        pass

    empty = pd.Series(dtype=float)
    try:
        store_ttl_cache("history_series", cache_key, _series_to_cache_payload(empty), _history_cache_ttl_seconds())
    except Exception:
        pass
    return empty, missing_entry("none", "dividend_history", [symbol], note="no provider returned dividends")


def _median_gap_days(ex_dates: List[pd.Timestamp]) -> Optional[int]:
    if len(ex_dates) < 2:
        return None
    gaps = [(ex_dates[i] - ex_dates[i - 1]).days for i in range(1, len(ex_dates))]
    if not gaps:
        return None
    gaps = sorted(gaps)
    return int(gaps[len(gaps) // 2])


def compute_dividend_fields(symbol: str, as_of: date, last_price: Optional[float]) -> Tuple[Dict[str, Any], Dict[str, dict], List[str]]:
    from typing import Any  # local import to avoid polluting global namespace

    start = as_of - timedelta(days=365 * 3 + 30)
    divs_primary, div_prov = fetch_dividend_history(symbol, start, as_of)

    # Attempt OpenBB validation separately
    divs_openbb: pd.Series = pd.Series(dtype=float)
    try:
        from openbb import obb  # type: ignore

        try:
            res = obb.equity.dividends(symbol=symbol, start_date=start.isoformat(), end_date=as_of.isoformat())
            if hasattr(res, "results") and isinstance(res.results, pd.DataFrame):
                df = res.results
            elif hasattr(res, "to_df"):
                df = res.to_df()
            else:
                df = None
            if df is not None and len(df) > 0:
                col = None
                for candidate in ("dividend", "Dividends", "amount", "value", "cash_dividend"):
                    if candidate in df.columns:
                        col = candidate
                        break
                if col:
                    divs_openbb = pd.to_numeric(df[col], errors="coerce")
                    divs_openbb.index = pd.to_datetime(df.index if df.index.name else df.get("date", df.index)).tz_localize(None)
                    divs_openbb = divs_openbb[(divs_openbb.index.date >= start) & (divs_openbb.index.date <= as_of)]
                    divs_openbb = divs_openbb.dropna().sort_index()
        except Exception:
            divs_openbb = pd.Series(dtype=float)
    except Exception:
        divs_openbb = pd.Series(dtype=float)

    prov: Dict[str, dict] = {}
    conflicts: List[str] = []
    fields: Dict[str, Any] = {}

    if divs_openbb.empty:
        prov["dividends_openbb_validation"] = missing_entry(
            provider="openbb",
            method="dividend_history_validation",
            inputs=[symbol],
            note="openbb unavailable or returned no dividends",
        )

    # trailing 12m dividend per share
    def _t12(series: pd.Series) -> Optional[float]:
        if series is None or len(series) == 0:
            return None
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=365)
        tail = series[(series.index >= cutoff) & (series.index <= pd.Timestamp(as_of))]
        if len(tail) == 0:
            return None
        return float(tail.sum())

    t12_primary = _t12(divs_primary)
    t12_openbb = _t12(divs_openbb)

    if t12_primary is not None:
        fields["trailing_12m_div_ps"] = round(t12_primary, 6)
        prov["trailing_12m_div_ps"] = div_prov
    elif t12_openbb is not None:
        fields["trailing_12m_div_ps"] = round(t12_openbb, 6)
        prov["trailing_12m_div_ps"] = prov_entry("openbb", "pulled", "dividend_history", [symbol])
    else:
        prov["trailing_12m_div_ps"] = missing_entry(div_prov.get("provider", "none"), "ttm_div_sum", [symbol], note="no dividend history available")

    if t12_primary is not None and t12_openbb is not None:
        diff = abs(t12_primary - t12_openbb)
        if diff > max(0.01, 0.02 * max(t12_primary, 1.0)):
            prov["trailing_12m_div_ps"] = conflict_entry(
                provider="yfinance+openbb",
                method="dividend_history_validation",
                inputs=[symbol],
                note=f"openbb ttm {t12_openbb:.4f} vs yahoo {t12_primary:.4f}",
                validators=["openbb"],
            )
            conflicts.append("holdings[].ultimate.trailing_12m_div_ps")
        else:
            val_by = prov.get("trailing_12m_div_ps", {}).get("validated_by", [])
            if isinstance(val_by, list):
                val_by.append("openbb")
                prov["trailing_12m_div_ps"]["validated_by"] = sorted(set(val_by))

    # ex-dates / frequency / projection
    ex_dates = list(divs_primary.index) if isinstance(divs_primary, pd.Series) else []
    if not ex_dates and isinstance(divs_openbb, pd.Series):
        ex_dates = list(divs_openbb.index)
    ex_dates = sorted(pd.to_datetime(ex_dates))
    if ex_dates:
        fields["last_ex_date"] = ex_dates[-1].date().isoformat()
        prov["last_ex_date"] = div_prov
        gap = _median_gap_days(ex_dates[-6:])  # use recent cadence
        freq = None
        if gap is not None:
            if gap < 45:
                freq = "monthly"
            elif gap < 120:
                freq = "quarterly"
            else:
                freq = "irregular"
        if freq:
            fields["distribution_frequency"] = freq
            prov["distribution_frequency"] = prov_entry(div_prov.get("provider", "yfinance"), "derived", "freq_from_exdate_gaps", [symbol])
        if gap:
            fields["next_ex_date_est"] = (ex_dates[-1] + pd.Timedelta(days=gap)).date().isoformat()
            prov["next_ex_date_est"] = prov_entry(div_prov.get("provider", "yfinance"), "derived", "median_gap_projection", [symbol])
    else:
        prov["last_ex_date"] = missing_entry(div_prov.get("provider", "none"), "last_ex_date", [symbol], note="no ex-dates found")
        prov["distribution_frequency"] = missing_entry(div_prov.get("provider", "none"), "freq_from_exdate_gaps", [symbol], note="insufficient ex-dates")
        prov["next_ex_date_est"] = missing_entry(div_prov.get("provider", "none"), "median_gap_projection", [symbol], note="insufficient ex-dates")

    # forward dividend per share: indicated -> trailing
    forward_ps = None
    info_rate = None
    if yf is not None:
        try:
            info = yf.Ticker(symbol).info or {}
            if isinstance(info.get("dividendRate"), (int, float)):
                info_rate = float(info["dividendRate"])
        except Exception:
            info_rate = None
    if info_rate and info_rate > 0:
        forward_ps = float(info_rate)
        prov["forward_12m_div_ps"] = prov_entry("yfinance", "pulled", "dividendRate", [symbol])
    elif fields.get("trailing_12m_div_ps") is not None:
        forward_ps = float(fields["trailing_12m_div_ps"])
        prov["forward_12m_div_ps"] = prov_entry(div_prov.get("provider", "yfinance"), "derived", "forward_equals_ttm", [symbol])
    else:
        prov["forward_12m_div_ps"] = missing_entry(div_prov.get("provider", "none"), "forward_estimate", [symbol], note="no indicated or trailing dividends")
    if forward_ps is not None:
        fields["forward_12m_div_ps"] = round(forward_ps, 6)

    # trailing / forward yields (per-share to pct)
    if last_price and last_price > 0:
        if fields.get("trailing_12m_div_ps") is not None:
            fields["trailing_12m_yield_pct"] = round(float(fields["trailing_12m_div_ps"]) / float(last_price) * 100.0, 3)
            prov["trailing_12m_yield_pct"] = prov_entry(div_prov.get("provider", "yfinance"), "derived", "ttm_yield", [symbol, "last_price"])
        if forward_ps is not None:
            fields["forward_yield_pct"] = round(float(forward_ps) / float(last_price) * 100.0, 3)
            prov["forward_yield_pct"] = prov_entry(div_prov.get("provider", "yfinance"), "derived", "forward_yield", [symbol, "last_price"])

    return fields, prov, conflicts


def _tol(field: str) -> float:
    tolerance_map = {
        "vol_30d_pct": 0.5,
        "vol_90d_pct": 0.5,
        "beta_3y": 0.1,
        "corr_1y": 0.05,
        "sharpe_1y": 0.1,
        "sortino_1y": 0.1,
        "downside_dev_1y_pct": 0.5,
        "max_drawdown_1y_pct": 0.5,
        "drawdown_duration_1y_days": 5.0,
        "calmar_1y": 0.2,
        "var_95_1d_pct": 0.1,
        "cvar_95_1d_pct": 0.1,
        "twr_1m_pct": 0.25,
        "twr_3m_pct": 0.25,
        "twr_6m_pct": 0.25,
        "twr_12m_pct": 0.25,
    }
    return tolerance_map.get(field, 0.25)


def _quantlib_var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if ql is None or returns is None or len(returns) == 0:
        return None, None
    mu = float(returns.mean())
    sigma = float(returns.std())
    if sigma == 0 or np.isnan(sigma):
        return None, None
    try:
        inv = ql.InvCumulativeNormalDistribution()(alpha)
    except Exception:
        return None, None
    pdf = ql.NormalDistribution()(inv)
    var = mu + inv * sigma
    cvar = mu - sigma * pdf / alpha
    return float(var * 100.0), float(cvar * 100.0)


def _vectorbt_returns(close: pd.Series) -> pd.Series:
    if vbt is None:
        raise ImportError("vectorbt unavailable")
    if close is None or len(close) < 2:
        return pd.Series(dtype=float)
    # vectorbt relies on numpy stride tricks; fall back gracefully if incompatible
    try:
        pf = vbt.Portfolio.from_prices(close, fees=0.0, freq="1D")
        return pf.returns()
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise ImportError(f"vectorbt failed: {exc}") from exc


def compute_risk_metrics_validated(
    symbol: str,
    close: pd.Series,
    bench_close: pd.Series,
    existing: Dict[str, Any],
    provider: str,
) -> Tuple[Dict[str, Any], Dict[str, dict], List[str]]:
    from typing import Any  # local import

    prov: Dict[str, dict] = {}
    conflicts: List[str] = []
    out: Dict[str, Any] = {}

    close = close.dropna()
    bench_close = bench_close.dropna()
    returns = close.pct_change().dropna()
    bench_returns = bench_close.pct_change().dropna()

    # Internal baseline (primary)
    internal = {
        "vol_30d_pct": metrics.compute_vol_pct(close, 30),
        "vol_90d_pct": metrics.compute_vol_pct(close, 90),
        "max_drawdown_1y_pct": metrics.compute_max_drawdown_pct(close.tail(252), 252),
        "drawdown_duration_1y_days": metrics.compute_drawdown_duration_days(close.tail(252), 252),
        "sharpe_1y": None,
        "sortino_1y": None,
        "downside_dev_1y_pct": None,
        "calmar_1y": None,
        "var_95_1d_pct": None,
        "cvar_95_1d_pct": None,
        "corr_1y": None,
        "beta_3y": None,
        "twr_1m_pct": metrics.compute_twr_pct(close, 21),
        "twr_3m_pct": metrics.compute_twr_pct(close, 63),
        "twr_6m_pct": metrics.compute_twr_pct(close, 126),
        "twr_12m_pct": metrics.compute_twr_pct(close, 252),
    }
    sharpe, sortino, downside_dev = metrics.compute_sharpe_sortino(close.tail(252), 252)
    internal["sharpe_1y"] = sharpe
    internal["sortino_1y"] = sortino
    internal["downside_dev_1y_pct"] = downside_dev
    internal["calmar_1y"] = metrics.compute_calmar(close.tail(252), 252)
    var_int, cvar_int = metrics.compute_var_cvar(close.tail(252), 252, level=0.95)
    internal["var_95_1d_pct"] = var_int
    internal["cvar_95_1d_pct"] = cvar_int
    beta, corr = metrics.compute_beta_and_corr(close.tail(252 * 3), bench_close.tail(252 * 3), 252 * 3)
    internal["beta_3y"] = beta
    corr_1y = metrics.compute_beta_and_corr(close.tail(252), bench_close.tail(252), 252)[1]
    internal["corr_1y"] = corr_1y

    # FinanceToolkit validations (where available)
    ftk_vals: Dict[str, Any] = {}
    if ftk_risk is not None and not returns.empty:
        try:
            ftk_vals["var_95_1d_pct"] = float(ftk_risk.get_var(returns, 0.05)) * 100.0
            ftk_vals["cvar_95_1d_pct"] = float(ftk_risk.get_cvar(returns, 0.05)) * 100.0
            ftk_vals["max_drawdown_1y_pct"] = float(ftk_risk.get_max_drawdown(returns.tail(252))) * 100.0
        except Exception:
            ftk_vals = {}

    # Vectorbt recomputation best-effort
    vbt_vals: Dict[str, Any] = {}
    if vbt is not None:
        try:
            r_vbt = _vectorbt_returns(close.tail(252))
            if not r_vbt.empty:
                vbt_vals["var_95_1d_pct"] = float(np.quantile(r_vbt.values, 0.05) * 100.0)
                vbt_vals["cvar_95_1d_pct"] = float(r_vbt[r_vbt <= (vbt_vals["var_95_1d_pct"] / 100.0)].mean() * 100.0)
                vbt_vals["vol_30d_pct"] = float(r_vbt.tail(30).std(ddof=1) * np.sqrt(252) * 100.0) if len(r_vbt) >= 30 else None
                vbt_vals["vol_90d_pct"] = float(r_vbt.tail(90).std(ddof=1) * np.sqrt(252) * 100.0) if len(r_vbt) >= 90 else None
        except Exception:
            vbt_vals = {}

    # QuantLib confirmation for VaR / CVaR
    ql_var, ql_cvar = _quantlib_var_cvar(returns.tail(252), 0.05)

    for field, base_value in internal.items():
        current = existing.get(field)
        value = current if current is not None else base_value
        if value is not None:
            out[field] = round(float(value), 3) if not isinstance(value, (int, np.integer)) else int(value)
            src_type = "pulled" if current is not None else "derived"
            prov[field] = prov_entry(provider, src_type, "price_history_metrics", [symbol])
        else:
            prov[field] = missing_entry(provider, "price_history_metrics", [symbol], note="insufficient history")

        # FinanceToolkit validation
        if field in ftk_vals and value is not None:
            diff = abs(float(ftk_vals[field]) - float(value))
            if diff > _tol(field):
                prov[field] = conflict_entry(
                    provider=f"{provider}+financetoolkit",
                    method="financetoolkit_validation",
                    inputs=[symbol],
                    note=f"{field} mismatch internal {value} vs financetoolkit {ftk_vals[field]}",
                    validators=["financetoolkit"],
                )
                conflicts.append(f"holdings[].ultimate.{field}")
            else:
                val_by = prov.get(field, {}).get("validated_by", [])
                if isinstance(val_by, list):
                    val_by.append("financetoolkit")
                    prov[field]["validated_by"] = sorted(set(val_by))

        # Vectorbt validation
        if field in vbt_vals and value is not None and vbt_vals[field] is not None:
            diff = abs(float(vbt_vals[field]) - float(value))
            if diff > _tol(field):
                prov[field] = conflict_entry(
                    provider=f"{provider}+vectorbt",
                    method="vectorbt_validation",
                    inputs=[symbol],
                    note=f"{field} mismatch internal {value} vs vectorbt {vbt_vals[field]}",
                    validators=(prov.get(field, {}).get("validated_by") or []) + ["vectorbt"],
                )
                conflicts.append(f"holdings[].ultimate.{field}")
            else:
                val_by = prov.get(field, {}).get("validated_by", [])
                if isinstance(val_by, list):
                    val_by.append("vectorbt")
                    prov[field]["validated_by"] = sorted(set(val_by))

        # QuantLib confirmation for VaR/CVaR
        if field in ("var_95_1d_pct", "cvar_95_1d_pct") and ql_var is not None and value is not None:
            ref_val = ql_var if field == "var_95_1d_pct" else ql_cvar
            if ref_val is not None:
                diff = abs(float(ref_val) - float(value))
                if diff > _tol(field):
                    prov[field] = conflict_entry(
                        provider=f"{provider}+quantlib",
                        method="quantlib_validation",
                        inputs=[symbol],
                        note=f"{field} mismatch internal {value} vs quantlib {ref_val}",
                        validators=(prov.get(field, {}).get("validated_by") or []) + ["quantlib"],
                    )
                    conflicts.append(f"holdings[].ultimate.{field}")
                else:
                    val_by = prov.get(field, {}).get("validated_by", [])
                    if isinstance(val_by, list):
                        val_by.append("quantlib")
                        prov[field]["validated_by"] = sorted(set(val_by))

    return out, prov, conflicts
