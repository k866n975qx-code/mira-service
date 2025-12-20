
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _safe_pct(x: float) -> float:
    return float(x) * 100.0


def _series_returns(close: pd.Series) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return close.pct_change().dropna()


def _max_drawdown(close: pd.Series) -> Tuple[Optional[float], Optional[int]]:
    close = close.dropna()
    if close.empty:
        return None, None
    peak = close.cummax()
    dd = (close / peak) - 1.0
    min_dd = float(dd.min())
    # drawdown duration (approx trading days) between peak and recovery; compute longest consecutive period below peak
    underwater = dd < 0
    if underwater.any():
        # compute longest consecutive True streak
        grp = (underwater != underwater.shift()).cumsum()
        lengths = underwater.groupby(grp).sum()
        max_len = int(lengths.max()) if not lengths.empty else 0
    else:
        max_len = 0
    return _safe_pct(min_dd), max_len


def _downside_dev(returns: pd.Series) -> Optional[float]:
    if returns.empty:
        return None
    downside = returns[returns < 0]
    if downside.empty:
        return 0.0
    return _safe_pct(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _sharpe(returns: pd.Series) -> Optional[float]:
    if returns.empty:
        return None
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    if sigma == 0 or np.isnan(sigma):
        return None
    return float(mu / sigma)


def _sortino(returns: pd.Series) -> Optional[float]:
    if returns.empty:
        return None
    mu = returns.mean() * TRADING_DAYS
    dd = returns[returns < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    if dd == 0 or np.isnan(dd):
        return None
    return float(mu / dd)


def _calmar(close: pd.Series, max_drawdown_pct: Optional[float]) -> Optional[float]:
    if close.empty or max_drawdown_pct is None:
        return None
    days = (close.index.max() - close.index.min()).days
    if days <= 0:
        return None
    total_ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
    ann = (1.0 + total_ret) ** (TRADING_DAYS / max(len(close) - 1, 1)) - 1.0
    mdd = abs(max_drawdown_pct) / 100.0
    if mdd == 0:
        return None
    return float(ann / mdd)


def _var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if returns.empty:
        return None, None
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(tail.mean()) if not tail.empty else var
    return _safe_pct(var), _safe_pct(cvar)


def _twr(close: pd.Series, window_days: int) -> Optional[float]:
    close = close.dropna()
    if close.empty:
        return None
    end = close.index.max()
    start = end - pd.Timedelta(days=window_days)
    sub = close[close.index >= start]
    if sub.empty:
        return None
    return _safe_pct(float(sub.iloc[-1] / sub.iloc[0] - 1.0))


def compute_risk_metrics(
    close: pd.Series,
    benchmark_close: Optional[pd.Series] = None,
) -> Dict[str, Optional[float]]:
    """Compute a consistent set of risk/perf metrics.

    Inputs should be daily close series indexed by datetime.
    Returns values are either float or None (caller can drop nulls).
    """
    close = close.dropna()
    out: Dict[str, Optional[float]] = {}

    if close.empty or len(close) < 30:
        return out

    rets = _series_returns(close)

    # Volatility
    out["vol_30d_pct"] = _safe_pct(rets.tail(30).std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) >= 30 else None
    out["vol_90d_pct"] = _safe_pct(rets.tail(90).std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) >= 90 else None

    # Drawdowns (1y)
    one_year_ago = close.index.max() - pd.Timedelta(days=365)
    close_1y = close[close.index >= one_year_ago]
    mdd, dd_dur = _max_drawdown(close_1y)
    out["max_drawdown_1y_pct"] = mdd
    out["drawdown_duration_1y_days"] = float(dd_dur) if dd_dur is not None else None

    # Sharpe/Sortino/Downside dev (1y)
    rets_1y = _series_returns(close_1y)
    out["downside_dev_1y_pct"] = _downside_dev(rets_1y)
    out["sharpe_1y"] = _sharpe(rets_1y)
    out["sortino_1y"] = _sortino(rets_1y)
    out["calmar_1y"] = _calmar(close_1y, mdd)

    # VaR/CVaR (1d, 95%)
    var95, cvar95 = _var_cvar(rets_1y, alpha=0.05)
    out["var_95_1d_pct"] = var95
    out["cvar_95_1d_pct"] = cvar95

    # Correlation & beta (1y)
    if benchmark_close is not None and not benchmark_close.dropna().empty:
        bench = benchmark_close.dropna()
        # align by date
        df = pd.DataFrame({"a": close_1y, "b": bench}).dropna()
        if len(df) >= 60:
            a = _series_returns(df["a"])
            b = _series_returns(df["b"])
            ab = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(ab) >= 60:
                out["corr_1y"] = float(ab["a"].corr(ab["b"]))
                var_b = float(ab["b"].var(ddof=1))
                out["beta_3y"] = float(ab["a"].cov(ab["b"]) / var_b) if var_b != 0 else None

    # TWR windows (approx from daily close)
    out["twr_1m_pct"] = _twr(close, 30)
    out["twr_3m_pct"] = _twr(close, 90)
    out["twr_6m_pct"] = _twr(close, 182)
    out["twr_12m_pct"] = _twr(close, 365)

    return out


def trailing_12m_dividend_per_share(dividends: pd.Series, as_of: pd.Timestamp) -> Optional[float]:
    if dividends is None or dividends.empty:
        return None
    dividends = dividends.dropna()
    if dividends.empty:
        return None
    start = as_of - pd.Timedelta(days=365)
    s = dividends[(dividends.index >= start) & (dividends.index <= as_of)]
    if s.empty:
        return 0.0
    return float(s.sum())
