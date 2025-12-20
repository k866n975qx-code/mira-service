
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple

import math
import numpy as np
import pandas as pd


def _returns_from_close(close: pd.Series) -> pd.Series:
    close = close.dropna()
    if len(close) < 2:
        return pd.Series(dtype=float)
    return close.pct_change().dropna()


def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
    return float(daily_vol * math.sqrt(trading_days))


def compute_vol_pct(close: pd.Series, window_days: int) -> Optional[float]:
    r = _returns_from_close(close).tail(window_days)
    if len(r) < max(10, window_days // 3):
        return None
    return annualize_vol(float(r.std(ddof=1))) * 100.0


def compute_beta_and_corr(asset_close: pd.Series, bench_close: pd.Series, window_days: int) -> Tuple[Optional[float], Optional[float]]:
    a = _returns_from_close(asset_close).tail(window_days)
    b = _returns_from_close(bench_close).tail(window_days)
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if df.shape[0] < max(30, window_days // 6):
        return None, None
    a = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = float(np.var(b, ddof=1))
    if var_b == 0.0:
        return None, None
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    beta = cov / var_b
    corr = float(np.corrcoef(a, b)[0, 1])
    return float(beta), float(corr)


def compute_max_drawdown_pct(close: pd.Series, window_days: int) -> Optional[float]:
    s = close.dropna().tail(window_days)
    if len(s) < max(30, window_days // 6):
        return None
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    return float(dd.min() * 100.0)


def compute_drawdown_duration_days(close: pd.Series, window_days: int) -> Optional[int]:
    s = close.dropna().tail(window_days)
    if isinstance(s, pd.DataFrame):
        # If a DataFrame sneaks through (e.g., multi-column Close), use the first column.
        s = s.iloc[:, 0]
    if len(s) < max(30, window_days // 6):
        return None
    running_max = s.cummax()
    in_dd = s < running_max
    if not in_dd.any():
        return 0
    # longest consecutive drawdown streak
    streak = 0
    best = 0
    for v in in_dd.values:
        if bool(v):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def compute_sharpe_sortino(close: pd.Series, window_days: int, rf_annual: float = 0.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    r = _returns_from_close(close).tail(window_days)
    if len(r) < max(30, window_days // 6):
        return None, None, None
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = r - rf_daily
    vol = float(r.std(ddof=1))
    if vol == 0.0:
        sharpe = None
    else:
        sharpe = float((excess.mean() / vol) * math.sqrt(252))
    downside = excess[excess < 0]
    dd = float(downside.std(ddof=1)) if len(downside) > 5 else 0.0
    sortino = None if dd == 0.0 else float((excess.mean() / dd) * math.sqrt(252))
    downside_dev_pct = None if dd == 0.0 else float(annualize_vol(dd) * 100.0)
    return sharpe, sortino, downside_dev_pct


def compute_calmar(close: pd.Series, window_days: int) -> Optional[float]:
    s = close.dropna().tail(window_days)
    if len(s) < max(30, window_days // 6):
        return None
    # CAGR over window
    n = len(s)
    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)
    years = n / 252.0
    if years <= 0:
        return None
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    mdd = compute_max_drawdown_pct(s, window_days)
    if mdd is None or mdd == 0:
        return None
    return float(cagr / abs(mdd / 100.0))


def compute_var_cvar(close: pd.Series, window_days: int, level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    r = _returns_from_close(close).tail(window_days)
    if len(r) < max(60, window_days // 4):
        return None, None
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) > 0 else q
    return float(q * 100.0), float(cvar * 100.0)


def compute_twr_pct(close: pd.Series, window_days: int) -> Optional[float]:
    s = close.dropna().tail(window_days)
    if len(s) < max(2, window_days // 6):
        return None
    return float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0)
