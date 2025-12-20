
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.services import market_data
from app.services.analytics import compute_risk_metrics, trailing_12m_dividend_per_share


# Fields we try to populate inside holdings[*].ultimate if missing/None
RISK_FIELDS = [
    "vol_30d_pct",
    "vol_90d_pct",
    "beta_3y",
    "max_drawdown_1y_pct",
    "sortino_1y",
    "downside_dev_1y_pct",
    "sharpe_1y",
    "calmar_1y",
    "drawdown_duration_1y_days",
    "var_95_1d_pct",
    "cvar_95_1d_pct",
    "corr_1y",
    "twr_1m_pct",
    "twr_3m_pct",
    "twr_6m_pct",
    "twr_12m_pct",
]

DIV_FIELDS = [
    "trailing_12m_div_ps",
    "trailing_12m_yield_pct",
]


def _as_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "close" in df.columns:
        s = df["close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        # best effort: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        s = df[num_cols[0]] if num_cols else pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def _need_any(ultimate: Dict, fields: List[str]) -> bool:
    for f in fields:
        if f not in ultimate or ultimate.get(f) is None:
            return True
    return False


def ensure_minimal_ultimate(
    holdings: List[Dict],
    as_of: Optional[date] = None,
    benchmark_symbol: str = "^GSPC",
) -> List[Dict]:
    """Fill missing/None 'ultimate' values using market_data fallbacks.

    This is intentionally conservative: we only compute from price/dividend histories
    and we never fabricate values if we lack enough data.
    """
    if as_of is None:
        as_of = date.today()
    start = as_of - timedelta(days=400)
    end = as_of

    # Fetch benchmark once (best effort)
    bench_close = pd.Series(dtype=float)
    try:
        bdf, _ = market_data.get_history(benchmark_symbol, start, end)
        bench_close = _as_close_series(bdf)
    except Exception:
        bench_close = pd.Series(dtype=float)

    for h in holdings:
        u = h.get("ultimate") or {}
        h["ultimate"] = u

        symbol = h.get("symbol")
        if not symbol:
            continue

        # ---- risk/performance metrics ----
        if _need_any(u, RISK_FIELDS):
            try:
                df, _ = market_data.get_history(symbol, start, end)
                close = _as_close_series(df)
                metrics = compute_risk_metrics(close, bench_close if not bench_close.empty else None)
                for k, v in metrics.items():
                    if k in RISK_FIELDS and (u.get(k) is None):
                        u[k] = v
            except Exception:
                # leave missing; snapshot layer can drop nulls for output
                pass

        # ---- trailing dividend ----
        if _need_any(u, DIV_FIELDS):
            try:
                divs = market_data.get_dividends(symbol)
                as_of_ts = pd.Timestamp(as_of)
                t12 = trailing_12m_dividend_per_share(divs, as_of_ts)
                if u.get("trailing_12m_div_ps") is None:
                    u["trailing_12m_div_ps"] = t12
                if u.get("trailing_12m_yield_pct") is None:
                    last_price = h.get("last_price")
                    if last_price and t12 is not None and last_price != 0:
                        u["trailing_12m_yield_pct"] = float(t12 / float(last_price) * 100.0)
            except Exception:
                pass

        # derived_fields housekeeping (avoid nulls, keep consistent)
        derived = u.get("derived_fields")
        if derived is None:
            u["derived_fields"] = []
        elif not isinstance(derived, list):
            u["derived_fields"] = [str(derived)]

    return holdings
