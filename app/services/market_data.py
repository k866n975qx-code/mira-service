
from __future__ import annotations

from dataclasses import dataclass
import copy
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import math
import logging

import pandas as pd

log = logging.getLogger(__name__)

# Optional deps (kept optional so service can boot even if provider is down/missing)
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

try:
    from openbb import obb  # type: ignore
except Exception:  # pragma: no cover
    obb = None

try:
    from yahooquery import Ticker as YQTicker  # type: ignore
except Exception:  # pragma: no cover
    YQTicker = None

try:
    from pandas_datareader import data as pdr  # type: ignore
except Exception:  # pragma: no cover
    pdr = None

try:
    from yahoo_fin import stock_info as yfin_si  # type: ignore
except Exception:  # pragma: no cover
    yfin_si = None


@dataclass(frozen=True)
class FieldProvenance:
    provider: str
    fetched_at: str  # ISO8601
    note: Optional[str] = None


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ingest_cache_ttl_seconds() -> int:
    try:
        return int(os.getenv("MIRA_INGEST_CACHE_TTL_SECONDS", "3600"))
    except Exception:
        return 3600


_INGEST_CACHE: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _INGEST_CACHE.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if expires_at < time.time():
        return None
    # Return a defensive copy to avoid callers mutating the cached object.
    if hasattr(value, "copy"):
        try:
            return value.copy(deep=True)  # type: ignore[arg-type]
        except Exception:
            try:
                return value.copy()  # type: ignore[arg-type]
            except Exception:
                pass
    return copy.deepcopy(value)


def _cache_set(key: str, value: Any) -> None:
    _INGEST_CACHE[key] = (time.time() + _ingest_cache_ttl_seconds(), value)


def _stooq_symbol(symbol: str) -> str:
    """
    Stooq uses lowercase symbols and often requires suffixes. For US equities/ETFs:
      AAPL -> aapl.us

    For indices, coverage varies; we keep the raw symbol as fallback.
    """
    s = symbol.strip()
    if s.startswith("^"):
        return s.lower()
    return f"{s.lower()}.us"


def get_history(symbol: str, start: date, end: date) -> Tuple[pd.DataFrame, FieldProvenance]:
    """
    Return OHLCV history with a DatetimeIndex and at least a 'Close' column.
    Provider order:
      1) yfinance
      2) openbb (if available)
      3) pandas-datareader (stooq)
      4) yahoo_fin
    """
    cache_key = f"hist:{symbol}:{start.isoformat()}:{end.isoformat()}"
    cached = _cache_get(cache_key)
    if cached:
        df, prov = cached
        return df, prov

    # 1) yfinance
    if yf is not None:
        try:
            df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), auto_adjust=False, progress=False)
            if df is not None and len(df) > 0 and "Close" in df.columns:
                prov = FieldProvenance(provider="yfinance", fetched_at=_now_iso())
                _cache_set(cache_key, (df, prov))
                return df, prov
        except Exception as e:
            log.warning("yfinance history failed for %s: %s", symbol, e)

    # 2) openbb (best effort)
    if obb is not None:
        try:
            res = obb.equity.price.historical(symbol=symbol, start_date=start.isoformat(), end_date=end.isoformat())
            df = None
            if hasattr(res, "results") and res.results is not None:
                df = res.results if isinstance(res.results, pd.DataFrame) else None
            elif hasattr(res, "to_df"):
                df = res.to_df()
            if df is not None and len(df) > 0:
                # Normalize column names to Title case like yfinance
                cols = {c: c.title() for c in df.columns}
                df = df.rename(columns=cols)
                if "Close" in df.columns:
                    prov = FieldProvenance(provider="openbb", fetched_at=_now_iso())
                    _cache_set(cache_key, (df, prov))
                    return df, prov
        except Exception as e:
            log.warning("openbb history failed for %s: %s", symbol, e)

    # 2) stooq via pandas-datareader
    if pdr is not None:
        try:
            stooq_sym = _stooq_symbol(symbol)
            df = pdr.DataReader(stooq_sym, "stooq", start, end)
            if df is not None and len(df) > 0:
                # stooq comes sorted desc; normalize
                df = df.sort_index()
                # standardize columns to Title case if needed
                cols = {c: c.title() for c in df.columns}
                df = df.rename(columns=cols)
                if "Close" in df.columns:
                    prov = FieldProvenance(provider="stooq", fetched_at=_now_iso(), note=f"symbol={stooq_sym}")
                    _cache_set(cache_key, (df, prov))
                    return df, prov
        except Exception as e:
            log.warning("stooq history failed for %s: %s", symbol, e)

    # 3) yahoo_fin
    if yfin_si is not None:
        try:
            df = yfin_si.get_data(symbol, start_date=start.isoformat(), end_date=end.isoformat(), index_as_date=True)
            if df is not None and len(df) > 0:
                # yahoo_fin uses lowercase columns
                rename = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "adjclose": "Adj Close",
                    "volume": "Volume",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                prov = FieldProvenance(provider="yahoo_fin", fetched_at=_now_iso())
                _cache_set(cache_key, (df, prov))
                return df, prov
        except Exception as e:
            log.warning("yahoo_fin history failed for %s: %s", symbol, e)

    # empty
    prov = FieldProvenance(provider="none", fetched_at=_now_iso(), note="no_history")
    empty_df = pd.DataFrame()
    _cache_set(cache_key, (empty_df, prov))
    return empty_df, prov


def get_quote(symbol: str) -> Tuple[Dict[str, Any], FieldProvenance]:
    """
    Best-effort quote metadata. Prefer yahooquery, fallback to yfinance.
    """
    cache_key = f"quote:{symbol}"
    cached = _cache_get(cache_key)
    if cached:
        out, prov = cached
        return out, prov

    if YQTicker is not None:
        try:
            t = YQTicker(symbol)
            # Use price + quoteType + summaryDetail where available
            out: Dict[str, Any] = {}
            price = (t.price or {}).get(symbol) if hasattr(t, "price") else None
            qtype = (t.quote_type or {}).get(symbol) if hasattr(t, "quote_type") else None
            sdet = (t.summary_detail or {}).get(symbol) if hasattr(t, "summary_detail") else None
            if isinstance(price, dict):
                out.update(price)
            if isinstance(qtype, dict):
                out.update(qtype)
            if isinstance(sdet, dict):
                out.update(sdet)
            if out:
                prov = FieldProvenance(provider="yahooquery", fetched_at=_now_iso())
                _cache_set(cache_key, (out, prov))
                return out, prov
        except Exception as e:
            log.warning("yahooquery quote failed for %s: %s", symbol, e)

    if yf is not None:
        try:
            info = yf.Ticker(symbol).info
            if isinstance(info, dict) and info:
                prov = FieldProvenance(provider="yfinance", fetched_at=_now_iso())
                _cache_set(cache_key, (info, prov))
                return info, prov
        except Exception as e:
            log.warning("yfinance info failed for %s: %s", symbol, e)

    prov = FieldProvenance(provider="none", fetched_at=_now_iso(), note="no_quote")
    _cache_set(cache_key, ({}, prov))
    return {}, prov


def get_dividends(symbol: str, start: date, end: date) -> Tuple[pd.Series, FieldProvenance]:
    """
    Best-effort dividends time series (payment events). Used mainly for validation,
    not as the authoritative cashflow source (transactions remain authoritative).
    """
    cache_key = f"div:{symbol}:{start.isoformat()}:{end.isoformat()}"
    cached = _cache_get(cache_key)
    if cached:
        s, prov = cached
        return s, prov

    # 0) openbb dividends
    if obb is not None:
        try:
            res = obb.equity.dividends(symbol=symbol, start_date=start.isoformat(), end_date=end.isoformat())
            div = None
            if hasattr(res, "results") and res.results is not None:
                df = res.results if isinstance(res.results, pd.DataFrame) else None
                if df is not None and len(df) > 0:
                    # Try common column names
                    for col in ["dividend", "Dividends", "amount", "value", "cash_dividend"]:
                        if col in df.columns:
                            div = df[col]
                            div.index = pd.to_datetime(df.index if df.index.name else df.get("date", df.index))
                            break
            if isinstance(div, pd.Series) and len(div) > 0:
                div = div[(div.index.date >= start) & (div.index.date <= end)]
                prov = FieldProvenance(provider="openbb", fetched_at=_now_iso())
                _cache_set(cache_key, (div, prov))
                return div, prov
        except Exception as e:
            log.warning("openbb dividends failed for %s: %s", symbol, e)

    if yf is not None:
        try:
            div = yf.Ticker(symbol).dividends
            if div is not None and len(div) > 0:
                div = div[(div.index.date >= start) & (div.index.date <= end)]
                prov = FieldProvenance(provider="yfinance", fetched_at=_now_iso())
                _cache_set(cache_key, (div, prov))
                return div, prov
        except Exception as e:
            log.warning("yfinance dividends failed for %s: %s", symbol, e)

    if YQTicker is not None:
        try:
            t = YQTicker(symbol)
            dh = t.dividend_history
            if isinstance(dh, pd.DataFrame) and len(dh) > 0:
                # yahooquery returns a DF; try common column names
                for col in ["dividend", "Dividends", "amount", "value"]:
                    if col in dh.columns:
                        s = dh[col]
                        s.index = pd.to_datetime(dh.index)
                        s = s[(s.index.date >= start) & (s.index.date <= end)]
                        prov = FieldProvenance(provider="yahooquery", fetched_at=_now_iso())
                        _cache_set(cache_key, (s, prov))
                        return s, prov
        except Exception as e:
            log.warning("yahooquery dividends failed for %s: %s", symbol, e)

    prov = FieldProvenance(provider="none", fetched_at=_now_iso(), note="no_dividends")
    empty = pd.Series(dtype=float)
    _cache_set(cache_key, (empty, prov))
    return empty, prov
