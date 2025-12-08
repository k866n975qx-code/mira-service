from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

# In-process TTL cache for latest prices
_LATEST_PRICE_CACHE: Dict[str, Dict[str, Any]] = {}


def _price_cache_ttl_seconds() -> int:
    try:
        return int(os.getenv("MIRA_PRICE_CACHE_TTL_SECONDS", "3600"))
    except Exception:
        return 3600


def get_latest_prices(symbols: List[str], bypass_cache: bool = False) -> Dict[str, float]:
    """
    Fetch latest close prices for a list of ticker symbols using yfinance.
    """
    prices: Dict[str, float] = {}
    now = datetime.utcnow()
    ttl = _price_cache_ttl_seconds()

    # cache first
    misses: List[str] = []
    for sym in symbols:
        if not bypass_cache:
            entry = _LATEST_PRICE_CACHE.get(sym)
            if entry and entry.get("expires_at") and entry["expires_at"] > now:
                prices[sym] = entry["price"]
                continue
        misses.append(sym)

    # batch download for remaining symbols
    if len(misses) >= 2:
        try:
            df = yf.download(
                misses,
                period="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
                close_df = df["Close"]
                for sym in close_df.columns:
                    try:
                        series = pd.to_numeric(close_df[sym], errors="coerce").dropna()
                        if series.empty:
                            continue
                        val = float(series.iloc[-1])
                        prices[sym] = val
                        _LATEST_PRICE_CACHE[sym] = {
                            "price": val,
                            "expires_at": now + timedelta(seconds=ttl),
                        }
                        if sym in misses:
                            misses.remove(sym)
                    except Exception:
                        continue
        except Exception:
            pass

    # fetch any remaining individually
    for sym in list(misses):
        try:
            hist = yf.Ticker(sym).history(period="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                val = float(hist["Close"].iloc[-1])
                prices[sym] = val
                _LATEST_PRICE_CACHE[sym] = {
                    "price": val,
                    "expires_at": now + timedelta(seconds=ttl),
                }
        except Exception:
            continue

    return prices


def get_daily_prices(symbols: List[str], days: int = 90) -> Dict[str, Dict[str, float]]:
    """
    Fetch daily close prices for the past N days for each symbol.
    Returns {SYM: {YYYY-MM-DD: close}}.
    """
    if not symbols or days <= 0:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    try:
        df = yf.download(
            symbols,
            period=f"{int(days)}d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        return result

    if isinstance(df.columns, pd.MultiIndex):
        try:
            close_df = df["Close"]
            for sym in close_df.columns:
                series = pd.to_numeric(close_df[sym], errors="coerce").dropna()
                if series.empty:
                    result[sym] = {}
                    continue
                daily: Dict[str, float] = {}
                for idx, val in series.items():
                    try:
                        daily[idx.date().isoformat()] = float(val)
                    except Exception:
                        continue
                result[sym] = daily
        except Exception:
            return result
    else:
        try:
            series = pd.to_numeric(df["Close"], errors="coerce").dropna()
            daily: Dict[str, float] = {}
            for idx, val in series.items():
                try:
                    daily[idx.date().isoformat()] = float(val)
                except Exception:
                    continue
            if symbols:
                result[symbols[0]] = daily
        except Exception:
            return result

    return result


def get_price_history(symbols: List[str], start: date, end: date) -> Dict[str, Dict[str, float]]:
    """
    Return daily close prices for each symbol between start and end (inclusive).
    """
    if not symbols:
        return {}

    start_str = start.isoformat()
    end_str = end.isoformat()

    result: Dict[str, Dict[str, float]] = {}

    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(start=start_str, end=end_str)
        except Exception:
            result[sym] = {}
            continue

        if hist.empty:
            result[sym] = {}
            continue

        daily_prices: Dict[str, float] = {}
        for idx, row in hist.iterrows():
            day_str = idx.date().isoformat()
            close = row.get("Close")
            if close is None:
                continue
            daily_prices[day_str] = float(close)

        result[sym] = daily_prices

    return result
