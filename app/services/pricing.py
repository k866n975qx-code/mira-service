from __future__ import annotations

from typing import Dict, List

import yfinance as yf


def get_latest_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch latest close prices for a list of ticker symbols using yfinance.
    Returns a dict: { "JEPI": 56.12, "SCHD": 31.45, ... }
    """
    prices: Dict[str, float] = {}

    # yfinance can batch-request, but we'll keep it simple + robust for now
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                last_close = float(hist["Close"].iloc[-1])
                prices[symbol] = last_close
        except Exception:
            # If anything blows up for a symbol, just skip it for now
            continue

    return prices
from datetime import date
from typing import Dict, List
import yfinance as yf


def get_price_history(
    symbols: List[str],
    start: date,
    end: date,
) -> Dict[str, Dict[str, float]]:
    """
    Return daily close prices for each symbol between start and end (inclusive).

    Output shape:
    {
        "VTI": {
            "2025-11-01": 250.12,
            "2025-11-02": 251.34,
            ...
        },
        "JEPI": { ... },
        ...
    }
    """
    if not symbols:
        return {}

    # yfinance wants strings
    start_str = start.isoformat()
    end_str = end.isoformat()

    result: Dict[str, Dict[str, float]] = {}

    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(start=start_str, end=end_str)
        except Exception:
            # if yfinance explodes for a ticker, just skip it
            result[sym] = {}
            continue

        if hist.empty:
            result[sym] = {}
            continue

        # hist.index is a DatetimeIndex; we map to YYYY-MM-DD -> close
        daily_prices: Dict[str, float] = {}
        for idx, row in hist.iterrows():
            day_str = idx.date().isoformat()
            close = row.get("Close")
            if close is None:
                continue
            daily_prices[day_str] = float(close)

        result[sym] = daily_prices

    return result