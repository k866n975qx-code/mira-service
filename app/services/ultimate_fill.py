from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.services.deterministic_metrics import compute_dividend_fields, compute_risk_metrics_validated, fetch_price_history
from app.services.market_data import get_quote
from app.services.provenance import missing_entry, prov_entry


def fill_ultimate_nulls(
    symbol: str,
    ultimate: Dict[str, Any],
    benchmark_symbol: str = "^GSPC",
    as_of: date | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Fill missing/None ultimate values using deterministic fallbacks:
      - Prices/dividends: yfinance -> yahooquery -> openbb/market_data
      - Risk validation: FinanceToolkit -> vectorbt -> QuantLib
    Returns (ultimate, provenance_by_field, conflict_paths).
    """
    as_of = as_of or date.today()
    prov: Dict[str, Any] = {}
    conflicts: List[str] = []

    # --- Identity via quote metadata (yahooquery/yfinance) ---
    quote, _ = get_quote(symbol)
    identity_map = {
        "name": ["shortName", "longName", "displayName"],
        "exchange": ["exchange", "fullExchangeName"],
        "currency": ["currency"],
        "category": ["category", "categoryName", "fundCategory"],
    }
    for out_key, in_keys in identity_map.items():
        if ultimate.get(out_key) is None:
            for k in in_keys:
                v = quote.get(k)
                if v is not None:
                    ultimate[out_key] = v
                    prov[out_key] = prov_entry("yahooquery", "pulled", "quote_identity", [symbol])
                    break
            if ultimate.get(out_key) is None:
                prov[out_key] = missing_entry("yahooquery", "quote_identity", [symbol], note="quote metadata missing")

    # --- Price history (primary yfinance/yahooquery) ---
    start_3y_dt = as_of - pd.Timedelta(days=365 * 3 + 30)
    start_3y = start_3y_dt.date() if hasattr(start_3y_dt, "date") else start_3y_dt
    close, price_prov = fetch_price_history(symbol, start=start_3y, end=as_of)
    bench_close, bench_prov = fetch_price_history(benchmark_symbol, start=start_3y, end=as_of)
    prov["price_history"] = price_prov
    prov["benchmark_history"] = bench_prov
    last_price = None
    try:
        last_price = float(close.dropna().iloc[-1])
    except Exception:
        last_price = None
    if last_price is not None:
        prov["last_price"] = prov_entry(price_prov.get("provider", "yfinance"), "pulled", "price_history_last", [symbol])
        prov["_last_price_value"] = last_price

    # --- Dividends: Yahoo pull, OpenBB validation/repair ---
    div_fields, div_prov, div_conflicts = compute_dividend_fields(symbol, as_of, last_price)
    conflicts.extend(div_conflicts)
    for k, v in div_fields.items():
        if ultimate.get(k) is None:
            ultimate[k] = v
    prov.update(div_prov)

    if ultimate.get("forward_yield_pct") is not None and ultimate.get("forward_yield_method") is None:
        ultimate["forward_yield_method"] = "indicated" if "forward_12m_div_ps" in div_fields else "t12m"
        prov["forward_yield_method"] = prov_entry(
            price_prov.get("provider", "internal"),
            "derived",
            "yield_method_flag",
            [symbol],
            note="set alongside forward_yield_pct",
        )

    # --- Risk/performance with validators (FinanceToolkit/vectorbt/QuantLib) ---
    risk_metrics, risk_prov, risk_conflicts = compute_risk_metrics_validated(
        symbol=symbol,
        close=close,
        bench_close=bench_close,
        existing=ultimate,
        provider=price_prov.get("provider", "yfinance"),
    )
    conflicts.extend(risk_conflicts)
    for k, v in risk_metrics.items():
        if ultimate.get(k) is None:
            ultimate[k] = v
    for k, entry in risk_prov.items():
        if k not in prov:
            prov[k] = entry

    # Track derived fields
    derived = set(ultimate.get("derived_fields") or [])
    for k, v in prov.items():
        if isinstance(v, dict) and v.get("source_type") in ("derived", "validated", "conflict"):
            derived.add(k)
    ultimate["derived_fields"] = sorted(derived)

    return ultimate, prov, conflicts
