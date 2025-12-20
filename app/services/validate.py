
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math


def validate_snapshot(snapshot: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    holdings = snapshot.get("holdings") or []
    if not isinstance(holdings, list):
        return ["holdings is not a list"]

    # Totals vs holdings
    mv_sum = 0.0
    cb_sum = 0.0
    w_sum = 0.0
    for h in holdings:
        if not isinstance(h, dict):
            continue
        mv = float(h.get("market_value") or 0.0)
        cb = float(h.get("cost_basis") or 0.0)
        mv_sum += mv
        cb_sum += cb
        w_sum += float(h.get("weight_pct") or 0.0)

        # yield consistency: current_yield ~= forward_12m_dividend / market_value
        f12 = h.get("forward_12m_dividend")
        cy = h.get("current_yield_pct")
        if f12 is not None and cy is not None and mv > 0:
            implied = (float(f12) / mv) * 100.0
            if abs(implied - float(cy)) > 0.35:  # 35 bps tolerance
                warnings.append(f"{h.get('symbol')}: current_yield_pct mismatch (implied {implied:.3f} vs {float(cy):.3f})")

        # ultimate top holdings sanity: single holding > 100% indicates levered/derivatives; flag
        ult = h.get("ultimate") or {}
        if isinstance(ult, dict):
            th = ult.get("top_holdings")
            if isinstance(th, list):
                for item in th:
                    if isinstance(item, dict) and float(item.get("weight_pct") or 0.0) > 100.0:
                        warnings.append(f"{h.get('symbol')}: top_holding {item.get('symbol')} weight_pct > 100% (derivatives/leveraged?)")

    totals = snapshot.get("totals") or {}
    if isinstance(totals, dict):
        mv = float(totals.get("market_value") or 0.0)
        cb = float(totals.get("cost_basis") or 0.0)
        if mv and abs(mv - mv_sum) > max(0.01 * mv, 1.0):
            warnings.append(f"totals.market_value mismatch vs holdings sum ({mv:.2f} vs {mv_sum:.2f})")
        if cb and abs(cb - cb_sum) > max(0.01 * cb, 1.0):
            warnings.append(f"totals.cost_basis mismatch vs holdings sum ({cb:.2f} vs {cb_sum:.2f})")

    if w_sum and abs(w_sum - 100.0) > 0.75:
        warnings.append(f"weights do not sum to ~100% (sum={w_sum:.3f})")

    # Dividend rollup sanity
    div = snapshot.get("dividends") or {}
    if isinstance(div, dict):
        win = (div.get("windows") or {}).get("30d") if isinstance(div.get("windows"), dict) else None
        realized = div.get("realized_mtd") or {}
        if isinstance(win, dict) and isinstance(realized, dict):
            w_total = float(win.get("total_dividends") or 0.0)
            r_total = float(realized.get("total_dividends") or 0.0)
            if abs(w_total - r_total) > 0.01:
                warnings.append(f"dividends.realized_mtd.total_dividends differs from dividends.windows.30d.total_dividends ({r_total:.2f} vs {w_total:.2f})")

    return warnings
