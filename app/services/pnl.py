from __future__ import annotations

from typing import Any, Dict, List


def compute_unrealized_pnl(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of holdings dicts with at least:
      - symbol
      - shares
      - cost_basis
      - last_price
      - market_value

    Compute per-symbol and aggregate unrealized P&L.

    Returns:
      {
        "by_symbol": {
          "VTI": {
             "symbol": "VTI",
             "shares": ...,
             "cost_basis": ...,
             "last_price": ...,
             "market_value": ...,
             "unrealized_pnl": ...,
             "unrealized_pct": ...
          },
          ...
        },
        "totals": {
          "cost_basis": ...,
          "market_value": ...,
          "unrealized_pnl": ...,
          "unrealized_pct": ...
        }
      }
    """
    by_symbol: Dict[str, Dict[str, Any]] = {}
    total_cost = 0.0
    total_value = 0.0

    for h in holdings:
        symbol = h.get("symbol")
        shares = float(h.get("shares") or 0.0)
        cost_basis = float(h.get("cost_basis") or 0.0)
        last_price = h.get("last_price")
        market_value = h.get("market_value")

        if last_price is None or market_value is None:
            pnl = None
            pnl_pct = None
        else:
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else None
            total_cost += cost_basis
            total_value += market_value

        by_symbol[symbol] = {
            "symbol": symbol,
            "shares": shares,
            "cost_basis": round(cost_basis, 2),
            "last_price": last_price,
            "market_value": market_value,
            "unrealized_pnl": round(pnl, 2) if pnl is not None else None,
            "unrealized_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        }

    totals: Dict[str, Any] = {
        "cost_basis": round(total_cost, 2),
        "market_value": round(total_value, 2),
        "unrealized_pnl": round(total_value - total_cost, 2),
    }

    if total_cost:
        totals["unrealized_pct"] = round(
            (totals["unrealized_pnl"] / total_cost) * 100.0, 2
        )
    else:
        totals["unrealized_pct"] = None

    return {
        "by_symbol": by_symbol,
        "totals": totals,
    }