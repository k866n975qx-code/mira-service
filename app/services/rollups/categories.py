from __future__ import annotations

from typing import Dict, List


def compute_category_exposure(holdings: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate weight and income share by category using holding.weight_pct and forward_12m_dividend.
    Category is pulled from ultimate.category or holding.category.
    """
    out: Dict[str, Dict[str, float]] = {}
    total_income = sum(float(h.get("forward_12m_dividend") or 0.0) for h in holdings)
    for h in holdings:
        cat = (h.get("ultimate") or {}).get("category") or h.get("category") or "Uncategorized"
        w = float(h.get("weight_pct") or 0.0)
        inc = float(h.get("forward_12m_dividend") or 0.0)
        row = out.setdefault(cat, {"weight_pct": 0.0, "income_pct": 0.0})
        row["weight_pct"] += w
        if total_income > 0:
            row["income_pct"] += (inc / total_income) * 100.0
    return {k: {"weight_pct": round(v["weight_pct"], 3), "income_pct": round(v["income_pct"], 3)} for k, v in out.items()}
