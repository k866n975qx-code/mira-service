from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

_SECTOR_KEYS = [
    "realestate",
    "consumer_cyclical",
    "basic_materials",
    "consumer_defensive",
    "technology",
    "communication_services",
    "financial_services",
    "utilities",
    "industrials",
    "energy",
    "healthcare",
]


def _norm_symbol(sym: Optional[str]) -> str:
    if not sym:
        return ""
    s = sym.upper().replace("/", ".").replace("-", ".")
    s = s.replace("BRK.B", "BRK.B").replace("BRK-B", "BRK.B")
    return s.strip()


def _merge_alphabet(sym: str) -> str:
    s = _norm_symbol(sym)
    return "ALPHABET (A+C)" if s in ("GOOG", "GOOGL") else s


def compute_lookthrough(holdings: List[Dict]) -> Dict:
    # coverage
    total_w = 0.0
    mapped_w = 0.0
    unmapped: List[str] = []
    for h in holdings:
        w = float(h.get("weight_pct") or 0.0)
        total_w += w
        if h.get("ultimate"):
            mapped_w += w
        else:
            sym = h.get("symbol")
            if sym:
                unmapped.append(sym)

    coverage_by_weight = round(mapped_w, 3)
    coverage_by_count = round(
        (sum(1 for h in holdings if h.get("ultimate")) / max(1, len(holdings))) * 100.0,
        3,
    )
    unknown_unmapped_pct = round(max(0.0, 100.0 - coverage_by_weight), 3)

    # sector mix (portfolio percent)
    sectors: DefaultDict[str, float] = defaultdict(float)
    for h in holdings:
        w = float(h.get("weight_pct") or 0.0)
        sw = (h.get("ultimate") or {}).get("sector_weights") or {}
        if not sw:
            continue
        for k in _SECTOR_KEYS:
            v = float(sw.get(k) or 0.0) / 100.0
            sectors[k] += w * v

    sectors_out = {k: round(v, 3) for k, v in sectors.items()}

    # top positions (portfolio percent)
    top: DefaultDict[str, float] = defaultdict(float)
    for h in holdings:
        w = float(h.get("weight_pct") or 0.0)
        tops = (h.get("ultimate") or {}).get("top_holdings") or []
        for t in tops:
            sym = _merge_alphabet(t.get("symbol") or "")
            if not sym or sym == _norm_symbol(h.get("symbol")):
                continue  # ignore self-holdings/wrappers
            wt_in_fund = float(t.get("weight_pct") or 0.0) / 100.0
            top[sym] += w * wt_in_fund

    top_sorted: List[Tuple[str, float]] = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    top_out = [
        {"symbol": s, "portfolio_weight_pct": round(v, 3)}
        for s, v in top_sorted[:25]
    ]

    return {
        "coverage": {
            "by_weight_pct": coverage_by_weight,
            "by_count_pct": coverage_by_count,
            "unknown_unmapped_pct": unknown_unmapped_pct,
            "unmapped_symbols": unmapped,
        },
        "sector_weights_portfolio_pct": sectors_out,
        "top_positions": top_out,
    }
