# app/services/tx_parser.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional


Direction = Literal["buy", "sell"]


@dataclass
class ParsedInvestmentTx:
    symbol: str
    shares: float
    direction: Direction  # "buy" or "sell"


BUY_SELL_REGEX = re.compile(
    r"(?P<shares>\d+(\.\d+)?)\s+shares of\s+(?P<symbol>[A-Za-z0-9\-.]+)\s+"
    r"(?P<verb>purchased|bought|sold)\.",
    re.IGNORECASE,
)


def _parse_plaid_metadata(raw: Any) -> dict[str, Any]:
    """
    Lunch Money sends plaid_metadata as a JSON string.
    If parsing fails, just return {} and fall back to payee heuristics.
    """
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def parse_investment_tx(tx: dict[str, Any]) -> Optional[ParsedInvestmentTx]:
    """
    Given a raw Lunch Money transaction dict, try to interpret it as an
    investment BUY or SELL and return ParsedInvestmentTx.

    - Uses payee text like:
        "3.49046 shares of VNQ sold. - SOLD"
        "0.61976 shares of O purchased. - PURCHASED"
    - Ignores dividends / pure cash events for holdings purposes.
    """

    payee: str = (
        tx.get("payee")
        or tx.get("original_name")
        or tx.get("display_name")
        or ""
    )
    payee_lower = payee.lower()

    # Skip obvious non-holdings events (dividends, pure cash)
    if "dividend" in payee_lower:
        return None

    # Parse plaid_metadata if present (may contain subtype = buy/sell/cash/etc.)
    pm = _parse_plaid_metadata(tx.get("plaid_metadata"))
    pm_subtype = (pm.get("subtype") or pm.get("type") or "").lower()

    # Determine direction (buy/sell)
    direction: Optional[Direction] = None
    if pm_subtype in ("buy", "sell"):
        direction = pm_subtype  # type: ignore[assignment]
    else:
        # Fallback: infer from payee text
        if " sold." in payee_lower:
            direction = "sell"
        elif " purchased." in payee_lower or " bought." in payee_lower:
            direction = "buy"

    if direction is None:
        # Not a buy/sell we know how to parse
        return None

    # Extract shares + symbol from payee
    m = BUY_SELL_REGEX.search(payee)
    if not m:
        return None

    try:
        shares = float(m.group("shares"))
    except ValueError:
        return None

    symbol = m.group("symbol").upper()

    return ParsedInvestmentTx(symbol=symbol, shares=shares, direction=direction)