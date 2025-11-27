# app/services/holdings.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List

from app.infra.models import LMTransaction
from app.services.tx_parser import ParsedInvestmentTx, parse_investment_tx

from sqlalchemy.orm import Session
from app.infra.models import HoldingSnapshot


@dataclass
class HoldingState:
    symbol: str
    shares: float = 0.0
    cost_basis: float = 0.0
    trades: int = 0


def build_holdings_from_transactions(
    txs,
    as_of=None,
) -> Dict[str, HoldingState]:
    """
    Build per-symbol holdings by walking all LMTransactions and interpreting
    ONLY investment buys/sells from plaid_metadata/payee.

    - Buys: shares += parsed.shares, cost_basis += cash outflow
    - Sells: shares -= parsed.shares, cost_basis -= cash inflow
    - Non-investment events (dividends, pure cash) are ignored.
    """
    # Sort by date/id for deterministic behavior
    sorted_txs: List[LMTransaction] = sorted(
        txs,
        key=lambda t: (t.date or date.min, t.id),
    )

    holdings: Dict[str, HoldingState] = {}

    for tx in sorted_txs:
        raw: Any = tx.raw or {}
        parsed: ParsedInvestmentTx | None = parse_investment_tx(raw)
        if parsed is None:
            # Not a buy/sell we care about (maybe dividend/cash)
            continue

        symbol = parsed.symbol
        direction = parsed.direction  # "buy" or "sell"

        # amount in LM is signed cash flow (buys positive, sells negative) on to_base
        amount_val = float(tx.amount or 0.0)
        cash_flow = abs(amount_val)

        # buys add shares + cost; sells subtract shares and cost
        if direction == "buy":
            delta_shares = parsed.shares
            delta_cost = cash_flow
        else:  # "sell"
            delta_shares = -parsed.shares
            # For v0, treat proceeds as negative cost_basis
            delta_cost = -cash_flow

        h = holdings.get(symbol)
        if h is None:
            h = HoldingState(symbol=symbol)
            holdings[symbol] = h

        h.shares += delta_shares
        h.cost_basis += delta_cost
        h.trades += 1

    # Drop anything that's effectively fully closed
    clean: Dict[str, HoldingState] = {}
    for sym, h in holdings.items():
        if abs(h.shares) < 1e-6:
            continue
        clean[sym] = h

    return clean


def reconstruct_holdings(
    txs: Iterable[LMTransaction],
    plaid_account_id: int,
    as_of: date,
) -> Dict[str, Any]:
    """
    High-level helper used by the /holdings endpoint.

    Assumes txs are already filtered for:
      - the given plaid_account_id
      - date <= as_of
    """
    holdings_map = build_holdings_from_transactions(txs)

    holdings_list = []
    for sym, h in sorted(holdings_map.items(), key=lambda kv: kv[0]):
        # Avoid negative cost_basis if we over-sell vs naive cost tracking
        cost_basis = h.cost_basis
        holdings_list.append(
            {
                "symbol": sym,
                "shares": h.shares,
                "cost_basis": cost_basis,
                "avg_cost": cost_basis / h.shares if h.shares else 0.0,
                "trades": h.trades,
            }
        )

    return {
        "as_of": as_of.isoformat(),
        "count": len(holdings_list),
        "holdings": holdings_list,
        "plaid_account_id": plaid_account_id,
    }

def persist_holding_snapshot(
    db: Session,
    plaid_account_id: int,
    snapshot: Dict[str, Any],
) -> HoldingSnapshot:
    """
    Upsert a daily snapshot for a given plaid_account_id into holding_snapshots.

    Expects `snapshot` in the same shape returned by
    /lm/holdings/{plaid_account_id}/snapshot, including an "as_of" key.
    """
    as_of_str = snapshot.get("as_of")
    if not as_of_str:
        # If somehow we get here without as_of, don't blow up the endpoint.
        # Just skip persistence.
        return None  # type: ignore[return-value]

    as_of_date = date.fromisoformat(as_of_str[:10])

    existing = (
        db.query(HoldingSnapshot)
        .filter(
            HoldingSnapshot.plaid_account_id == plaid_account_id,
            HoldingSnapshot.as_of_date == as_of_date,
        )
        .one_or_none()
    )

    if existing:
        existing.snapshot = snapshot
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    row = HoldingSnapshot(
        plaid_account_id=plaid_account_id,
        as_of_date=as_of_date,
        snapshot=snapshot,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row