#!/usr/bin/env python3
"""
Benchmark holdings snapshot load time using live services + your DB.

Usage:
  python scripts/bench_snapshot.py [--plaid-account-id 123456] [--as-of YYYY-MM-DD] [--refresh-first]

Reads DB and Lunch Money settings from your normal .env via app.infra.settings.
Runs twice to show cache effect: first run uses --refresh-first flag (default True),
second run always uses cached mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import Optional

# Ensure repo root is on sys.path when run as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.api.routes.holdings import get_valued_holdings_for_plaid_account
from app.infra.db import SessionLocal
from app.infra.models import LMTransaction
from app.infra.settings import settings  # noqa: F401  # ensure .env is loaded


def _pick_plaid_account_id(db, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    rows = (
        db.query(LMTransaction.plaid_account_id)
        .filter(LMTransaction.plaid_account_id != None)  # noqa: E711
        .distinct()
        .order_by(LMTransaction.plaid_account_id.asc())
        .all()
    )
    if not rows:
        raise SystemExit("No plaid_account_id found in lm_transactions; sync first.")
    return int(rows[0][0])


def _run_once(db, pid: int, as_of_date: date, refresh: bool) -> dict:
    start = time.perf_counter()
    snap = get_valued_holdings_for_plaid_account(
        plaid_account_id=pid,
        as_of=as_of_date,
        refresh=refresh,
        goal_monthly=2000.0,
        symbols=None,
        apr_current_pct=None,
        apr_future_pct=None,
        apr_future_date=None,
        margin_mode=None,
        enrich=True,
        perf=True,
        perf_method="accurate",
        slim=False,
        db=db,
    )
    elapsed = time.perf_counter() - start
    totals = snap.get("totals") or {}
    meta = snap.get("meta") or {}
    return {
        "elapsed_sec": round(elapsed, 3),
        "holdings": len(snap.get("holdings") or []),
        "market_value": totals.get("market_value"),
        "served_from": meta.get("served_from"),
        "cache_origin": (meta.get("cache") or {}).get("origin") or meta.get("cache_origin"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark holdings snapshot load time.")
    parser.add_argument("--plaid-account-id", type=int, help="Plaid account id to benchmark. Defaults to first found in DB.")
    parser.add_argument("--as-of", type=str, help="Optional as-of date YYYY-MM-DD. Defaults to today.")
    parser.add_argument(
        "--refresh-first",
        action="store_true",
        default=True,
        help="Force the first run to bypass price cache (default: True).",
    )
    parser.add_argument(
        "--no-refresh-first",
        action="store_false",
        dest="refresh_first",
        help="Do not bypass caches on the first run.",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    with SessionLocal() as db:
        pid = _pick_plaid_account_id(db, args.plaid_account_id)
        print(f"[bench] plaid_account_id={pid} as_of={as_of.isoformat()} refresh_first={args.refresh_first}")

        first = _run_once(db, pid, as_of, refresh=args.refresh_first)
        print(f"[first ] {first['elapsed_sec']}s holdings={first['holdings']} mv={first['market_value']} served_from={first['served_from']} cache_origin={first['cache_origin']}")

        second = _run_once(db, pid, as_of, refresh=False)
        print(f"[second] {second['elapsed_sec']}s holdings={second['holdings']} mv={second['market_value']} served_from={second['served_from']} cache_origin={second['cache_origin']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
