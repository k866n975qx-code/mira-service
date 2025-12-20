#!/usr/bin/env python3
"""
Backfill and persist daily portfolio snapshots for a plaid_account_id using the DB.

Finds the earliest LMTransaction date for the account and walks day-by-day to today,
calling the existing holdings snapshot builder (refresh=True so it recomputes).
Snapshots are persisted to docs/portfolio/snapshots via the existing helper.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

# Ensure repo root on sys.path before importing app.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.api.routes.holdings import get_valued_holdings_for_plaid_account  # noqa: E402
from app.infra.db import SessionLocal  # noqa: E402
from app.infra.models import LMTransaction  # noqa: E402


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill daily portfolio snapshots.")
    parser.add_argument("--plaid-account-id", type=int, default=317631, help="Plaid account id (default: 317631)")
    parser.add_argument(
        "--start",
        type=str,
        help="Optional start date (YYYY-MM-DD). Defaults to earliest transaction date.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Optional end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="If set, do not force refresh (will reuse caches); default is to refresh.",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        # determine start date
        if args.start:
            start_date = date.fromisoformat(args.start)
        else:
            row = (
                db.query(LMTransaction)
                .filter(LMTransaction.plaid_account_id == args.plaid_account_id)
                .order_by(LMTransaction.date.asc())
                .first()
            )
            if not row or not row.date:
                raise SystemExit("No transactions found for this plaid_account_id.")
            start_date = row.date

        end_date = date.fromisoformat(args.end) if args.end else date.today()

        print(f"[backfill] plaid_account_id={args.plaid_account_id} start={start_date} end={end_date} refresh={not args.skip_refresh}")
        count = 0
        for d in _daterange(start_date, end_date):
            try:
                get_valued_holdings_for_plaid_account(
                    plaid_account_id=args.plaid_account_id,
                    as_of=d,
                    refresh=not args.skip_refresh,
                    slim=True,
                    db=db,
                )
                count += 1
                if count % 10 == 0:
                    print(f"  processed {count} days through {d}")
            except Exception as e:
                print(f"  failed for {d}: {e}")
                continue

        print(f"[backfill] completed: {count} snapshots persisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
