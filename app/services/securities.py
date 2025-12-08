from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.infra.settings import settings
from app.infra.models import SecurityMapping

_STOPWORDS = {"OF", "DIVIDEND", "RECEIVED", "PAYMENT", "INTEREST"}


def should_overwrite_symbol(sym: Optional[str]) -> bool:
    if not sym:
        return True
    s = sym.strip().upper()
    return s in _STOPWORDS or len(s) <= 1


@lru_cache(maxsize=1)
def _load_cusip_catalog(path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    """
    Load a mapping { CUSIP -> (SYMBOL, DESCRIPTION) } from CSV.
    Expected headers: cusip,symbol,description
    """
    fn = path or "CUSIP.csv"
    fn = os.path.abspath(fn)
    catalog: Dict[str, Tuple[str, str]] = {}
    with open(fn, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = (row.get("cusip") or "").strip().upper()
            s = (row.get("symbol") or "").strip().upper()
            d = (row.get("description") or "").strip()
            if c and s:
                catalog[c] = (s, d)
    return catalog


def resolve_symbol_from_cusip(db: Session, cusip: Optional[str]) -> Optional[str]:
    """
    Try DB mapping first; fall back to CSV; persist DB mapping for future use.
    """
    if not cusip:
        return None
    c = cusip.strip().upper()

    # 1) DB cache
    try:
        m = (
            db.query(SecurityMapping)
            .filter(SecurityMapping.cusip == c, SecurityMapping.is_active == True)
            .one_or_none()
        )
        if m and m.ticker:
            return m.ticker.upper()
    except Exception:
        # don't block on DB lookup errors
        pass

    # 2) CSV fallback
    cat = _load_cusip_catalog(settings.cusip_csv_path)
    rec = cat.get(c)
    if not rec:
        return None

    ticker, name = rec[0], rec[1] if len(rec) > 1 else None

    # 3) Persist mapping for future requests (optional but helpful)
    try:
        obj = db.query(SecurityMapping).filter(SecurityMapping.cusip == c).one_or_none()
        if obj:
            # update if changed
            changed = False
            if obj.ticker != ticker:
                obj.ticker = ticker
                changed = True
            if name and obj.name != name:
                obj.name = name
                changed = True
            if not obj.is_active:
                obj.is_active = True
                changed = True
            if changed:
                db.add(obj)
                db.commit()
        else:
            obj = SecurityMapping(cusip=c, ticker=ticker, name=name, is_active=True)
            db.add(obj)
            db.commit()
    except Exception:
        db.rollback()  # best-effort

    return ticker
