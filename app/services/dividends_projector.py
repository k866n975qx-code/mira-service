from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

_DEFAULT_LAG_DAYS: Dict[str, int] = {}


def _infer_next_ex_date(h: Dict, as_of: date) -> Optional[date]:
    u = h.get("ultimate") or {}
    ex = u.get("next_ex_date_est") or h.get("last_ex_date")
    if not ex:
        return None
    try:
        return ex if isinstance(ex, date) else date.fromisoformat(str(ex))
    except Exception:
        return None


def _infer_pay_date(ex_date: date, symbol: str, freq: str) -> date:
    lag = _DEFAULT_LAG_DAYS.get(symbol, 10 if freq == "monthly" else 20)
    return ex_date + timedelta(days=lag)


def project_paydate_window(
    holdings: List[Dict], win_start: date, win_end: date
) -> Dict:
    events: List[Dict] = []
    total = 0.0
    for h in holdings:
        u = h.get("ultimate") or {}
        freq = (u.get("distribution_frequency") or "").lower()
        ex = _infer_next_ex_date(h, win_end)
        if not ex or not freq:
            continue
        pay = _infer_pay_date(ex, h.get("symbol", ""), freq)
        if win_start <= pay <= win_end:
            fwd = float(h.get("forward_12m_dividend") or 0.0)
            if "monthly" in freq:
                amt = fwd / 12.0
            elif "quarter" in freq:
                amt = fwd / 4.0
            else:
                amt = fwd / 12.0
            events.append(
                {
                    "symbol": h.get("symbol"),
                    "ex_date_est": ex.isoformat(),
                    "pay_date_est": pay.isoformat(),
                    "amount_est": round(amt, 2),
                }
            )
            total += amt
    return {
        "mode": "paydate",
        "window": {"start": win_start.isoformat(), "end": win_end.isoformat()},
        "projected": round(total, 3),
        "expected_events": events,
    }


def project_upcoming_exdates(holdings: List[Dict], start: date, end: date) -> Dict:
    events: List[Dict] = []
    total = 0.0
    for h in holdings:
        u = h.get("ultimate") or {}
        ex = u.get("next_ex_date_est")
        if not ex:
            continue
        try:
            ex_d = ex if isinstance(ex, date) else date.fromisoformat(str(ex))
        except Exception:
            continue
        if ex_d < start or ex_d > end:
            continue
        amt = h.get("projected_monthly_dividend")
        if amt is None:
            fwd = float(h.get("forward_12m_dividend") or 0.0)
            amt = fwd / 12.0
        amt = round(float(amt), 2)
        events.append(
            {
                "symbol": h.get("symbol"),
                "ex_date_est": ex_d.isoformat(),
                "amount_est": amt,
            }
        )
        total += amt
    events_sorted = sorted(events, key=lambda e: e.get("ex_date_est", ""))
    sum_events = round(sum(e.get("amount_est", 0.0) for e in events_sorted), 2)
    projected_total = round(total, 2)
    return {
        "window": {"mode": "exdate", "start": start.isoformat(), "end": end.isoformat()},
        "projected": projected_total,
        "events": events_sorted,
        "meta": {
            "sum_of_events": sum_events,
            "matches_projected": abs(sum_events - projected_total) < 0.005,
        },
    }
