from datetime import date

from app.services.snapshot_normalizer import normalize_snapshot


def test_normalize_snapshot_restores_root_keys_and_rollups():
    raw = {
        "as_of": "2025-12-18",
        "holdings": [
            {
                "symbol": "AAA",
                "market_value": 1000.0,
                "cost_basis": 900.0,
                "projected_monthly_dividend": 10.0,
                "forward_12m_dividend": 120.0,
                "ultimate": {
                    "twr_1m_pct": 2.0,
                    "twr_3m_pct": 3.0,
                    "twr_6m_pct": 4.0,
                    "twr_12m_pct": 5.0,
                    "vol_90d_pct": 10.0,
                    "max_drawdown_1y_pct": -20.0,
                    "beta_3y": 0.7,
                },
            },
            {
                "symbol": "BBB",
                "market_value": 2000.0,
                "cost_basis": 1900.0,
                "projected_monthly_dividend": 20.0,
                "forward_12m_dividend": 180.0,
                "ultimate": {
                    "twr_1m_pct": 1.5,
                    "twr_3m_pct": 2.5,
                    "twr_6m_pct": 3.5,
                    "twr_12m_pct": 4.5,
                    "vol_90d_pct": 12.0,
                    "max_drawdown_1y_pct": -18.0,
                    "beta_3y": 0.8,
                },
            },
        ],
        "totals": {},
        "income": {},
        "portfolio_rollups": {"performance": {}, "risk": {}},
        "meta": {"snapshot_created_at": date(2025, 12, 17).isoformat()},
    }

    snap = normalize_snapshot(raw)

    assert snap["total_market_value"] == snap["totals"]["market_value"]
    assert "margin_loan_balance" not in snap
    assert "margin_to_portfolio_pct" not in snap

    rollups = snap["portfolio_rollups"]
    assert rollups["performance"]
    assert rollups["risk"]
    assert rollups["benchmark"] == "^GSPC"
    assert rollups["meta"].get("version")

    meta = snap["meta"]
    assert "warnings" not in meta
    assert meta.get("notes")
