from datetime import date

from app.services.portfolio import compute_portfolio_rollups


def test_rollups_shape_minimal():
    holdings = [
        {
            "symbol": "JEPI",
            "shares": 10,
            "market_value": 574.1,
            "cost_basis": 560.0,
            "forward_12m_dividend": 45.0,
            "ultimate": {
                "twr_1m_pct": 2.0,
                "twr_3m_pct": 3.0,
                "twr_6m_pct": 4.0,
                "twr_12m_pct": 5.0,
            },
        },
        {
            "symbol": "JEPQ",
            "shares": 10,
            "market_value": 579.1,
            "cost_basis": 580.0,
            "forward_12m_dividend": 60.0,
            "ultimate": {
                "twr_1m_pct": 3.0,
                "twr_3m_pct": 4.0,
                "twr_6m_pct": 6.0,
                "twr_12m_pct": 9.0,
            },
        },
    ]
    roll = compute_portfolio_rollups(
        1, date(2025, 12, 18), holdings, perf_method="accurate", include_mwr=False
    )
    assert "income" in roll and "performance" in roll and "risk" in roll
    for k in [
        "twr_1m_pct",
        "twr_3m_pct",
        "twr_6m_pct",
        "twr_12m_pct",
        "benchmark_twr_12m_pct",
        "alpha_12m_pct",
        "twr_1m_approx_pct",
    ]:
        assert k in roll["performance"]
