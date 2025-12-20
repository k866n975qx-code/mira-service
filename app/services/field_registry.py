from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FieldSpec:
    path: str
    primary_source: str
    fallback_sources: List[str]
    validator: Optional[str]
    source_type: str  # pulled | derived | validated | missing
    method: Optional[str]
    required_inputs: List[str]


def get_field_registry() -> List[FieldSpec]:
    """
    Central registry of important snapshot fields with their sourcing plan.
    This is referenced for coverage accounting and future deterministic fallback wiring.
    """
    specs: List[FieldSpec] = []

    # Prices / returns inputs
    specs.append(
        FieldSpec(
            path="holdings[].last_price",
            primary_source="yfinance",
            fallback_sources=["yahooquery", "openbb"],
            validator="vectorbt",
            source_type="pulled",
            method="price_history_last",
            required_inputs=["price_history.symbol"],
        )
    )

    # Risk / performance (holdings)
    for field, method in [
        ("vol_30d_pct", "annualized_vol_window"),
        ("vol_90d_pct", "annualized_vol_window"),
        ("beta_3y", "beta_from_returns_3y"),
        ("corr_1y", "corr_from_returns_1y"),
        ("sharpe_1y", "sharpe_from_returns_1y"),
        ("sortino_1y", "sortino_from_returns_1y"),
        ("downside_dev_1y_pct", "downside_dev_from_returns_1y"),
        ("max_drawdown_1y_pct", "drawdown_from_close_1y"),
        ("drawdown_duration_1y_days", "drawdown_from_close_1y"),
        ("calmar_1y", "calmar_from_cagr_mdd"),
        ("var_95_1d_pct", "var_cvar_from_returns"),
        ("cvar_95_1d_pct", "var_cvar_from_returns"),
        ("twr_1m_pct", "twr_from_close_window"),
        ("twr_3m_pct", "twr_from_close_window"),
        ("twr_6m_pct", "twr_from_close_window"),
        ("twr_12m_pct", "twr_from_close_window"),
    ]:
        specs.append(
            FieldSpec(
                path=f"holdings[].ultimate.{field}",
                primary_source="internal",
                fallback_sources=["financetoolkit", "vectorbt"],
                validator="quantlib",
                source_type="derived",
                method=method,
                required_inputs=["price_history.symbol", "benchmark_history"],
            )
        )

    # Dividends / income (holdings)
    specs.extend(
        [
            FieldSpec(
                path="holdings[].ultimate.trailing_12m_div_ps",
                primary_source="yfinance",
                fallback_sources=["yahooquery", "openbb"],
                validator="financetoolkit",
                source_type="pulled",
                method="ttm_div_sum",
                required_inputs=["dividends_history.symbol"],
            ),
            FieldSpec(
                path="holdings[].last_ex_date",
                primary_source="yfinance",
                fallback_sources=["openbb"],
                validator="openbb",
                source_type="pulled",
                method="last_exdate_from_events",
                required_inputs=["dividends_history.symbol"],
            ),
            FieldSpec(
                path="holdings[].ultimate.distribution_frequency",
                primary_source="derived",
                fallback_sources=["openbb"],
                validator="openbb",
                source_type="derived",
                method="freq_from_exdate_gaps",
                required_inputs=["dividends_history.symbol"],
            ),
            FieldSpec(
                path="holdings[].ultimate.next_ex_date_est",
                primary_source="derived",
                fallback_sources=["openbb"],
                validator="openbb",
                source_type="derived",
                method="median_gap_projection",
                required_inputs=["dividends_history.symbol"],
            ),
            FieldSpec(
                path="holdings[].ultimate.trailing_12m_yield_pct",
                primary_source="derived",
                fallback_sources=["financetoolkit"],
                validator="financetoolkit",
                source_type="derived",
                method="ttm_yield",
                required_inputs=["holdings[].ultimate.trailing_12m_div_ps", "holdings[].last_price"],
            ),
            FieldSpec(
                path="holdings[].ultimate.forward_yield_pct",
                primary_source="derived",
                fallback_sources=["financetoolkit"],
                validator="financetoolkit",
                source_type="derived",
                method="forward_yield",
                required_inputs=["holdings[].ultimate.forward_yield_method", "holdings[].last_price"],
            ),
            FieldSpec(
                path="holdings[].forward_12m_dividend",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="forward_from_ttm_shares",
                required_inputs=["holdings[].ultimate.forward_12m_div_ps", "holdings[].shares"],
            ),
            FieldSpec(
                path="holdings[].projected_monthly_dividend",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="monthly_from_forward",
                required_inputs=["holdings[].forward_12m_dividend"],
            ),
        ]
    )

    # Metadata
    for meta_field in ("name", "exchange", "currency", "category"):
        specs.append(
            FieldSpec(
                path=f"holdings[].ultimate.{meta_field}",
                primary_source="yahooquery",
                fallback_sources=["yfinance"],
                validator="internal",
                source_type="pulled",
                method="quote_identity",
                required_inputs=[],
            )
        )

    # Portfolio rollups (performance & risk)
    for label in ("1m", "3m", "6m", "12m"):
        specs.append(
            FieldSpec(
                path=f"portfolio_rollups.performance.twr_{label}_pct",
                primary_source="vectorbt",
                fallback_sources=["financetoolkit", "internal"],
                validator="quantlib",
                source_type="derived",
                method="portfolio_twr",
                required_inputs=["holdings[].ultimate.twr_fields", "price_history"],
            )
        )
    for field in [
        "vol_30d_pct",
        "vol_90d_pct",
        "max_drawdown_1y_pct",
        "beta_portfolio",
        "downside_dev_1y_pct",
        "sharpe_1y",
        "calmar_1y",
        "drawdown_duration_1y_days",
        "var_95_1d_pct",
        "cvar_95_1d_pct",
        "corr_1y",
    ]:
        specs.append(
            FieldSpec(
                path=f"portfolio_rollups.risk.{field}",
                primary_source="internal",
                fallback_sources=["financetoolkit", "vectorbt"],
                validator="quantlib",
                source_type="derived",
                method="portfolio_risk_rollup",
                required_inputs=["holdings", "price_history"],
            )
        )

    # Income totals and dividend windows
    specs.extend(
        [
            FieldSpec(
                path="income.forward_12m_total",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="income_from_holdings",
                required_inputs=["holdings.forward_12m_dividend"],
            ),
            FieldSpec(
                path="income.projected_monthly_income",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="income_from_holdings",
                required_inputs=["holdings.projected_monthly_dividend"],
            ),
            FieldSpec(
                path="income.portfolio_current_yield_pct",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="income_from_holdings",
                required_inputs=["income.forward_12m_total", "totals.market_value"],
            ),
            FieldSpec(
                path="income.portfolio_yield_on_cost_pct",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="income_from_holdings",
                required_inputs=["income.forward_12m_total", "totals.cost_basis"],
            ),
            FieldSpec(
                path="dividends.windows.30d.total_dividends",
                primary_source="db",
                fallback_sources=["internal"],
                validator="internal",
                source_type="pulled",
                method="realized_dividends_window",
                required_inputs=["dividend_events"],
            ),
            FieldSpec(
                path="dividends.windows.qtd.total_dividends",
                primary_source="db",
                fallback_sources=["internal"],
                validator="internal",
                source_type="pulled",
                method="realized_dividends_window",
                required_inputs=["dividend_events"],
            ),
            FieldSpec(
                path="dividends.windows.ytd.total_dividends",
                primary_source="db",
                fallback_sources=["internal"],
                validator="internal",
                source_type="pulled",
                method="realized_dividends_window",
                required_inputs=["dividend_events"],
            ),
            FieldSpec(
                path="dividends_upcoming.projected",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="derived",
                method="project_upcoming_dividends",
                required_inputs=["holdings[].ultimate.next_ex_date_est"],
            ),
            FieldSpec(
                path="dividends_upcoming.meta.matches_projected",
                primary_source="derived",
                fallback_sources=["internal"],
                validator="internal",
                source_type="validated",
                method="project_upcoming_dividends",
                required_inputs=["dividends_upcoming.events"],
            ),
        ]
    )

    return specs
