"""Base schema (consolidated)

Revision ID: 0001_base_schema
Revises: 
Create Date: 2025-03-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_base_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "lm_accounts",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("subtype", sa.Text(), nullable=True),
        sa.Column("balance", sa.Numeric(), nullable=True),
        sa.Column("currency", sa.Text(), nullable=False, server_default="USD"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_margin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_cash_like", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_external_synced", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "lm_categories",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("is_income", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_transfer", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_group", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "spending_buckets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_fixed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("priority_level", sa.Text(), nullable=False, server_default="need"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "lm_transactions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("lm_tx_id", sa.BigInteger(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("payee", sa.Text(), nullable=True),
        sa.Column("amount", sa.Numeric(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False, server_default="USD"),
        sa.Column("account_id", sa.BigInteger(), nullable=True),
        sa.Column("category_id", sa.BigInteger(), nullable=True),
        sa.Column("plaid_account_id", sa.BigInteger(), nullable=True),
        sa.Column("is_pending", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_income", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_transfer", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["account_id"], ["lm_accounts.id"]),
        sa.ForeignKeyConstraint(["category_id"], ["lm_categories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lm_transactions_lm_tx_id", "lm_transactions", ["lm_tx_id"], unique=True)
    op.create_index("ix_lm_transactions_plaid_account_id", "lm_transactions", ["plaid_account_id"], unique=False)

    op.create_table(
        "category_bucket_map",
        sa.Column("lm_category_id", sa.BigInteger(), nullable=False),
        sa.Column("bucket_id", sa.Integer(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.ForeignKeyConstraint(["bucket_id"], ["spending_buckets.id"]),
        sa.ForeignKeyConstraint(["lm_category_id"], ["lm_categories.id"]),
        sa.PrimaryKeyConstraint("lm_category_id", "bucket_id"),
    )

    op.create_table(
        "snapshots",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("portfolio_value", sa.Numeric(), nullable=False),
        sa.Column("positions_value", sa.Numeric(), nullable=False),
        sa.Column("cash", sa.Numeric(), nullable=False),
        sa.Column("margin_debt", sa.Numeric(), nullable=False),
        sa.Column("net_liquidation", sa.Numeric(), nullable=False),
        sa.Column("annual_income", sa.Numeric(), nullable=False),
        sa.Column("monthly_income", sa.Numeric(), nullable=False),
        sa.Column("portfolio_yield", sa.Numeric(), nullable=False),
        sa.Column("margin_pct", sa.Numeric(), nullable=True),
        sa.Column("margin_band", sa.Text(), nullable=True),
        sa.Column("goal_monthly_target", sa.Numeric(), nullable=True),
        sa.Column("goal_progress_pct", sa.Numeric(), nullable=True),
        sa.Column("raw_request", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "snapshot_positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", postgresql.UUID(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Numeric(), nullable=False),
        sa.Column("price", sa.Numeric(), nullable=False),
        sa.Column("value", sa.Numeric(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "holding_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("plaid_account_id", sa.BigInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plaid_account_id", "as_of_date", name="uq_holding_snapshots_plaid_date"),
    )
    op.create_index("ix_holding_snapshots_plaid_account_id", "holding_snapshots", ["plaid_account_id"], unique=False)
    op.create_index("ix_holding_snapshots_as_of_date", "holding_snapshots", ["as_of_date"], unique=False)

    op.create_table(
        "dividend_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("plaid_account_id", sa.BigInteger(), nullable=False),
        sa.Column("lm_transaction_id", sa.BigInteger(), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=True),
        sa.Column("cusip", sa.String(length=32), nullable=True),
        sa.Column("pay_date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="lunchmoney"),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["lm_transaction_id"], ["lm_transactions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_dividend_events_plaid_account_id", "dividend_events", ["plaid_account_id"], unique=False)
    op.create_index("ix_dividend_events_pay_date", "dividend_events", ["pay_date"], unique=False)
    op.create_index("ix_dividend_events_symbol", "dividend_events", ["symbol"], unique=False)
    op.create_index("ix_dividend_events_lm_transaction_id", "dividend_events", ["lm_transaction_id"], unique=False)

    op.create_table(
        "price_quotes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="yfinance"),
        sa.Column("last_price", sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "as_of_date", "source", name="uq_price_quotes_symbol_date_source"),
    )
    op.create_index("ix_price_quotes_symbol", "price_quotes", ["symbol"], unique=False)
    op.create_index("ix_price_quotes_as_of_date", "price_quotes", ["as_of_date"], unique=False)

    op.create_table(
        "security_mappings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("cusip", sa.String(length=32), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_security_mappings_cusip", "security_mappings", ["cusip"], unique=True)
    op.create_index("ix_security_mappings_ticker", "security_mappings", ["ticker"], unique=False)

    op.create_table(
        "bills",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("frequency", sa.String(length=16), nullable=False, server_default="monthly"),
        sa.Column("due_day", sa.Integer(), nullable=True),
        sa.Column("auto_pay", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("funded_pct", sa.Numeric(5, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("linked_account_id", sa.BigInteger(), nullable=True),
        sa.Column("next_due_date", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["linked_account_id"], ["lm_accounts.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_bills_linked_account_id", "bills", ["linked_account_id"], unique=False)
    op.create_index("ix_bills_next_due_date", "bills", ["next_due_date"], unique=False)

    op.create_table(
        "reserve_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("bills_buffer", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("emergency_target", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("margin_safety_net", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("total_recommended", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("actual_liquidity", sa.Numeric(18, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("coverage_pct", sa.Numeric(5, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="red"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_reserve_snapshots_as_of", "reserve_snapshots", ["as_of"], unique=False)

    op.create_table(
        "budget_categories",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ux_budget_categories_name", "budget_categories", ["name"], unique=True)

    op.create_table(
        "budget_targets",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("budget_category_id", sa.BigInteger(), nullable=False),
        sa.Column("period", sa.String(length=32), nullable=False, server_default="monthly"),
        sa.Column("target_amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["budget_category_id"], ["budget_categories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_budget_targets_budget_category_id", "budget_targets", ["budget_category_id"], unique=False)

    op.create_table(
        "transaction_budgets",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("lm_transaction_id", sa.BigInteger(), nullable=False),
        sa.Column("budget_category_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["lm_transaction_id"], ["lm_transactions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["budget_category_id"], ["budget_categories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_transaction_budgets_lm_transaction_id", "transaction_budgets", ["lm_transaction_id"], unique=False)
    op.create_index("ix_transaction_budgets_budget_category_id", "transaction_budgets", ["budget_category_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_transaction_budgets_budget_category_id", table_name="transaction_budgets")
    op.drop_index("ix_transaction_budgets_lm_transaction_id", table_name="transaction_budgets")
    op.drop_table("transaction_budgets")

    op.drop_index("ix_budget_targets_budget_category_id", table_name="budget_targets")
    op.drop_table("budget_targets")

    op.drop_index("ux_budget_categories_name", table_name="budget_categories")
    op.drop_table("budget_categories")

    op.drop_index("ix_reserve_snapshots_as_of", table_name="reserve_snapshots")
    op.drop_table("reserve_snapshots")

    op.drop_index("ix_bills_next_due_date", table_name="bills")
    op.drop_index("ix_bills_linked_account_id", table_name="bills")
    op.drop_table("bills")

    op.drop_index("ix_security_mappings_ticker", table_name="security_mappings")
    op.drop_index("ix_security_mappings_cusip", table_name="security_mappings")
    op.drop_table("security_mappings")

    op.drop_index("ix_price_quotes_as_of_date", table_name="price_quotes")
    op.drop_index("ix_price_quotes_symbol", table_name="price_quotes")
    op.drop_table("price_quotes")

    op.drop_index("ix_dividend_events_lm_transaction_id", table_name="dividend_events")
    op.drop_index("ix_dividend_events_symbol", table_name="dividend_events")
    op.drop_index("ix_dividend_events_pay_date", table_name="dividend_events")
    op.drop_index("ix_dividend_events_plaid_account_id", table_name="dividend_events")
    op.drop_table("dividend_events")

    op.drop_index("ix_holding_snapshots_as_of_date", table_name="holding_snapshots")
    op.drop_index("ix_holding_snapshots_plaid_account_id", table_name="holding_snapshots")
    op.drop_table("holding_snapshots")

    op.drop_table("snapshot_positions")
    op.drop_table("snapshots")

    op.drop_table("category_bucket_map")

    op.drop_index("ix_lm_transactions_plaid_account_id", table_name="lm_transactions")
    op.drop_index("ix_lm_transactions_lm_tx_id", table_name="lm_transactions")
    op.drop_table("lm_transactions")

    op.drop_table("spending_buckets")
    op.drop_table("lm_categories")
    op.drop_table("lm_accounts")
