from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .db import Base 

# ---------- Lunch Money core ----------


class LMAccount(Base):
    __tablename__ = "lm_accounts"

    id = Column(BigInteger, primary_key=True)  # LM account/asset id
    name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)        # "bank", "credit", "investment", etc.
    subtype = Column(Text, nullable=True)      # "checking", "savings", "brokerage", etc.
    balance = Column(Numeric, nullable=True)
    currency = Column(Text, nullable=False, default="USD")
    is_active = Column(Boolean, nullable=False, default=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)

    is_margin = Column(Boolean, nullable=False, default=False)          # margin loan account?
    is_cash_like = Column(Boolean, nullable=False, default=False)       # checking/savings/etc.
    is_external_synced = Column(Boolean, nullable=False, default=False) # your "manual but API-updated" account

    raw = Column(JSONB, nullable=False, default=dict)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    transactions = relationship("LMTransaction", back_populates="account")


class LMCategory(Base):
    __tablename__ = "lm_categories"

    id = Column(BigInteger, primary_key=True)  # LM category id
    name = Column(Text, nullable=False)

    is_income = Column(Boolean, nullable=False, default=False)
    is_transfer = Column(Boolean, nullable=False, default=False)
    is_group = Column(Boolean, nullable=False, default=False)
    parent_id = Column(BigInteger, nullable=True)

    raw = Column(JSONB, nullable=False, default=dict)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    transactions = relationship("LMTransaction", back_populates="category")


class LMTransaction(Base):
    __tablename__ = "lm_transactions"

    id = Column(BigInteger, primary_key=True)  # LM transaction id
    date = Column(Date, nullable=False)
    payee = Column(Text, nullable=True)
    amount = Column(Numeric, nullable=False)   # convention: negative = spend, positive = income
    currency = Column(Text, nullable=False, default="USD")

    account_id = Column(
        BigInteger, ForeignKey("lm_accounts.id"), nullable=True
    )
    category_id = Column(
        BigInteger, ForeignKey("lm_categories.id"), nullable=True
    )

    is_pending = Column(Boolean, nullable=False, default=False)
    is_income = Column(Boolean, nullable=False, default=False)
    is_transfer = Column(Boolean, nullable=False, default=False)

    notes = Column(Text, nullable=True)
    raw = Column(JSONB, nullable=False, default=dict)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    account = relationship("LMAccount", back_populates="transactions")
    category = relationship("LMCategory", back_populates="transactions")


# ---------- Spending buckets (for future safe-spend logic) ----------


class SpendingBucket(Base):
    __tablename__ = "spending_buckets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)     # e.g. "Housing", "Food", "Subscriptions"
    description = Column(Text, nullable=True)

    is_fixed = Column(Boolean, nullable=False, default=False)   # rent, insurance, etc.
    priority_level = Column(
        Text, nullable=False, default="need"
    )  # "need" | "want" | "luxury"

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    category_maps = relationship("CategoryBucketMap", back_populates="bucket")


class CategoryBucketMap(Base):
    __tablename__ = "category_bucket_map"

    lm_category_id = Column(
        BigInteger,
        ForeignKey("lm_categories.id"),
        primary_key=True,
    )
    bucket_id = Column(
        Integer,
        ForeignKey("spending_buckets.id"),
        primary_key=True,
    )

    is_default = Column(Boolean, nullable=False, default=True)

    # relationships
    category = relationship("LMCategory")
    bucket = relationship("SpendingBucket", back_populates="category_maps")


# ---------- Portfolio snapshots (Mira math output) ----------


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True)
    taken_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    source = Column(Text, nullable=False)  # "manual" | "lunchmoney"

    portfolio_value = Column(Numeric, nullable=False)
    positions_value = Column(Numeric, nullable=False)
    cash = Column(Numeric, nullable=False)
    margin_debt = Column(Numeric, nullable=False)
    net_liquidation = Column(Numeric, nullable=False)

    annual_income = Column(Numeric, nullable=False)
    monthly_income = Column(Numeric, nullable=False)
    portfolio_yield = Column(Numeric, nullable=False)

    margin_pct = Column(Numeric, nullable=True)
    margin_band = Column(Text, nullable=True)  # "safe" | "caution" | "danger" | "unknown"

    goal_monthly_target = Column(Numeric, nullable=True)
    goal_progress_pct = Column(Numeric, nullable=True)

    raw_request = Column(JSONB, nullable=True)  # original /snapshot request payload
    notes = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    positions = relationship("SnapshotPosition", back_populates="snapshot")


class SnapshotPosition(Base):
    __tablename__ = "snapshot_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )

    ticker = Column(Text, nullable=False)
    quantity = Column(Numeric, nullable=False)
    price = Column(Numeric, nullable=False)
    value = Column(Numeric, nullable=False)

    category = Column(Text, nullable=False)  # "covered_call_etf", "core_etf", etc.

    snapshot = relationship("Snapshot", back_populates="positions")
