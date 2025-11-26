from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
import datetime as dt

Base = declarative_base()


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    balance = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=dt.datetime.utcnow)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True)
    account_id = Column(String, ForeignKey("accounts.id"))
    date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String)
    payee = Column(String)
    notes = Column(String)

    account = relationship("Account")


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=True)
    source = Column(String, default="lunchmoney")
    last_updated = Column(DateTime, default=dt.datetime.utcnow)