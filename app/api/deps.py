from collections.abc import Generator

from sqlalchemy.orm import Session

from app.infra.db import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Yield a DB session and clean it up afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()