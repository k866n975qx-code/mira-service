from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", summary="Health check")
async def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Basic health check.
    Also pings the DB so we know Mira's storage is alive.
    """
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "service": "mira",
        "status": "ok",
        "db_ok": db_ok,
    }