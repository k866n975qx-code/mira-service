from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.lm_sync import router as lm_sync_router
from app.api.routes.lm_raw import router as lm_raw_router
from app.api.routes.holdings import router as holdings_router
from app.api.routes.accounts import router as accounts_router
from app.api.routes.cashflow import router as cashflow_router
from app.api.routes.bills import router as bills_router
from app.api.routes.reserves import router as reserves_router
from app.api.routes.liquidity import router as liquidity_router
from app.api.routes.runway import router as runway_router
from app.api.routes.insights import router as insights_router
from app.api.routes.budgets import router as budgets_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Mira Service",
        version="0.1.0",
        description="Mira: finance engine (Lunch Money + portfolio math).",
    )

    app.include_router(health_router)
    app.include_router(lm_sync_router)
    app.include_router(lm_raw_router)
    app.include_router(holdings_router)
    app.include_router(accounts_router)
    app.include_router(cashflow_router)
    app.include_router(bills_router)
    app.include_router(reserves_router)
    app.include_router(liquidity_router)
    app.include_router(runway_router)
    app.include_router(insights_router)
    app.include_router(budgets_router)

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {"service": "mira", "status": "running"}

    return app


app = create_app()