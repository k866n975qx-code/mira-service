from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.lm_sync import router as lm_sync_router
from app.api.routes.holdings import router as holdings_router
from app.api.routes.dividends import router as dividends_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Mira Service",
        version="0.3.0",
        description="Mira: Lunch Money sync (M1-only), holdings snapshot, dividends.",
    )

    app.include_router(health_router)
    app.include_router(lm_sync_router)
    app.include_router(holdings_router)
    app.include_router(dividends_router)

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {"service": "mira", "status": "running"}

    return app


app = create_app()
