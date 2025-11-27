from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.lm_sync import router as lm_sync_router
from app.api.routes.lm_raw import router as lm_raw_router
from app.api.routes.holdings import router as holdings_router



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

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {"service": "mira", "status": "running"}

    return app


app = create_app()