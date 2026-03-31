from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info(
        "devpilot_startup",
        version=settings.app_version,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("devpilot_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DevPilot AI",
        description="AI Codebase Intelligence Agent — RAG + Multi-Agent system",
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from app.api.routes.health import router as health_router
    from app.api.routes.ingest import router as ingest_router
    from app.api.routes.query import router as query_router
    from app.api.routes.webhook import router as webhook_router

    app.include_router(health_router)
    app.include_router(ingest_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    app.include_router(webhook_router, prefix="/api")

    return app


app = create_app()
