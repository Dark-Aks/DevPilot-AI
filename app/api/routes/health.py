from __future__ import annotations

from fastapi import APIRouter, status

from app.config import settings
from app.utils.cache import cache_stats

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
        "caches": cache_stats(),
    }
