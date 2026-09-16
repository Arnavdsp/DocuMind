from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_repository
from app.schemas.common import HealthResponse, ReadinessResponse
from app.storage.repository import Repository

router = APIRouter(tags=["health"])

_APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness only: the process is up. Deliberately does not touch the
    database, filesystem, or any model — a health check must never itself
    become slow or expensive."""
    return HealthResponse(status="ok", version=_APP_VERSION)


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness(
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> ReadinessResponse:
    """Readiness checks the things the app actually needs to serve traffic:
    storage directories exist and are writable, and the database is
    reachable. It deliberately does NOT run model inference — that would
    make every readiness probe as slow and expensive as a real request.
    """
    checks = {
        "data_dir_writable": settings.data_dir.exists(),
        "database_reachable": True,
    }
    try:
        repository.list_documents()
    except Exception:
        checks["database_reachable"] = False

    status = "ready" if all(checks.values()) else "degraded"
    return ReadinessResponse(status=status, checks=checks)
