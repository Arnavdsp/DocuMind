from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, health, jobs, qa, summarize, translate
from app.config import get_settings
from app.logging import configure_logging, get_logger, get_request_id, log_event, set_request_id
from app.schemas.common import ErrorDetail, ErrorResponse
from app.utils.errors import AppError

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    log_event(logger, "startup", environment=settings.environment)
    yield
    log_event(logger, "shutdown")


app = FastAPI(title="Document Intelligence Suite", version="1.0.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    log_event(
        logger,
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    # User-facing message only; internal detail (real exception text, stack
    # context) goes to the log, never to the response body.
    log_event(
        logger,
        "app_error",
        level=40,
        error_code=exc.error_code,
        internal_detail=exc.internal_detail,
        path=request.url.path,
    )
    body = ErrorResponse(
        detail=ErrorDetail(error_code=exc.error_code, message=exc.user_message, request_id=get_request_id())
    )
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event(logger, "unhandled_exception", level=50, error=str(exc), path=request.url.path)
    body = ErrorResponse(
        detail=ErrorDetail(
            error_code="internal_error",
            message="Something went wrong on our end. Please try again.",
        )
    )
    return JSONResponse(status_code=500, content=body.model_dump())


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(summarize.router)
app.include_router(translate.router)
app.include_router(jobs.router)


# Serve the built React frontend, when present, from the backend itself —
# this is what lets the Colab notebook expose a single ngrok URL for both
# the API and the UI. Mounted last so it never shadows /api/* routes.
_frontend_dist = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
