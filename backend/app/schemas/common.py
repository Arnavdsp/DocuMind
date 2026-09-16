from __future__ import annotations

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Consistent error envelope. `message` is safe to show a user;
    `error_code` lets the frontend branch on error type without parsing text.
    """

    error_code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class StageTimings(BaseModel):
    """Measured wall-clock milliseconds per pipeline stage.

    `None` means the stage did not run — never `0`. A zero would be
    indistinguishable from a stage that ran instantaneously, and would make
    every published latency percentile quietly wrong. The frontend renders
    `None` as an em-dash, exactly as it does for every other unmeasured value.

    `lexical_ms` and `fuse_ms` exist for the hybrid retrieval channel and stay
    `None` until it lands, rather than being added later and silently changing
    the meaning of a field clients already read.
    """

    embed_ms: float | None = None
    lexical_ms: float | None = None
    fuse_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]
