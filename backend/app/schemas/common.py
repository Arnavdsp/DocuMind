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


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]
