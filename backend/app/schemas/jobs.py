from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from app.schemas.documents import ProcessingStage


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobRecord(BaseModel):
    job_id: str
    document_id: str
    status: JobStatus
    stage: ProcessingStage
    progress: float  # 0.0 - 1.0
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
