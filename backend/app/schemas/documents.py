from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProcessingStage(str, Enum):
    UPLOADING = "uploading"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    OCR = "ocr"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ExtractionMethod(str, Enum):
    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    MIXED = "mixed"
    PLAIN_TEXT = "plain_text"


class PageInfo(BaseModel):
    """Metadata about a single page of a document."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    char_count: int
    extraction_method: ExtractionMethod
    ocr_confidence: float | None = Field(
        default=None, description="Mean OCR confidence (0-1) when OCR was used, else null."
    )
    is_low_quality: bool = Field(
        default=False, description="True when extraction likely missed content on this page."
    )


class DocumentSummaryMetrics(BaseModel):
    word_count: int
    character_count: int
    page_count: int
    estimated_reading_minutes: int


class DocumentRecord(BaseModel):
    """Public representation of an ingested document."""

    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: ProcessingStage
    created_at: datetime
    updated_at: datetime
    metrics: DocumentSummaryMetrics | None = None
    pages: list[PageInfo] = Field(default_factory=list)
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]


class DocumentUploadResponse(BaseModel):
    document: DocumentRecord
    job_id: str
