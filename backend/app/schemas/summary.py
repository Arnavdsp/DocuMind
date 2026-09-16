from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    force_refresh: bool = False


class StructuredSummary(BaseModel):
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    important_numbers: list[str] = Field(default_factory=list)
    methodology: str | None = None
    limitations: str | None = None


class SummarizeResponse(BaseModel):
    document_id: str
    summary: StructuredSummary
    strategy: str = Field(description="'direct' for short documents, 'map_reduce' for long ones.")
    model_used: str
    cached: bool = False
