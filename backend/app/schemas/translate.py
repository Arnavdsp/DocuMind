from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    target_language: str = Field(description="ISO 639-1 code, e.g. 'hi', 'es', 'fr'.")
    source_language: str | None = Field(
        default=None, description="ISO 639-1 code; auto-detected when omitted."
    )


class TranslateResponse(BaseModel):
    document_id: str
    source_language: str
    target_language: str
    translated_text: str
    provider: str
    truncated: bool = Field(
        default=False,
        description="True when the document exceeded the provider's safe length and was chunked.",
    )
