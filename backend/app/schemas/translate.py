from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    target_language: str = Field(description="ISO 639-1 code, e.g. 'hi', 'es', 'fr'.")
    source_language: str | None = Field(
        default=None, description="ISO 639-1 code; auto-detected when omitted."
    )


class TranslateResponse(BaseModel):
    document_id: str
    source_language: str = Field(
        description="ISO 639-1 code, or 'auto' when it was neither declared by the "
        "caller nor determined by the detector. 'auto' is what was actually sent "
        "to the provider, so it reports what happened rather than guessing a code."
    )
    source_language_detected: bool = Field(
        default=False,
        description="True only when the detector identified the language. False when "
        "the caller declared it, or when it could not be determined — so a client can "
        "tell a measured language from a declared or unknown one.",
    )
    target_language: str
    translated_text: str
    provider: str
    truncated: bool = Field(
        default=False,
        description="True when the document exceeded the provider's safe length and was chunked.",
    )
