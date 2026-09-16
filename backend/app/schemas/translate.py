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
        description="True when content was genuinely lost. Mirrors content_dropped. "
        "Segmentation alone is NOT truncation — see segments_total for that.",
    )
    segments_total: int | None = Field(
        default=None, description="Sentence-boundary segments the input was split into."
    )
    segments_translated: int | None = Field(
        default=None,
        description="Segments that came back with content. Equal to segments_total on a "
        "clean run; a shortfall means content was dropped and content_dropped is true.",
    )
    content_dropped: bool = Field(
        default=False,
        description="True when a segment is missing, or when the output is implausibly "
        "short for its input. Measured, not assumed.",
    )
    dropped_reason: str | None = Field(
        default=None,
        description="'segment_missing' | 'output_length_implausible' | null.",
    )
    length_ratio: float | None = Field(
        default=None,
        description="Output characters per input character. Null when there was no input. "
        "Reported whether or not it looks suspect, so the reader can judge it.",
    )
