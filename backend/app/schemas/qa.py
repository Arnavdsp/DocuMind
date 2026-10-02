from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import StageTimings


class GroundingLevel(str, Enum):
    """A technically-defensible, non-numeric indicator of how well the
    retrieved evidence supports an answer. Deliberately NOT presented as a
    calibrated truth probability — see AskResponse.relevance_score for the
    underlying (retrieval, not truth) score this is derived from.
    """

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_id: str | None = Field(
        default=None, description="Groups turns for multi-turn questioning about a document."
    )


class Citation(BaseModel):
    chunk_id: str
    page_number: int | None
    section: str | None
    snippet: str = Field(description="Short supporting excerpt, truncated for display.")
    relevance_score: float = Field(ge=0.0, le=1.0)


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    abstained: bool = Field(
        description="True when the system declined to answer due to insufficient evidence."
    )
    grounding: GroundingLevel
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Top retrieval relevance score. NOT a probability the answer is correct."
    )
    citations: list[Citation] = Field(default_factory=list)
    model_used: str
    timings_ms: StageTimings | None = Field(
        default=None,
        description="Measured per-stage wall-clock time. A stage that did not run "
        "reports null, never 0, so latency percentiles built from these are honest.",
    )
