"""Grounded answer generation.

The generator only ever sees retrieved evidence (never the whole document),
is explicitly instructed to abstain when the evidence doesn't answer the
question, and every answer carries citations back to source pages — so the
UI can show "Sources: Page 7, Page 8" instead of an unverifiable claim.
"""

from __future__ import annotations

from app.config import Settings
from app.rag.vector_store import ScoredChunk
from app.schemas.qa import Citation, GroundingLevel
from app.services.model_service import ModelService

_SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the numbered evidence passages provided. If the passages do not "
    "contain enough information to answer confidently, respond exactly with: "
    '"I don\'t have enough information in this document to answer that." '
    "Never use outside knowledge. Keep answers concise and cite passage "
    "numbers like [1] inline when helpful."
)

_ABSTENTION_TEXT = "I don't have enough information in this document to answer that."


def _build_evidence_block(candidates: list[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(candidates, start=1):
        page = f"page {sc.chunk.page_number}" if sc.chunk.page_number else "unknown page"
        lines.append(f"[{i}] ({page}) {sc.chunk.text}")
    return "\n\n".join(lines)


def _snippet(text: str, max_chars: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def build_citations(candidates: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            chunk_id=sc.chunk.chunk_id,
            page_number=sc.chunk.page_number,
            section=sc.chunk.section,
            snippet=_snippet(sc.chunk.text),
            relevance_score=round(min(max(sc.score, 0.0), 1.0), 4),
        )
        for sc in candidates
    ]


def generate_grounded_answer(
    *,
    question: str,
    candidates: list[ScoredChunk],
    grounding: GroundingLevel,
    model_service: ModelService,
    settings: Settings,
) -> tuple[str, bool]:
    """Returns (answer_text, abstained)."""
    if grounding == GroundingLevel.NONE or not candidates:
        return _ABSTENTION_TEXT, True

    evidence = _build_evidence_block(candidates)
    user_prompt = f"Evidence passages:\n\n{evidence}\n\nQuestion: {question}"

    answer = model_service.generate(
        _SYSTEM_PROMPT,
        user_prompt,
        max_new_tokens=settings.generation_max_new_tokens,
        temperature=settings.generation_temperature,
    ).strip()

    if not answer:
        return _ABSTENTION_TEXT, True

    abstained = _ABSTENTION_TEXT.lower() in answer.lower()
    return answer, abstained
