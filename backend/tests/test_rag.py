from __future__ import annotations

from app.rag.chunking import Chunk
from app.rag.generation import build_citations, generate_grounded_answer
from app.rag.vector_store import ScoredChunk
from app.schemas.qa import GroundingLevel
from app.services.model_service import MockModelService


def _scored_chunk(text: str, page: int, score: float) -> ScoredChunk:
    chunk = Chunk(
        document_id="doc1",
        chunk_id=f"doc1:p{page}:0",
        page_number=page,
        section=None,
        text=text,
        start_offset=0,
        end_offset=len(text),
        token_estimate=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=score)


def test_abstains_when_grounding_is_none():
    answer, abstained = generate_grounded_answer(
        question="What is the capital of France?",
        candidates=[],
        grounding=GroundingLevel.NONE,
        model_service=MockModelService(),
        settings=__import__("app.config", fromlist=["get_settings"]).get_settings(),
    )
    assert abstained is True
    assert "don't have enough information" in answer.lower()


def test_answers_when_grounded():
    from app.config import get_settings

    candidates = [_scored_chunk("The experiment used a 20 MHz sampling frequency.", 7, 0.6)]
    answer, abstained = generate_grounded_answer(
        question="What sampling frequency was used?",
        candidates=candidates,
        grounding=GroundingLevel.STRONG,
        model_service=MockModelService(),
        settings=get_settings(),
    )
    assert abstained is False
    assert answer.strip() != ""


def test_citations_reference_correct_pages():
    candidates = [
        _scored_chunk("First relevant passage about sampling frequency.", 7, 0.9),
        _scored_chunk("Second relevant passage about methodology.", 8, 0.5),
    ]
    citations = build_citations(candidates)
    assert [c.page_number for c in citations] == [7, 8]
    assert all(0.0 <= c.relevance_score <= 1.0 for c in citations)


def test_citation_snippets_are_truncated():
    long_text = "word " * 500
    candidates = [_scored_chunk(long_text, 1, 0.5)]
    citations = build_citations(candidates)
    assert len(citations[0].snippet) < len(long_text)
