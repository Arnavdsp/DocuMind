"""Retrieval orchestration.

This is the piece that replaces the original implementation's flaw of
re-embedding every chunk on every query: embeddings are computed once at
ingestion (see `services/ingestion_pipeline.py`) and persisted in the
`VectorStore`. A query only ever computes ONE new embedding (the question
itself), searches the cached index, and reranks a small candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.config import Settings
from app.rag.reranker import Reranker
from app.rag.vector_store import ScoredChunk, VectorStore
from app.schemas.qa import GroundingLevel
from app.services.model_service import ModelService


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[ScoredChunk]
    top_score: float
    grounding: GroundingLevel
    # Measured inside this module because only here are the stage boundaries
    # actually known. A caller timing `retrieve()` as a whole cannot separate
    # embedding from search from reranking without guessing.
    # None means the stage did not run — never 0 (NFR-5).
    embed_ms: float | None = None
    search_ms: float | None = None
    rerank_ms: float | None = None


def classify_grounding(top_score: float, *, min_relevance: float) -> GroundingLevel:
    if top_score < min_relevance:
        return GroundingLevel.NONE
    if top_score < min_relevance + 0.15:
        return GroundingLevel.WEAK
    if top_score < min_relevance + 0.35:
        return GroundingLevel.MODERATE
    return GroundingLevel.STRONG


def retrieve(
    *,
    document_id: str,
    question: str,
    model_service: ModelService,
    vector_store: VectorStore,
    reranker: Reranker,
    settings: Settings,
) -> RetrievalResult:
    started = perf_counter()
    query_embedding = model_service.embed([question])[0]
    embedded_at = perf_counter()

    initial = vector_store.search(document_id, query_embedding, top_k=settings.retrieval_top_k)
    searched_at = perf_counter()

    if not initial:
        # Reranking genuinely did not run, so its timing stays None rather
        # than being reported as 0 ms.
        return RetrievalResult(
            candidates=[],
            top_score=0.0,
            grounding=GroundingLevel.NONE,
            embed_ms=(embedded_at - started) * 1000,
            search_ms=(searched_at - embedded_at) * 1000,
            rerank_ms=None,
        )

    reranked = reranker.rerank(question, initial, top_k=settings.rerank_top_k)
    reranked_at = perf_counter()

    top_score = reranked[0].score if reranked else 0.0
    grounding = classify_grounding(top_score, min_relevance=settings.min_relevance_score)
    return RetrievalResult(
        candidates=reranked,
        top_score=top_score,
        grounding=grounding,
        embed_ms=(embedded_at - started) * 1000,
        search_ms=(searched_at - embedded_at) * 1000,
        rerank_ms=(reranked_at - searched_at) * 1000,
    )
