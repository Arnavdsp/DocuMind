"""Retrieval orchestration.

This is the piece that replaces the original implementation's flaw of
re-embedding every chunk on every query: embeddings are computed once at
ingestion (see `services/ingestion_pipeline.py`) and persisted in the
`VectorStore`. A query only ever computes ONE new embedding (the question
itself), searches the cached index, and reranks a small candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    query_embedding = model_service.embed([question])[0]

    initial = vector_store.search(document_id, query_embedding, top_k=settings.retrieval_top_k)
    if not initial:
        return RetrievalResult(candidates=[], top_score=0.0, grounding=GroundingLevel.NONE)

    reranked = reranker.rerank(question, initial, top_k=settings.rerank_top_k)
    top_score = reranked[0].score if reranked else 0.0
    grounding = classify_grounding(top_score, min_relevance=settings.min_relevance_score)
    return RetrievalResult(candidates=reranked, top_score=top_score, grounding=grounding)
