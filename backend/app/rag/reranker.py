"""Reranking of retrieved chunks.

Bi-encoder (embedding) retrieval is fast but coarse. Reranking rescoring a
smaller candidate set improves precision before the candidates are shown to
the generator. Two implementations are provided behind one interface:

  * `LexicalOverlapReranker` — a fast, dependency-free heuristic (token
    overlap + coverage) used as the default so the system has *some*
    reranking signal even with no extra model configured.
  * `CrossEncoderReranker` — a real cross-encoder (e.g.
    `cross-encoder/ms-marco-MiniLM-L-6-v2`) used when
    `settings.reranker_model` is configured; scores are combined with the
    original retrieval score rather than replacing it outright.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.rag.vector_store import ScoredChunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class LexicalOverlapReranker(Reranker):
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens or not candidates:
            return candidates[:top_k]

        rescored: list[ScoredChunk] = []
        for candidate in candidates:
            chunk_tokens = _tokenize(candidate.chunk.text)
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            # Blend: retrieval (semantic) score dominates, lexical overlap
            # nudges results that share the query's specific terminology
            # (numbers, named entities, technical terms) toward the top.
            combined = (0.75 * candidate.score) + (0.25 * overlap)
            rescored.append(ScoredChunk(chunk=candidate.chunk, score=combined))

        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]


class CrossEncoderReranker(Reranker):
    """Wraps a sentence-transformers CrossEncoder. Loaded lazily via the
    model service so importing this module never triggers a model download.
    """

    def __init__(self, model_name: str, model_service):
        self._model_name = model_name
        self._model_service = model_service

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        cross_encoder = self._model_service.get_cross_encoder(self._model_name)
        pairs = [(query, sc.chunk.text) for sc in candidates]
        raw_scores = cross_encoder.predict(pairs)
        rescored = [ScoredChunk(chunk=sc.chunk, score=float(raw)) for sc, raw in zip(candidates, raw_scores)]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]


def build_reranker(*, reranker_model: str | None, model_service) -> Reranker:
    if reranker_model:
        return CrossEncoderReranker(reranker_model, model_service)
    return LexicalOverlapReranker()
