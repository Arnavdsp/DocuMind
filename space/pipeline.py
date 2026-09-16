"""Thin adapter between the Gradio UI and the real application.

**This module defines no retrieval logic.** Chunking, embedding, retrieval,
reranking, grounding classification and abstention are imported from
`app.*` and called unchanged. If a retrieval behaviour needs fixing it is
fixed in the backend and this Space inherits it.

That rule is not stylistic. Two copies of retrieval logic diverge within a
week, and then the public demo and the published evaluation numbers describe
different systems — which would be the worst possible version of a
fabricated number.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.ingestion.extractors import extract
from app.rag.chunking import chunk_document
from app.rag.generation import build_citations, generate_grounded_answer
from app.rag.reranker import build_reranker
from app.rag.retrieval import retrieve
from app.rag.vector_store import NumpyVectorStore
from app.schemas.qa import GroundingLevel
from app.services.model_service import get_model_service
from app.services.summarization_service import summarize_document
from app.services.translation_service import get_translation_provider, translate_document

_settings = get_settings()
_vector_store = NumpyVectorStore(_settings.index_dir)
_pages_by_document: dict[str, list] = {}
_titles: dict[str, str] = {}


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    title: str
    pages: int
    chunks: int
    words: int


@dataclass(frozen=True)
class AskResult:
    answer: str
    abstained: bool
    grounding: GroundingLevel
    top_score: float
    citations: list
    candidates: list
    model_used: str
    # None means the stage did not run — never 0 (NFR-5).
    embed_ms: float | None
    search_ms: float | None
    rerank_ms: float | None
    generate_ms: float | None
    total_ms: float


def backend_name() -> str:
    return get_model_service().backend_name


def device_info() -> dict[str, str]:
    return get_model_service().device_info


def is_mock_embeddings() -> bool:
    """True when retrieval scores carry no meaning and the UI must say so."""
    name = backend_name()
    return name == "mock" or "embeddings: mock" in name


def ingest(path: str | Path, *, title: str | None = None) -> IngestResult:
    """Extract, chunk, embed and index a document. Mirrors run_ingestion's
    steps without the job/repository bookkeeping the Space has no use for."""
    path = Path(path)
    model = get_model_service()

    result = extract(extension=path.suffix.lower(), data=path.read_bytes(), settings=_settings)
    if not result.full_text.strip():
        raise ValueError(f"No text could be extracted from {path.name}.")

    document_id = uuid.uuid5(uuid.NAMESPACE_URL, f"documind:{path.name}:{len(result.full_text)}").hex
    chunks = chunk_document(
        result.pages,
        document_id=document_id,
        target_tokens=_settings.chunk_target_tokens,
        overlap_tokens=_settings.chunk_overlap_tokens,
    )
    _vector_store.add(chunks, model.embed([c.text for c in chunks]))

    _pages_by_document[document_id] = result.pages
    _titles[document_id] = title or path.name

    return IngestResult(
        document_id=document_id,
        title=_titles[document_id],
        pages=len(result.pages),
        chunks=len(chunks),
        words=sum(len(p.text.split()) for p in result.pages),
    )


def ask(document_id: str, question: str) -> AskResult:
    """One grounded question. Timings are measured per stage, never estimated."""
    model = get_model_service()
    reranker = build_reranker(reranker_model=_settings.reranker_model, model_service=model)

    started = time.perf_counter()
    retrieval = retrieve(
        document_id=document_id,
        question=question,
        model_service=model,
        vector_store=_vector_store,
        reranker=reranker,
        settings=_settings,
    )
    retrieved_at = time.perf_counter()

    answer, abstained = generate_grounded_answer(
        question=question,
        candidates=retrieval.candidates,
        grounding=retrieval.grounding,
        model_service=model,
        settings=_settings,
    )
    finished = time.perf_counter()

    return AskResult(
        answer=answer,
        abstained=abstained,
        grounding=retrieval.grounding,
        top_score=retrieval.top_score,
        citations=[] if abstained else build_citations(retrieval.candidates),
        candidates=retrieval.candidates,
        model_used=model.backend_name,
        # Read from the backend's own stage boundaries rather than timed from
        # out here, which could not separate embed from search from rerank.
        embed_ms=retrieval.embed_ms,
        search_ms=retrieval.search_ms,
        rerank_ms=retrieval.rerank_ms,
        # Generation is skipped entirely when the retrieval gate abstains, so
        # its timing is null rather than a near-zero implying a model call.
        generate_ms=(finished - retrieved_at) * 1000 if retrieval.candidates else None,
        total_ms=(finished - started) * 1000,
    )


def candidate_rows(document_id: str, question: str) -> tuple[list[list], float]:
    """The retrieval pool with the lid off: what came back, and how it scored.

    Retrieval only — no generation, so this costs nothing but a local
    embedding and a rerank pass.
    """
    model = get_model_service()
    reranker = build_reranker(reranker_model=_settings.reranker_model, model_service=model)

    started = time.perf_counter()
    query_embedding = model.embed([question])[0]
    dense = _vector_store.search(document_id, query_embedding, top_k=_settings.retrieval_top_k)
    reranked = reranker.rerank(question, dense, top_k=_settings.rerank_top_k)
    elapsed = (time.perf_counter() - started) * 1000

    survived = {sc.chunk.chunk_id: i for i, sc in enumerate(reranked, start=1)}
    rows = []
    for rank, scored in enumerate(dense, start=1):
        final = survived.get(scored.chunk.chunk_id)
        rows.append(
            [
                rank,
                f"{scored.score:.4f}",
                str(final) if final else "—",
                scored.chunk.page_number if scored.chunk.page_number else "—",
                scored.chunk.text[:180].replace("\n", " ") + ("…" if len(scored.chunk.text) > 180 else ""),
            ]
        )
    return rows, elapsed


def summarize(document_id: str):
    """Structured summary plus the strategy that produced it.

    Returns (StructuredSummary, strategy, elapsed_ms). `strategy` is "direct"
    for short documents and "map_reduce" for long ones — reported rather than
    inferred, so the caller can see which path ran.
    """
    pages = _pages_by_document.get(document_id)
    if pages is None:
        raise ValueError("That document is not loaded. Read it into memory first.")
    started = time.perf_counter()
    summary, strategy = summarize_document(pages, model_service=get_model_service(), settings=_settings)
    return summary, strategy, (time.perf_counter() - started) * 1000


def translate(document_id: str, target_language: str, source_language: str | None = None):
    """Translate the whole document, reporting real segment counts.

    Segmentation and counting live in `app.services.translation_service`; this
    only passes values through, so the Space and the API report identically.
    """
    pages = _pages_by_document.get(document_id)
    if pages is None:
        raise ValueError("That document is not loaded. Read it into memory first.")

    full_text = "\n\n".join(p.text for p in pages if p.text)
    provider = get_translation_provider(_settings.translation_provider, get_model_service())

    # Three distinct cases kept distinct, exactly as the API route does it:
    # declared by the caller, detected, or genuinely unknown ("auto").
    declared = source_language or None
    detected = provider.detect_language(full_text) if not declared else None
    source = declared or detected or "auto"

    started = time.perf_counter()
    result = translate_document(provider, full_text, source=source, target=target_language)
    elapsed = (time.perf_counter() - started) * 1000
    return result, source, detected is not None, provider.name, elapsed


def translation_enabled() -> bool:
    return _settings.translation_provider != "none"


def title_for(document_id: str) -> str:
    return _titles.get(document_id, document_id)


def page_count(document_id: str) -> int:
    return len(_pages_by_document.get(document_id, []))


def chunk_count(document_id: str) -> int:
    """The true count from the index — not estimated from character length."""
    if not _vector_store.exists(document_id):
        return 0
    return len(_vector_store.get(document_id))
