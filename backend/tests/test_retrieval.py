from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.chunking import Chunk
from app.rag.reranker import LexicalOverlapReranker
from app.rag.retrieval import classify_grounding, retrieve
from app.rag.vector_store import NumpyVectorStore
from app.schemas.qa import GroundingLevel
from app.services.model_service import MockModelService


def _chunk(doc_id: str, idx: int, page: int, text: str) -> Chunk:
    return Chunk(
        document_id=doc_id,
        chunk_id=f"{doc_id}:p{page}:{idx}",
        page_number=page,
        section=None,
        text=text,
        start_offset=0,
        end_offset=len(text),
        token_estimate=len(text.split()),
    )


def test_vector_store_persists_and_reloads(tmp_path: Path):
    store = NumpyVectorStore(tmp_path)
    model = MockModelService()
    chunks = [
        _chunk("docA", 0, 1, "The sampling frequency was 20 MHz during the experiment."),
        _chunk("docA", 1, 2, "Bananas are a good source of potassium and fiber."),
    ]
    embeddings = model.embed([c.text for c in chunks])
    store.add(chunks, embeddings)

    # Fresh store instance pointed at the same directory should see the data.
    reloaded_store = NumpyVectorStore(tmp_path)
    assert reloaded_store.exists("docA")
    fetched = reloaded_store.get("docA")
    assert len(fetched) == 2


def test_vector_store_search_ranks_relevant_chunk_first(tmp_path: Path):
    store = NumpyVectorStore(tmp_path)
    model = MockModelService()
    chunks = [
        _chunk("docB", 0, 1, "The sampling frequency was 20 MHz during the experiment."),
        _chunk("docB", 1, 2, "Bananas are a good source of potassium and fiber."),
    ]
    store.add(chunks, model.embed([c.text for c in chunks]))

    query_embedding = model.embed(["What sampling frequency was used?"])[0]
    results = store.search("docB", query_embedding, top_k=2)
    assert results[0].chunk.text.startswith("The sampling frequency")


def test_classify_grounding_thresholds():
    assert classify_grounding(0.0, min_relevance=0.2) == GroundingLevel.NONE
    assert classify_grounding(0.25, min_relevance=0.2) == GroundingLevel.WEAK
    assert classify_grounding(0.45, min_relevance=0.2) == GroundingLevel.MODERATE
    assert classify_grounding(0.9, min_relevance=0.2) == GroundingLevel.STRONG


def test_retrieve_end_to_end(tmp_path: Path):
    settings = get_settings()
    store = NumpyVectorStore(tmp_path)
    model = MockModelService()
    chunks = [
        _chunk("docC", 0, 1, "The sampling frequency was 20 MHz during the experiment."),
        _chunk("docC", 1, 2, "Bananas are a good source of potassium and fiber."),
    ]
    store.add(chunks, model.embed([c.text for c in chunks]))

    result = retrieve(
        document_id="docC",
        question="What sampling frequency was used in the experiment?",
        model_service=model,
        vector_store=store,
        reranker=LexicalOverlapReranker(),
        settings=settings,
    )
    assert result.candidates
    assert result.candidates[0].chunk.page_number == 1


def test_retrieve_on_missing_document_abstains(tmp_path: Path):
    settings = get_settings()
    store = NumpyVectorStore(tmp_path)
    model = MockModelService()
    result = retrieve(
        document_id="does-not-exist",
        question="anything",
        model_service=model,
        vector_store=store,
        reranker=LexicalOverlapReranker(),
        settings=settings,
    )
    assert result.grounding == GroundingLevel.NONE
    assert result.candidates == []
