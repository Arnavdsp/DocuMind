"""Per-stage latency instrumentation (FR-30, GAP-9).

The load-bearing property here is nullability, not the numbers. A stage that
did not run must report `null`, never `0` — a zero is indistinguishable from
a stage that ran instantaneously, and every latency percentile built from
these would be quietly wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.chunking import Chunk
from app.rag.reranker import LexicalOverlapReranker
from app.rag.retrieval import retrieve
from app.rag.vector_store import NumpyVectorStore
from app.schemas.common import StageTimings
from app.schemas.qa import GroundingLevel
from app.services.model_service import MockModelService


def _chunk(idx: int, page: int, text: str) -> Chunk:
    return Chunk(
        document_id="docT",
        chunk_id=f"docT:p{page}:{idx}",
        page_number=page,
        section=None,
        text=text,
        start_offset=0,
        end_offset=len(text),
        token_estimate=len(text.split()),
    )


@pytest.fixture
def store(tmp_path: Path) -> NumpyVectorStore:
    store = NumpyVectorStore(tmp_path)
    model = MockModelService()
    chunks = [
        _chunk(0, 1, "The sampling frequency was 20 MHz during the experiment."),
        _chunk(1, 2, "Bananas are a good source of potassium and fiber."),
    ]
    store.add(chunks, model.embed([c.text for c in chunks]))
    return store


def _retrieve(store: NumpyVectorStore, question: str, document_id: str = "docT"):
    return retrieve(
        document_id=document_id,
        question=question,
        model_service=MockModelService(),
        vector_store=store,
        reranker=LexicalOverlapReranker(),
        settings=get_settings(),
    )


# --- the schema ---------------------------------------------------------------


def test_unrun_stages_default_to_none_not_zero():
    timings = StageTimings(total_ms=12.5)
    assert timings.embed_ms is None
    assert timings.lexical_ms is None
    assert timings.fuse_ms is None
    assert timings.rerank_ms is None
    assert timings.generate_ms is None


def test_nulls_survive_serialization():
    """NFR-5: a null must not become 0 on the way out through JSON."""
    payload = StageTimings(total_ms=12.5, embed_ms=3.0).model_dump()
    assert payload["lexical_ms"] is None
    assert payload["rerank_ms"] is None
    assert payload["embed_ms"] == 3.0

    as_json = StageTimings(total_ms=12.5).model_dump_json()
    assert '"generate_ms":null' in as_json.replace(" ", "")
    assert '"generate_ms":0' not in as_json.replace(" ", "")


def test_total_ms_is_required():
    """Total is always measurable, so it is not nullable."""
    with pytest.raises(Exception):
        StageTimings()


# --- retrieval instrumentation --------------------------------------------------


def test_retrieval_reports_stage_timings(store):
    result = _retrieve(store, "what sampling frequency was used")
    assert result.embed_ms is not None and result.embed_ms >= 0
    assert result.search_ms is not None and result.search_ms >= 0
    assert result.rerank_ms is not None and result.rerank_ms >= 0


def test_rerank_timing_is_none_when_nothing_was_retrieved(store):
    """An unknown document returns no candidates, so reranking genuinely did
    not run — its timing must be null rather than a misleading 0.0."""
    result = _retrieve(store, "anything", document_id="no-such-document")
    assert result.candidates == []
    assert result.grounding == GroundingLevel.NONE
    assert result.rerank_ms is None
    # Embedding and search did run, so they are measured.
    assert result.embed_ms is not None
    assert result.search_ms is not None


def test_timings_are_additive_not_overlapping(store):
    """Stages are measured at boundaries, so the parts cannot exceed a
    wall-clock measurement of the whole by more than scheduling noise."""
    from time import perf_counter

    started = perf_counter()
    result = _retrieve(store, "sampling frequency")
    wall_ms = (perf_counter() - started) * 1000

    parts = result.embed_ms + result.search_ms + result.rerank_ms
    assert parts <= wall_ms + 5.0


# --- through the API ------------------------------------------------------------


def _ready_document(client) -> str:
    from tests.conftest import read_fixture
    from tests.test_api_documents import _upload, _wait_until_ready

    response = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = response.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)
    return document_id


def test_ask_response_carries_timings(client):
    document_id = _ready_document(client)
    response = client.post(
        f"/api/documents/{document_id}/ask",
        json={"question": "What was the sampling frequency used in the experiment?"},
    )
    assert response.status_code == 200
    timings = response.json()["timings_ms"]
    assert timings is not None
    assert timings["total_ms"] > 0
    assert timings["embed_ms"] is not None


def test_hybrid_channel_timings_are_null_until_that_channel_exists(client):
    """lexical_ms and fuse_ms are declared now so the field's meaning never
    changes later. Until BM25 lands they must be null, not 0."""
    document_id = _ready_document(client)
    response = client.post(
        f"/api/documents/{document_id}/ask",
        json={"question": "What was the sampling frequency used in the experiment?"},
    )
    timings = response.json()["timings_ms"]
    assert timings["lexical_ms"] is None
    assert timings["fuse_ms"] is None


def test_existing_ask_fields_are_unchanged(client):
    """R2: timings are additive. Nothing a current client reads may move."""
    document_id = _ready_document(client)
    body = client.post(
        f"/api/documents/{document_id}/ask",
        json={"question": "What was the sampling frequency used in the experiment?"},
    ).json()
    for field in (
        "conversation_id",
        "question",
        "answer",
        "abstained",
        "grounding",
        "relevance_score",
        "citations",
        "model_used",
    ):
        assert field in body, f"{field} disappeared from AskResponse"
