"""Regression tests for fixes made while restoring the source tree."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import get_settings
from app.ingestion.extractors import ExtractedPage, extract_image
from app.rag.chunking import Chunk, chunk_document
from app.rag.reranker import CrossEncoderReranker
from app.rag.vector_store import NumpyVectorStore, ScoredChunk
from app.schemas.documents import ExtractionMethod, ProcessingStage
from app.schemas.jobs import JobStatus
from app.services.translation_service import _sentence_aware_chunks
from app.storage.blob_store import DocumentBlobStore
from app.utils.errors import DocumentTooLarge
from tests.conftest import read_fixture
from tests.test_api_documents import _upload, _wait_until_ready


def _page(text: str) -> ExtractedPage:
    return ExtractedPage(page_number=1, text=text, extraction_method=ExtractionMethod.NATIVE_TEXT)


# -- upload -----------------------------------------------------------------


def test_reupload_of_ready_document_returns_a_finished_job(client):
    first = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = first.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    second = _upload(client, "again.txt", read_fixture("sample.txt"), "text/plain")
    job = client.get(f"/api/jobs/{second.json()['job_id']}").json()
    # a pending job here would never finish, and the client polls until it does
    assert job["status"] == "succeeded"
    assert job["progress"] == 1.0


def test_reupload_during_ingestion_reuses_the_in_flight_job(client):
    from app.dependencies import get_repository

    first = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = first.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)
    # pretend ingestion is still running: both the document and its job
    repository = get_repository()
    repository.update_document_status(document_id, status=ProcessingStage.EMBEDDING)
    repository.update_job(
        first.json()["job_id"], status=JobStatus.RUNNING, stage=ProcessingStage.EMBEDDING, progress=0.5
    )

    second = _upload(client, "again.txt", read_fixture("sample.txt"), "text/plain")
    assert second.json()["job_id"] == first.json()["job_id"]
    job = client.get(f"/api/jobs/{second.json()['job_id']}").json()
    assert job["status"] == "running"


def test_concurrent_claims_schedule_one_ingestion(tmp_path):
    import threading

    from app.storage.repository import Repository

    repository = Repository(tmp_path / "meta.db")
    barrier = threading.Barrier(8)
    results = []

    def claim():
        barrier.wait()
        results.append(
            repository.claim_upload(
                document_id="a" * 32, filename="a.txt", content_type="text/plain", size_bytes=1
            )
        )

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(created for _, created in results) == 1
    assert len({job.job_id for job, _ in results}) == 1


def test_failed_document_is_claimed_again(tmp_path):
    from app.storage.repository import Repository

    repository = Repository(tmp_path / "meta.db")
    kwargs = dict(document_id="b" * 32, filename="b.txt", content_type="text/plain", size_bytes=1)
    first, _ = repository.claim_upload(**kwargs)
    repository.update_document_status(kwargs["document_id"], status=ProcessingStage.FAILED)

    second, created = repository.claim_upload(**kwargs)
    assert created
    assert second.job_id != first.job_id


def _claim_with_owner(tmp_path, owner):
    import sqlite3

    from app.storage.repository import Repository

    repository = Repository(tmp_path / "meta.db")
    kwargs = dict(document_id="c" * 32, filename="c.txt", content_type="text/plain", size_bytes=1)
    first, _ = repository.claim_upload(**kwargs)
    repository.update_document_status(kwargs["document_id"], status=ProcessingStage.EMBEDDING)
    with sqlite3.connect(tmp_path / "meta.db") as conn:
        conn.execute("UPDATE jobs SET status = 'running', owner = ? WHERE job_id = ?", (owner, first.job_id))
    second, created = repository.claim_upload(**kwargs)
    return repository, first, second, created


def test_job_of_an_exited_worker_is_not_reused(tmp_path):
    import socket
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # its pid now belongs to no running process
    repository, first, second, created = _claim_with_owner(
        tmp_path, f"{socket.gethostname()}:{proc.pid}:deadbeef"
    )
    assert created
    assert second.job_id != first.job_id
    assert repository.get_job(first.job_id).status == JobStatus.FAILED


def test_job_of_a_restarted_process_with_the_same_pid_is_not_reused(tmp_path):
    import os
    import socket

    _, first, second, created = _claim_with_owner(
        tmp_path, f"{socket.gethostname()}:{os.getpid()}:previous-run"
    )
    assert created
    assert second.job_id != first.job_id


def test_job_owned_by_another_host_is_reused(tmp_path):
    _, first, second, created = _claim_with_owner(tmp_path, "some-other-host:1:abc")
    assert not created
    assert second.job_id == first.job_id


def test_oversized_upload_is_rejected(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    resp = _upload(client, "big.txt", b"x" * 100, "text/plain")
    assert resp.status_code == 413


# -- ingestion --------------------------------------------------------------


def test_image_pixel_limit_is_checked_before_decoding(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_image_pixels", 100)
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buf, format="PNG")
    with pytest.raises(DocumentTooLarge):
        extract_image(buf.getvalue(), settings=settings)


# -- chunking ---------------------------------------------------------------


def test_chunk_offsets_slice_back_to_the_source_text():
    text = "  First sentence here.\n\nSecond   one follows.  Third is last.\n"
    page = _page(text)
    chunks = chunk_document([page], document_id="d", target_tokens=4, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        span = text[c.start_offset : c.end_offset]
        assert span.split() == c.text.split()


@pytest.mark.parametrize("target,overlap", [(0, 0), (-5, 0), (10, 10), (10, 20), (10, -1)])
def test_chunk_document_rejects_bad_sizes(target, overlap):
    with pytest.raises(ValueError):
        chunk_document([_page("A sentence.")], document_id="d", target_tokens=target, overlap_tokens=overlap)


# -- retrieval --------------------------------------------------------------


def _chunk(i: int) -> Chunk:
    return Chunk("d", f"d:p1:{i}", 1, None, f"text {i}", 0, 6, 2)


def test_vector_store_rejects_mismatched_embeddings(tmp_path: Path):
    store = NumpyVectorStore(tmp_path)
    with pytest.raises(ValueError):
        store.add([_chunk(0), _chunk(1)], np.ones((3, 4), dtype=np.float32))


def test_cross_encoder_scores_are_bounded_and_blended():
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [12.0, -12.0]

    class FakeModels:
        def get_cross_encoder(self, name):
            return FakeCrossEncoder()

    reranker = CrossEncoderReranker("fake", FakeModels())
    out = reranker.rerank("q", [ScoredChunk(_chunk(0), 0.4), ScoredChunk(_chunk(1), 0.9)], top_k=2)
    assert all(0.0 <= sc.score <= 1.0 for sc in out)
    assert out[0].chunk.chunk_id == "d:p1:0"  # strong cross-encoder signal wins
    assert out[1].score == pytest.approx(0.5 * 0.9 + 0.5 * (1 / (1 + np.exp(12))))


# -- translation ------------------------------------------------------------


def test_translation_chunks_respect_max_chars_without_sentence_breaks():
    text = "word " * 400  # no sentence punctuation at all, like OCR or table text
    chunks = _sentence_aware_chunks(text.strip(), max_chars=100)
    assert chunks and all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_translation_hard_cuts_a_single_long_token():
    chunks = _sentence_aware_chunks("x" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


# -- storage ----------------------------------------------------------------


def test_blob_store_rejects_non_hex_ids(tmp_path: Path):
    store = DocumentBlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.load_pages("../escape")
    assert list(tmp_path.iterdir()) == []  # reads no longer create directories
