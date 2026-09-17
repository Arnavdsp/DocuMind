"""True chunk count (FR-31, GAP-7).

The UI previously printed a "~"-prefixed estimate derived from character
count, because no endpoint returned the real number. It was the last
estimated value on screen. The index knows its own row count; this records it
at indexing time and exposes it.

The important assertion is the migration one: a document indexed before this
field existed reports null and is **not** back-filled with the old estimate.
An estimate presented as a measurement is the exact failure the field was
added to remove.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from app.schemas.documents import ProcessingStage
from app.storage.repository import Repository

from tests.conftest import read_fixture
from tests.test_api_documents import _upload, _wait_until_ready


def test_document_record_defaults_chunk_count_to_none(tmp_path: Path):
    repo = Repository(tmp_path / "app.db")
    record = repo.create_document(
        document_id="d1", filename="a.txt", content_type="text/plain", size_bytes=10
    )
    assert record.chunk_count is None


def test_chunk_count_round_trips(tmp_path: Path):
    repo = Repository(tmp_path / "app.db")
    repo.create_document(document_id="d1", filename="a.txt", content_type="text/plain", size_bytes=10)
    repo.update_document_status("d1", status=ProcessingStage.READY, chunk_count=37)
    assert repo.get_document("d1").chunk_count == 37


def test_chunk_count_survives_an_unrelated_status_update(tmp_path: Path):
    """COALESCE keeps it; a later status write must not wipe a real count."""
    repo = Repository(tmp_path / "app.db")
    repo.create_document(document_id="d1", filename="a.txt", content_type="text/plain", size_bytes=10)
    repo.update_document_status("d1", status=ProcessingStage.READY, chunk_count=12)
    repo.update_document_status("d1", status=ProcessingStage.READY, error_message=None)
    assert repo.get_document("d1").chunk_count == 12


def test_zero_chunks_is_preserved_as_a_real_measurement(tmp_path: Path):
    """A measured 0 means an empty index. It must not be confused with
    'unknown', which is null."""
    repo = Repository(tmp_path / "app.db")
    repo.create_document(document_id="d1", filename="a.txt", content_type="text/plain", size_bytes=10)
    repo.update_document_status("d1", status=ProcessingStage.READY, chunk_count=0)
    # COALESCE treats 0 as present, so it is stored rather than skipped.
    assert repo.get_document("d1").chunk_count == 0


def test_migration_adds_the_column_to_a_pre_existing_database(tmp_path: Path):
    """A database created before chunk_count existed must open, report null,
    and never be back-filled with the old estimate."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metrics_json TEXT,
            pages_json TEXT,
            error_message TEXT
        );
        INSERT INTO documents VALUES
          ('old', 'legacy.txt', 'text/plain', 10, 'ready',
           '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
           NULL, NULL, NULL);
        """
    )
    legacy.commit()
    legacy.close()

    repo = Repository(db_path)
    record = repo.get_document("old")
    assert record is not None
    assert record.chunk_count is None, "a pre-upgrade document must not be back-filled"


def test_migration_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "app.db"
    Repository(db_path)
    Repository(db_path)  # must not raise "duplicate column name"
    repo = Repository(db_path)
    assert repo.list_documents() == []


def test_ingested_document_reports_its_true_chunk_count(client):
    response = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = response.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    body = client.get(f"/api/documents/{document_id}").json()
    assert body["chunk_count"] is not None
    assert body["chunk_count"] >= 1


@pytest.mark.parametrize("field", ["document_id", "filename", "status", "metrics", "pages"])
def test_existing_document_fields_are_unchanged(client, field):
    """R2: chunk_count is additive; nothing a current client reads may move."""
    response = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = response.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)
    assert field in client.get(f"/api/documents/{document_id}").json()
