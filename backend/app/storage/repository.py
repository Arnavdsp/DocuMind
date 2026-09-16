"""Metadata persistence.

Document/page/job metadata lives in SQLite (a single file, zero extra
infrastructure — appropriate for the Colab/demo deployment). The schema and
access pattern are intentionally simple so migrating to PostgreSQL later is
a matter of swapping the connection layer, not rewriting call sites.
Document *content* (raw bytes, extracted text) lives separately in
`DocumentBlobStore`; embeddings live in the `VectorStore`. Keeping these
three concerns apart is what makes independent scaling/migration possible.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.documents import DocumentRecord, DocumentSummaryMetrics, PageInfo, ProcessingStage
from app.schemas.jobs import JobRecord, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
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

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    document_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- documents -------------------------------------------------------------
    def create_document(
        self, *, document_id: str, filename: str, content_type: str, size_bytes: int
    ) -> DocumentRecord:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(document_id, filename, content_type, size_bytes, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, filename, content_type, size_bytes, ProcessingStage.UPLOADING.value, now, now),
            )
        return self.get_document(document_id)  # type: ignore[return-value]

    def update_document_status(
        self,
        document_id: str,
        *,
        status: ProcessingStage,
        metrics: DocumentSummaryMetrics | None = None,
        pages: list[PageInfo] | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ?, "
                "metrics_json = COALESCE(?, metrics_json), "
                "pages_json = COALESCE(?, pages_json), "
                "error_message = ? WHERE document_id = ?",
                (
                    status.value,
                    _now(),
                    json.dumps(metrics.model_dump()) if metrics else None,
                    json.dumps([p.model_dump() for p in pages]) if pages is not None else None,
                    error_message,
                    document_id,
                ),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._row_to_document(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM summaries WHERE document_id = ?", (document_id,))

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            status=ProcessingStage(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metrics=(
                DocumentSummaryMetrics(**json.loads(row["metrics_json"])) if row["metrics_json"] else None
            ),
            pages=[PageInfo(**p) for p in json.loads(row["pages_json"])] if row["pages_json"] else [],
            error_message=row["error_message"],
        )

    # -- jobs -----------------------------------------------------------------
    def create_job(self, *, job_id: str, document_id: str) -> JobRecord:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, document_id, status, stage, progress, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    document_id,
                    JobStatus.PENDING.value,
                    ProcessingStage.UPLOADING.value,
                    0.0,
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: ProcessingStage | None = None,
        progress: float | None = None,
        error_message: str | None = None,
    ) -> None:
        current = self.get_job(job_id)
        if current is None:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, progress = ?, updated_at = ?, error_message = ? "
                "WHERE job_id = ?",
                (
                    (status or current.status).value,
                    (stage or current.stage).value,
                    progress if progress is not None else current.progress,
                    _now(),
                    error_message,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return JobRecord(
            job_id=row["job_id"],
            document_id=row["document_id"],
            status=JobStatus(row["status"]),
            stage=ProcessingStage(row["stage"]),
            progress=row["progress"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error_message=row["error_message"],
        )

    # -- summaries (cached deterministic outputs) ------------------------------
    def save_summary(self, document_id: str, payload: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (document_id, payload_json, created_at) VALUES (?, ?, ?)",
                (document_id, json.dumps(payload), _now()),
            )

    def get_summary(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM summaries WHERE document_id = ?", (document_id,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None
