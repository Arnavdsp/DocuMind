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
import os
import socket
import sqlite3
import threading
import uuid
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
    error_message TEXT,
    chunk_count INTEGER
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT,
    owner TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    document_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


# Ingestion runs as a background task inside the process that accepted the upload,
# so a job dies with that process. Each job records which process owns it.
_PROCESS_TOKEN = uuid.uuid4().hex


def _process_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{_PROCESS_TOKEN}"


def _owner_is_gone(owner: str | None) -> bool:
    """True only when the process that owns a job has definitely exited.

    Anything we can't check (no owner recorded, or another host) counts as alive,
    so a job that is still running somewhere is never taken over.
    """
    if not owner:
        return False
    host, pid, token = owner.rsplit(":", 2)
    if host != socket.gethostname():
        return False
    if int(pid) == os.getpid():
        # Same pid but a different token: this process was restarted and reused
        # the pid, which is the usual case in a container where the server is pid 1.
        return token != _PROCESS_TOKEN
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class Repository:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn) -> None:
        """Additive, nullable, idempotent. Never destructive.

        A document indexed before chunk_count existed reports NULL, which the
        UI renders as an em-dash. It is deliberately NOT back-filled with the
        old character-length estimate: an estimate presented as a measurement
        is the exact failure this field was added to remove.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        if "chunk_count" not in existing:
            conn.execute("ALTER TABLE documents ADD COLUMN chunk_count INTEGER")
        job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "owner" not in job_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN owner TEXT")

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
        chunk_count: int | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ?, "
                "metrics_json = COALESCE(?, metrics_json), "
                "pages_json = COALESCE(?, pages_json), "
                "chunk_count = COALESCE(?, chunk_count), "
                "error_message = ? WHERE document_id = ?",
                (
                    status.value,
                    _now(),
                    json.dumps(metrics.model_dump()) if metrics else None,
                    json.dumps([p.model_dump() for p in pages]) if pages is not None else None,
                    chunk_count,
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
            chunk_count=row["chunk_count"] if "chunk_count" in row.keys() else None,
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

    def claim_upload(
        self, *, document_id: str, filename: str, content_type: str, size_bytes: int
    ) -> tuple[JobRecord, bool]:
        """Find or create the job for an upload, in one write transaction.

        Returns (job, created). Only the caller that gets created=True should save
        the blob and schedule ingestion. A concurrent upload of the same bytes gets
        the in-flight job back, and a READY document gets a job that is already
        finished. A FAILED document is ingested again, and so is one whose job was
        orphaned because the process running it exited (a restart mid-ingestion).
        """
        now = _now()
        job_id = str(uuid.uuid4())
        created = False
        with self._lock, self._connect() as conn:
            # IMMEDIATE takes SQLite's write lock up front, so two workers sharing
            # the file can't both see "no document" and both insert one.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            status = ProcessingStage(row["status"]) if row else None
            latest = conn.execute(
                "SELECT job_id, status, owner FROM jobs WHERE document_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (document_id,),
            ).fetchone()
            in_flight = (
                status not in (None, ProcessingStage.READY, ProcessingStage.FAILED)
                and latest is not None
                and JobStatus(latest["status"]) in (JobStatus.PENDING, JobStatus.RUNNING)
            )
            if in_flight and _owner_is_gone(latest["owner"]):
                conn.execute(
                    "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? WHERE job_id = ?",
                    (JobStatus.FAILED.value, "Ingestion stopped: its worker exited.", now, latest["job_id"]),
                )
                in_flight = False

            if status == ProcessingStage.READY:
                job_values = (JobStatus.SUCCEEDED, ProcessingStage.READY, 1.0)
            elif in_flight:
                job_id = latest["job_id"]
                job_values = None
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(document_id, filename, content_type, size_bytes, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (document_id, filename, content_type, size_bytes, ProcessingStage.UPLOADING.value, now, now),
                )
                job_values = (JobStatus.PENDING, ProcessingStage.UPLOADING, 0.0)
                created = True

            if job_values is not None:
                job_status, stage, progress = job_values
                conn.execute(
                    "INSERT INTO jobs "
                    "(job_id, document_id, status, stage, progress, created_at, updated_at, owner) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (job_id, document_id, job_status.value, stage.value, progress, now, now, _process_owner()),
                )
        return self.get_job(job_id), created  # type: ignore[return-value]

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
