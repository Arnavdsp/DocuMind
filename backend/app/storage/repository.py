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


_ACTIVE = (JobStatus.PENDING.value, JobStatus.RUNNING.value)

# True while job ? is still pending/running and is the newest job for document ?.
# A delete or a replacement upload ends that, and the worker must stop writing.
_OWNS_DOCUMENT = (
    "EXISTS (SELECT 1 FROM jobs j WHERE j.job_id = ? AND j.status IN (?, ?) "
    "AND j.job_id = (SELECT job_id FROM jobs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1))"
)


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
    try:
        host, pid_text, token = owner.rsplit(":", 2)
        pid = int(pid_text)
    except ValueError:
        return False  # not something we wrote; can't check it, so treat it as alive
    if host != socket.gethostname():
        return False
    if pid == os.getpid():
        # Same pid but a different token: this process was restarted and reused
        # the pid, which is the usual case in a container where the server is pid 1.
        return token != _PROCESS_TOKEN
    try:
        os.kill(pid, 0)
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
            # Take the write lock before checking columns, so two processes opening
            # an old database at once can't both try to add the same column.
            conn.execute("BEGIN IMMEDIATE")
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
        job_id: str | None = None,
    ) -> bool:
        """Returns False if nothing was updated. With job_id, the update only
        applies while that job still owns the document (see _OWNS_DOCUMENT)."""
        fence = f" AND {_OWNS_DOCUMENT}" if job_id else ""
        fence_args = (job_id, *_ACTIVE, document_id) if job_id else ()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE documents SET status = ?, updated_at = ?, "
                "metrics_json = COALESCE(?, metrics_json), "
                "pages_json = COALESCE(?, pages_json), "
                "chunk_count = COALESCE(?, chunk_count), "
                "error_message = ? WHERE document_id = ?" + fence,
                (
                    status.value,
                    _now(),
                    json.dumps(metrics.model_dump()) if metrics else None,
                    json.dumps([p.model_dump() for p in pages]) if pages is not None else None,
                    chunk_count,
                    error_message,
                    document_id,
                    *fence_args,
                ),
            )
        return cur.rowcount > 0

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
            # End any ingestion still running for it, so the worker stops before
            # writing pages, vectors or status for a document that is gone.
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, error_message = ?, updated_at = ? "
                "WHERE document_id = ? AND status IN (?, ?)",
                (
                    JobStatus.FAILED.value,
                    ProcessingStage.FAILED.value,
                    "The document was deleted.",
                    _now(),
                    document_id,
                    *_ACTIVE,
                ),
            )
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
        if_active: bool = False,
    ) -> None:
        """With if_active, a job that has already ended (failed by a delete or a
        replacement upload) is left as it is."""
        current = self.get_job(job_id)
        if current is None:
            return
        fence = " AND status IN (?, ?)" if if_active else ""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, progress = ?, updated_at = ?, error_message = ? "
                "WHERE job_id = ?" + fence,
                (
                    (status or current.status).value,
                    (stage or current.stage).value,
                    progress if progress is not None else current.progress,
                    _now(),
                    error_message,
                    job_id,
                    *(_ACTIVE if if_active else ()),
                ),
            )

    def owns_document(self, job_id: str, document_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_OWNS_DOCUMENT}", (job_id, *_ACTIVE, document_id)).fetchone()
        return bool(row[0])

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
            latest_active = latest is not None and JobStatus(latest["status"]) in (
                JobStatus.PENDING,
                JobStatus.RUNNING,
            )
            in_flight = (
                status not in (None, ProcessingStage.READY, ProcessingStage.FAILED)
                and latest_active
                and not _owner_is_gone(latest["owner"])
            )

            if status == ProcessingStage.READY:
                job_values = (JobStatus.SUCCEEDED, ProcessingStage.READY, 1.0)
            elif in_flight:
                job_id = latest["job_id"]
                job_values = None
            else:
                if latest_active:
                    # Replacing a job that still reads as active (its worker exited, or
                    # died between failing the document and failing the job): end it, so
                    # a client still polling it stops waiting.
                    conn.execute(
                        "UPDATE jobs SET status = ?, stage = ?, error_message = ?, updated_at = ? "
                        "WHERE job_id = ?",
                        (
                            JobStatus.FAILED.value,
                            ProcessingStage.FAILED.value,
                            "Ingestion stopped before finishing.",
                            now,
                            latest["job_id"],
                        ),
                    )
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

    def fail_ownerless_active_jobs(self) -> int:
        """Fail pending/running jobs that were created before owners were recorded.

        claim_upload can't tell whether such a job's worker is still alive, so it
        keeps handing it back. Run this once after upgrading, when every process
        running the old code has stopped (see app.storage.maintenance). Returns the
        number of jobs failed.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, error_message = ?, updated_at = ? "
                "WHERE owner IS NULL AND status IN (?, ?)",
                (
                    JobStatus.FAILED.value,
                    ProcessingStage.FAILED.value,
                    "Ingestion stopped before finishing.",
                    _now(),
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
            return cur.rowcount

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
