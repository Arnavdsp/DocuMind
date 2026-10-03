"""End-to-end ingestion pipeline.

Runs as a background job (see api/routes/documents.py) so the upload
request returns immediately with a job_id the frontend can poll, instead of
blocking on OCR/embedding for potentially long documents.

Embeddings are computed exactly once here, at ingestion time, and persisted
to the VectorStore — this is the fix for the original architecture's
"re-embed everything on every question" flaw.
"""

from __future__ import annotations

from app.config import Settings
from app.ingestion.extractors import extract
from app.logging import get_logger, log_event
from app.rag.chunking import chunk_document
from app.rag.vector_store import VectorStore
from app.schemas.documents import DocumentSummaryMetrics, PageInfo, ProcessingStage
from app.schemas.jobs import JobStatus
from app.services.model_service import ModelService
from app.storage.blob_store import DocumentBlobStore
from app.storage.repository import Repository
from app.utils.errors import AppError

logger = get_logger(__name__)


def _word_count(pages) -> int:
    return sum(len(p.text.split()) for p in pages)


def _superseded(document_id: str, job_id: str) -> None:
    log_event(logger, "ingestion_superseded", document_id=document_id, job_id=job_id)


def run_ingestion(
    *,
    document_id: str,
    job_id: str,
    extension: str,
    raw_bytes: bytes,
    repository: Repository,
    blob_store: DocumentBlobStore,
    vector_store: VectorStore,
    model_service: ModelService,
    settings: Settings,
) -> None:
    try:
        repository.update_job(
            job_id, status=JobStatus.RUNNING, stage=ProcessingStage.EXTRACTING, progress=0.1, if_active=True
        )
        result = extract(extension=extension, data=raw_bytes, settings=settings)

        if not result.full_text.strip():
            repository.update_document_status(
                document_id,
                status=ProcessingStage.FAILED,
                error_message="No text could be extracted from this document.",
                job_id=job_id,
            )
            repository.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage=ProcessingStage.FAILED,
                progress=1.0,
                error_message="No extractable text.",
                if_active=True,
            )
            return

        if not repository.owns_document(job_id, document_id):
            return _superseded(document_id, job_id)
        blob_store.save_pages(document_id, result.pages)
        repository.update_job(job_id, stage=ProcessingStage.CHUNKING, progress=0.4, if_active=True)

        chunks = chunk_document(
            result.pages,
            document_id=document_id,
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

        repository.update_job(job_id, stage=ProcessingStage.EMBEDDING, progress=0.6, if_active=True)
        if chunks:
            embeddings = model_service.embed([c.text for c in chunks])
            repository.update_job(job_id, stage=ProcessingStage.INDEXING, progress=0.85, if_active=True)
            if not repository.owns_document(job_id, document_id):
                return _superseded(document_id, job_id)
            vector_store.add(chunks, embeddings)

        page_infos = [
            PageInfo(
                page_number=p.page_number,
                char_count=len(p.text),
                extraction_method=p.extraction_method,
                ocr_confidence=p.ocr_confidence,
                is_low_quality=p.is_low_quality,
            )
            for p in result.pages
        ]
        metrics = DocumentSummaryMetrics(
            word_count=_word_count(result.pages),
            character_count=len(result.full_text),
            page_count=result.page_count,
            estimated_reading_minutes=max(1, round(_word_count(result.pages) / 200)),
        )

        finished = repository.update_document_status(
            document_id,
            status=ProcessingStage.READY,
            metrics=metrics,
            pages=page_infos,
            # The count the index actually holds, not an estimate derived from
            # character length. This is what lets the UI drop its "~" prefix.
            chunk_count=len(chunks),
            job_id=job_id,
        )
        if not finished:
            # Deleted or replaced while we were indexing. If the document is gone,
            # remove what we just wrote; a replacement writes its own copy.
            if repository.get_document(document_id) is None:
                blob_store.delete(document_id)
                vector_store.delete(document_id)
            return _superseded(document_id, job_id)
        repository.update_job(
            job_id, status=JobStatus.SUCCEEDED, stage=ProcessingStage.READY, progress=1.0, if_active=True
        )
        log_event(
            logger,
            "ingestion_succeeded",
            document_id=document_id,
            pages=result.page_count,
            chunks=len(chunks),
        )

    except AppError as exc:
        log_event(logger, "ingestion_failed", level=40, document_id=document_id, error=str(exc))
        repository.update_document_status(
            document_id, status=ProcessingStage.FAILED, error_message=exc.user_message, job_id=job_id
        )
        repository.update_job(
            job_id,
            status=JobStatus.FAILED,
            stage=ProcessingStage.FAILED,
            progress=1.0,
            error_message=exc.user_message,
            if_active=True,
        )
    except Exception as exc:  # never let an unexpected error strand a job as "running" forever
        log_event(logger, "ingestion_unexpected_error", level=50, document_id=document_id, error=str(exc))
        message = "The document could not be processed. Try a smaller file or retry."
        repository.update_document_status(
            document_id, status=ProcessingStage.FAILED, error_message=message, job_id=job_id
        )
        repository.update_job(
            job_id,
            status=JobStatus.FAILED,
            stage=ProcessingStage.FAILED,
            progress=1.0,
            error_message=message,
            if_active=True,
        )
