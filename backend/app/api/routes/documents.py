from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from fastapi import File as FastAPIFile

from app.config import Settings, get_settings
from app.dependencies import get_blob_store, get_repository, get_vector_store
from app.ingestion.validation import validate_upload
from app.rag.vector_store import VectorStore
from app.schemas.documents import (
    DocumentListResponse,
    DocumentRecord,
    DocumentUploadResponse,
    ProcessingStage,
)
from app.schemas.jobs import JobStatus
from app.services.ingestion_pipeline import run_ingestion
from app.services.model_service import ModelService, get_model_service
from app.storage.blob_store import DocumentBlobStore, compute_document_id
from app.storage.repository import Repository
from app.utils.errors import DocumentNotFound, FileTooLarge, UploadConflict

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastAPIFile(...),
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
    vector_store: VectorStore = Depends(get_vector_store),
    model_service: ModelService = Depends(get_model_service),
) -> DocumentUploadResponse:
    # Read in bounded chunks so an oversized body is rejected before it is all in memory.
    buf = bytearray()
    while chunk := await file.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > settings.max_upload_bytes:
            raise FileTooLarge(f"Files must be under {settings.max_upload_bytes // (1024 * 1024)} MB.")
    raw_bytes = bytes(buf)
    validated = validate_upload(
        filename=file.filename or "upload",
        declared_content_type=file.content_type,
        data=raw_bytes,
        settings=settings,
    )

    document_id = compute_document_id(raw_bytes)

    # Content-addressed: identical bytes map to the same document. The claim is
    # atomic, so a re-upload (even a concurrent one) gets the existing job back
    # instead of starting a second ingestion that would race on the blob and index.
    job, created = repository.claim_upload(
        document_id=document_id,
        filename=validated.safe_filename,
        content_type=validated.content_type,
        size_bytes=validated.size_bytes,
    )
    document = repository.get_document(document_id)
    if not created:
        return DocumentUploadResponse(document=document, job_id=job.job_id)

    try:
        # in a thread: waiting on the document lock must not block the event loop
        saved = await asyncio.to_thread(
            _save_raw_if_owned, blob_store, repository, job.job_id, document_id, validated.extension, raw_bytes
        )
    except Exception:
        # Without this the claimed job would stay pending and every re-upload
        # would be handed back a job that never runs.
        # Scoped to this upload's job: if the document was deleted and uploaded
        # again meanwhile, the new upload's row and job are left alone.
        repository.update_document_status(document_id, status=ProcessingStage.FAILED, job_id=job.job_id)
        repository.update_job(
            job.job_id, status=JobStatus.FAILED, stage=ProcessingStage.FAILED, if_active=True
        )
        raise
    if not saved:
        # deleted between the claim and the write; nothing was stored or scheduled
        raise UploadConflict()

    background_tasks.add_task(
        run_ingestion,
        document_id=document_id,
        job_id=job.job_id,
        extension=validated.extension,
        raw_bytes=raw_bytes,
        repository=repository,
        blob_store=blob_store,
        vector_store=vector_store,
        model_service=model_service,
        settings=settings,
    )

    return DocumentUploadResponse(document=document, job_id=job.job_id)


@router.get("", response_model=DocumentListResponse)
async def list_documents(repository: Repository = Depends(get_repository)) -> DocumentListResponse:
    return DocumentListResponse(documents=repository.list_documents())


@router.get("/{document_id}", response_model=DocumentRecord)
async def get_document(document_id: str, repository: Repository = Depends(get_repository)) -> DocumentRecord:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    return document


def _save_raw_if_owned(
    blob_store: DocumentBlobStore,
    repository: Repository,
    job_id: str,
    document_id: str,
    extension: str,
    raw_bytes: bytes,
) -> bool:
    with blob_store.lock(document_id):
        if not repository.owns_document(job_id, document_id):
            return False
        blob_store.save_raw(document_id, extension, raw_bytes)
        return True


def remove_document(
    document_id: str, repository: Repository, blob_store: DocumentBlobStore, vector_store: VectorStore
) -> None:
    # Under the document lock, so an ingestion that is mid-write finishes that
    # write first, and then sees its job failed and writes nothing more.
    with blob_store.lock(document_id):
        repository.delete_document(document_id)
        blob_store.delete(document_id)
        vector_store.delete(document_id)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    # plain def: waiting on the document lock must not block the event loop
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    remove_document(document_id, repository, blob_store, vector_store)


@router.get("/{document_id}/pages")
async def get_document_pages(
    document_id: str,
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
) -> dict:
    """Source viewer support: extracted text per page, with extraction
    metadata so the frontend can flag OCR/low-quality pages."""
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    pages = blob_store.load_pages(document_id)
    return {
        "document_id": document_id,
        "pages": [
            {
                "page_number": p.page_number,
                "text": p.text,
                "extraction_method": p.extraction_method.value,
                "ocr_confidence": p.ocr_confidence,
                "is_low_quality": p.is_low_quality,
            }
            for p in pages
        ],
    }
