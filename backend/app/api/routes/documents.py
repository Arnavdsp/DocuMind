from __future__ import annotations

import uuid

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
from app.utils.errors import DocumentNotFound, FileTooLarge

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

    # Content-addressed cache hit: identical bytes were already uploaded.
    existing = repository.get_document(document_id)
    if existing and existing.status == ProcessingStage.READY:
        # Nothing to ingest, so the job is finished as soon as it exists; a pending
        # job here would leave the client polling forever.
        job = repository.create_job(job_id=str(uuid.uuid4()), document_id=document_id)
        repository.update_job(
            job.job_id, status=JobStatus.SUCCEEDED, stage=ProcessingStage.READY, progress=1.0
        )
        return DocumentUploadResponse(document=existing, job_id=job.job_id)
    if existing and existing.status != ProcessingStage.FAILED:
        # Still ingesting: hand back the in-flight job instead of starting a second
        # ingestion that would reset the document and race on its blob and index.
        in_flight = repository.get_latest_job_for_document(document_id)
        if in_flight is not None:
            return DocumentUploadResponse(document=existing, job_id=in_flight.job_id)

    blob_store.save_raw(document_id, validated.extension, raw_bytes)
    document = repository.create_document(
        document_id=document_id,
        filename=validated.safe_filename,
        content_type=validated.content_type,
        size_bytes=validated.size_bytes,
    )
    job = repository.create_job(job_id=str(uuid.uuid4()), document_id=document_id)

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


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    repository.delete_document(document_id)
    blob_store.delete(document_id)
    vector_store.delete(document_id)


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
