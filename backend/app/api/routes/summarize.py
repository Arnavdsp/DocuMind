from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_blob_store, get_repository
from app.schemas.documents import ProcessingStage
from app.schemas.summary import StructuredSummary, SummarizeRequest, SummarizeResponse
from app.services.model_service import ModelService, get_model_service
from app.services.summarization_service import summarize_document
from app.storage.blob_store import DocumentBlobStore
from app.storage.repository import Repository
from app.utils.errors import DocumentNotFound, DocumentNotReady

router = APIRouter(prefix="/api/documents", tags=["summarize"])


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
async def summarize(
    document_id: str,
    request: SummarizeRequest,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
    model_service: ModelService = Depends(get_model_service),
) -> SummarizeResponse:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    if document.status != ProcessingStage.READY:
        raise DocumentNotReady()

    if not request.force_refresh:
        cached = repository.get_summary(document_id)
        if cached:
            return SummarizeResponse(
                document_id=document_id,
                summary=StructuredSummary(**cached["summary"]),
                strategy=cached["strategy"],
                model_used=cached["model_used"],
                cached=True,
            )

    pages = blob_store.load_pages(document_id)
    summary, strategy = summarize_document(pages, model_service=model_service, settings=settings)

    repository.save_summary(
        document_id,
        {"summary": summary.model_dump(), "strategy": strategy, "model_used": model_service.backend_name},
    )

    return SummarizeResponse(
        document_id=document_id,
        summary=summary,
        strategy=strategy,
        model_used=model_service.backend_name,
        cached=False,
    )
