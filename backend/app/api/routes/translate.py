from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_blob_store, get_repository
from app.schemas.documents import ProcessingStage
from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.translation_service import get_translation_provider, translate_document
from app.services.model_service import ModelService, get_model_service
from app.storage.blob_store import DocumentBlobStore
from app.storage.repository import Repository
from app.utils.errors import DocumentNotFound, DocumentNotReady

router = APIRouter(prefix="/api/documents", tags=["translate"])


@router.post("/{document_id}/translate", response_model=TranslateResponse)
async def translate(
    document_id: str,
    request: TranslateRequest,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    blob_store: DocumentBlobStore = Depends(get_blob_store),
    model_service: ModelService = Depends(get_model_service),
) -> TranslateResponse:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    if document.status != ProcessingStage.READY:
        raise DocumentNotReady()

    pages = blob_store.load_pages(document_id)
    full_text = "\n\n".join(p.text for p in pages if p.text)

    provider = get_translation_provider(settings.translation_provider, model_service)

    # Three distinct cases, kept distinct: the caller declared it, the detector
    # found it, or nobody knows. The third is reported as "auto" — the value
    # actually sent to the provider — rather than defaulted to a real language
    # code, which would put a fabricated fact on the response.
    declared = request.source_language
    detected = provider.detect_language(full_text) if not declared else None
    source_language = declared or detected or "auto"

    result = translate_document(
        provider, full_text, source=source_language, target=request.target_language
    )

    return TranslateResponse(
        document_id=document_id,
        source_language=source_language,
        source_language_detected=detected is not None,
        target_language=request.target_language,
        translated_text=result.text,
        provider=provider.name,
        # Previously passed False unconditionally. Now an observation: true
        # when the input needed more than one sentence-boundary segment.
        truncated=result.segments_total > 1,
        segments_total=result.segments_total,
        segments_translated=result.segments_translated,
        content_dropped=result.content_dropped,
    )
