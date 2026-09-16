from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_blob_store, get_repository
from app.schemas.documents import ProcessingStage
from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.translation_service import get_translation_provider
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
) -> TranslateResponse:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    if document.status != ProcessingStage.READY:
        raise DocumentNotReady()

    pages = blob_store.load_pages(document_id)
    full_text = "\n\n".join(p.text for p in pages if p.text)

    provider = get_translation_provider(settings.translation_provider)
    source_language = request.source_language or provider.detect_language(full_text)
    translated = provider.translate(full_text, source=source_language, target=request.target_language)

    return TranslateResponse(
        document_id=document_id,
        source_language=source_language,
        target_language=request.target_language,
        translated_text=translated,
        provider=provider.name,
        truncated=False,
    )
