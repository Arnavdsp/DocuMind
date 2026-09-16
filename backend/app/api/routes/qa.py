from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_repository, get_reranker, get_vector_store
from app.rag.generation import build_citations, generate_grounded_answer
from app.rag.reranker import Reranker
from app.rag.retrieval import retrieve
from app.rag.vector_store import VectorStore
from app.schemas.documents import ProcessingStage
from app.schemas.qa import AskRequest, AskResponse
from app.services.model_service import ModelService, get_model_service
from app.storage.repository import Repository
from app.utils.errors import DocumentNotFound, DocumentNotReady

router = APIRouter(prefix="/api/documents", tags=["qa"])


@router.post("/{document_id}/ask", response_model=AskResponse)
async def ask_document(
    document_id: str,
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    vector_store: VectorStore = Depends(get_vector_store),
    reranker: Reranker = Depends(get_reranker),
    model_service: ModelService = Depends(get_model_service),
) -> AskResponse:
    document = repository.get_document(document_id)
    if not document:
        raise DocumentNotFound()
    if document.status != ProcessingStage.READY:
        raise DocumentNotReady()

    result = retrieve(
        document_id=document_id,
        question=request.question,
        model_service=model_service,
        vector_store=vector_store,
        reranker=reranker,
        settings=settings,
    )

    answer, abstained = generate_grounded_answer(
        question=request.question,
        candidates=result.candidates,
        grounding=result.grounding,
        model_service=model_service,
        settings=settings,
    )

    citations = [] if abstained else build_citations(result.candidates)

    return AskResponse(
        conversation_id=request.conversation_id or str(uuid.uuid4()),
        question=request.question,
        answer=answer,
        abstained=abstained,
        grounding=result.grounding,
        relevance_score=round(min(max(result.top_score, 0.0), 1.0), 4),
        citations=citations,
        model_used=model_service.backend_name,
    )
