from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.rag.reranker import Reranker, build_reranker
from app.rag.vector_store import NumpyVectorStore, VectorStore
from app.services.model_service import get_model_service
from app.storage.blob_store import DocumentBlobStore
from app.storage.repository import Repository


@lru_cache
def get_repository() -> Repository:
    settings = get_settings()
    return Repository(settings.db_path)


@lru_cache
def get_blob_store() -> DocumentBlobStore:
    settings = get_settings()
    return DocumentBlobStore(settings.documents_dir)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return NumpyVectorStore(settings.index_dir)


def get_reranker() -> Reranker:
    settings = get_settings()
    return build_reranker(reranker_model=settings.reranker_model, model_service=get_model_service())


def reset_singletons() -> None:
    """Test helper: clears cached singletons so each test can point them at
    a fresh temporary directory instead of sharing state across tests."""
    get_repository.cache_clear()
    get_blob_store.cache_clear()
    get_vector_store.cache_clear()
    get_model_service.cache_clear()
    get_settings.cache_clear()
