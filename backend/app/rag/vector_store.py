"""Vector storage abstraction.

`VectorStore` defines the operations the rest of the app depends on
(add/search/delete/get). `NumpyVectorStore` is a local, dependency-light
implementation suitable for the Colab/demo deployment: embeddings are
computed once at ingestion time, cached to disk per document, and reused
for every subsequent query — the system never re-embeds a document's
chunks to answer a question. Swapping in FAISS, Qdrant, or pgvector for a
real deployment means implementing this same interface; nothing in the
retrieval or API layers needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.chunking import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, document_id: str, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def get(self, document_id: str) -> list[Chunk]: ...

    @abstractmethod
    def delete(self, document_id: str) -> None: ...

    @abstractmethod
    def exists(self, document_id: str) -> bool: ...


class NumpyVectorStore(VectorStore):
    """Cosine-similarity search over embeddings kept in memory and mirrored
    to disk as one .npz per document (embeddings + chunk metadata), keyed
    by document_id so re-ingesting the same content hash is a cache hit
    rather than a recompute.
    """

    def __init__(self, persist_dir: Path):
        self._dir = persist_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[list[Chunk], np.ndarray]] = {}

    def _path(self, document_id: str) -> Path:
        return self._dir / f"{document_id}.npz"

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        document_id = chunks[0].document_id
        normalized = _normalize(embeddings)
        self._cache[document_id] = (chunks, normalized)
        np.savez_compressed(
            self._path(document_id),
            embeddings=normalized,
            chunk_ids=np.array([c.chunk_id for c in chunks]),
            page_numbers=np.array([c.page_number or -1 for c in chunks]),
            sections=np.array([c.section or "" for c in chunks]),
            texts=np.array([c.text for c in chunks]),
            starts=np.array([c.start_offset for c in chunks]),
            ends=np.array([c.end_offset for c in chunks]),
            tokens=np.array([c.token_estimate for c in chunks]),
        )

    def _load(self, document_id: str) -> tuple[list[Chunk], np.ndarray] | None:
        if document_id in self._cache:
            return self._cache[document_id]
        path = self._path(document_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            chunks = [
                Chunk(
                    document_id=document_id,
                    chunk_id=str(cid),
                    page_number=(int(pn) if int(pn) >= 0 else None),
                    section=(str(sec) or None),
                    text=str(text),
                    start_offset=int(start),
                    end_offset=int(end),
                    token_estimate=int(tok),
                )
                for cid, pn, sec, text, start, end, tok in zip(
                    data["chunk_ids"],
                    data["page_numbers"],
                    data["sections"],
                    data["texts"],
                    data["starts"],
                    data["ends"],
                    data["tokens"],
                )
            ]
            embeddings = data["embeddings"]
        self._cache[document_id] = (chunks, embeddings)
        return chunks, embeddings

    def search(self, document_id: str, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        loaded = self._load(document_id)
        if not loaded:
            return []
        chunks, embeddings = loaded
        if len(chunks) == 0:
            return []
        query = _normalize(query_embedding.reshape(1, -1))[0]
        scores = embeddings @ query
        top_k = min(top_k, len(chunks))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [ScoredChunk(chunk=chunks[i], score=float(scores[i])) for i in top_idx]

    def get(self, document_id: str) -> list[Chunk]:
        loaded = self._load(document_id)
        return loaded[0] if loaded else []

    def delete(self, document_id: str) -> None:
        self._cache.pop(document_id, None)
        path = self._path(document_id)
        if path.exists():
            path.unlink()

    def exists(self, document_id: str) -> bool:
        return document_id in self._cache or self._path(document_id).exists()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
