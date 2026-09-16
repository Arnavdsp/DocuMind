"""Content-addressed storage for raw uploads and extracted page text.

Documents are identified by a SHA-256 hash of their content, not filename —
this gives deterministic caching (re-uploading the same file is a cache
hit, not reprocessing) and duplicate detection for free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.ingestion.extractors import ExtractedPage
from app.schemas.documents import ExtractionMethod


def compute_document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


class DocumentBlobStore:
    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _doc_dir(self, document_id: str) -> Path:
        d = self._root / document_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_raw(self, document_id: str, extension: str, data: bytes) -> Path:
        path = self._doc_dir(document_id) / f"original{extension}"
        path.write_bytes(data)
        return path

    def raw_path(self, document_id: str, extension: str) -> Path:
        return self._doc_dir(document_id) / f"original{extension}"

    def exists(self, document_id: str) -> bool:
        return self._doc_dir(document_id).joinpath("pages.json").exists()

    def save_pages(self, document_id: str, pages: list[ExtractedPage]) -> None:
        payload = [
            {
                "page_number": p.page_number,
                "text": p.text,
                "extraction_method": p.extraction_method.value,
                "ocr_confidence": p.ocr_confidence,
                "is_low_quality": p.is_low_quality,
            }
            for p in pages
        ]
        (self._doc_dir(document_id) / "pages.json").write_text(json.dumps(payload))

    def load_pages(self, document_id: str) -> list[ExtractedPage]:
        path = self._doc_dir(document_id) / "pages.json"
        if not path.exists():
            return []
        raw = json.loads(path.read_text())
        return [
            ExtractedPage(
                page_number=p["page_number"],
                text=p["text"],
                extraction_method=ExtractionMethod(p["extraction_method"]),
                ocr_confidence=p["ocr_confidence"],
                is_low_quality=p["is_low_quality"],
            )
            for p in raw
        ]

    def delete(self, document_id: str) -> None:
        import shutil

        d = self._doc_dir(document_id)
        if d.exists():
            shutil.rmtree(d)
