from __future__ import annotations

from app.ingestion.extractors import ExtractedPage
from app.rag.chunking import chunk_document
from app.schemas.documents import ExtractionMethod


def _page(n: int, text: str) -> ExtractedPage:
    return ExtractedPage(page_number=n, text=text, extraction_method=ExtractionMethod.NATIVE_TEXT)


def test_chunking_preserves_page_attribution():
    pages = [
        _page(1, "Sentence one here. Sentence two follows. Sentence three ends it."),
        _page(2, "Page two starts now. It has its own sentences. All separate from page one."),
    ]
    chunks = chunk_document(pages, document_id="doc1", target_tokens=50, overlap_tokens=5)
    assert chunks, "expected at least one chunk"
    assert all(c.document_id == "doc1" for c in chunks)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunks_never_span_multiple_pages():
    pages = [_page(1, "A. " * 5), _page(2, "B. " * 5)]
    chunks = chunk_document(pages, document_id="doc2", target_tokens=200, overlap_tokens=10)
    for chunk in chunks:
        assert "A." not in chunk.text or "B." not in chunk.text


def test_small_target_tokens_produces_multiple_chunks_per_page():
    long_text = " ".join(f"This is sentence number {i} with some extra words." for i in range(40))
    pages = [_page(1, long_text)]
    chunks = chunk_document(pages, document_id="doc3", target_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(c.chunk_id.startswith("doc3:p1:") for c in chunks)


def test_empty_page_produces_no_chunks():
    pages = [_page(1, "")]
    chunks = chunk_document(pages, document_id="doc4", target_tokens=100, overlap_tokens=10)
    assert chunks == []
