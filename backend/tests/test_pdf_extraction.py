from __future__ import annotations

from app.config import get_settings
from app.ingestion.extractors import extract_image, extract_pdf, extract_txt
from app.schemas.documents import ExtractionMethod
from tests.conftest import read_fixture


def test_extract_txt_returns_single_page():
    result = extract_txt(read_fixture("sample.txt"))
    assert result.page_count == 1
    assert "Photosynthesis" in result.full_text
    assert result.pages[0].extraction_method == ExtractionMethod.PLAIN_TEXT


def test_extract_pdf_is_page_aware():
    settings = get_settings()
    result = extract_pdf(read_fixture("sample.pdf"), settings=settings)
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "sampling frequency" in result.full_text.lower()
    # Native text layer present -> should not need OCR
    assert all(p.extraction_method == ExtractionMethod.NATIVE_TEXT for p in result.pages)


def test_extract_image_runs_ocr_and_reports_confidence():
    settings = get_settings()
    result = extract_image(read_fixture("sample.png"), settings=settings)
    assert result.page_count == 1
    page = result.pages[0]
    assert page.extraction_method == ExtractionMethod.OCR
    # OCR is probabilistic; assert the pipeline ran and produced *some*
    # signal rather than asserting exact transcription text.
    assert page.ocr_confidence is None or 0.0 <= page.ocr_confidence <= 1.0
    assert isinstance(page.text, str)
