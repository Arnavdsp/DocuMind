"""Page-aware document extraction.

Unlike a whole-document OCR-or-not decision, each PDF page is evaluated
independently: pages with a real text layer use native extraction, pages
that come back empty *or* suspiciously sparse (e.g. a page that's mostly a
scanned figure with a one-line native caption) fall through to OCR. This
avoids both silently losing scanned pages and needlessly OCR-ing pages that
already have perfectly good extractable text.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image

from app.config import Settings
from app.ingestion.ocr import run_ocr
from app.logging import get_logger, log_event
from app.schemas.documents import ExtractionMethod
from app.utils.errors import DocumentTooLarge, ExtractionFailed

logger = get_logger(__name__)

# A page is considered "sparse" (candidate for OCR fallback) when its native
# text layer has fewer than this many non-whitespace characters. Pure image
# pages typically extract to 0; pages with a stray caption/page-number often
# extract to a handful of characters but no real body text.
_SPARSE_TEXT_THRESHOLD = 20


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: ExtractionMethod
    ocr_confidence: float | None = None
    is_low_quality: bool = False


@dataclass
class ExtractionResult:
    pages: list[ExtractedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_pdf(data: bytes, *, settings: Settings) -> ExtractionResult:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency issue, not user error
        raise ExtractionFailed(internal_detail=str(exc)) from exc

    pages: list[ExtractedPage] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > settings.max_pages_per_document:
                raise DocumentTooLarge(
                    f"This document has {len(pdf.pages)} pages; the limit is "
                    f"{settings.max_pages_per_document}."
                )
            for i, page in enumerate(pdf.pages, start=1):
                native_text = (page.extract_text() or "").strip()

                if len(native_text) >= _SPARSE_TEXT_THRESHOLD:
                    pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=native_text,
                            extraction_method=ExtractionMethod.NATIVE_TEXT,
                        )
                    )
                    continue

                # Sparse or empty native layer -> OCR this page.
                try:
                    rendered = page.to_image(resolution=200).original
                except Exception as exc:
                    log_event(logger, "page_render_failed", level=30, page=i, error=str(exc))
                    pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=native_text,
                            extraction_method=ExtractionMethod.NATIVE_TEXT,
                            is_low_quality=True,
                        )
                    )
                    continue

                ocr_result = run_ocr(rendered, max_pixels=settings.max_image_pixels)
                combined_text = (native_text + "\n" + ocr_result.text).strip()
                method = ExtractionMethod.MIXED if native_text else ExtractionMethod.OCR
                pages.append(
                    ExtractedPage(
                        page_number=i,
                        text=combined_text,
                        extraction_method=method,
                        ocr_confidence=ocr_result.confidence,
                        is_low_quality=(not ocr_result.succeeded) or not combined_text,
                    )
                )
    except DocumentTooLarge:
        raise
    except Exception as exc:
        raise ExtractionFailed(internal_detail=str(exc)) from exc

    return ExtractionResult(pages=pages)


def extract_image(data: bytes, *, settings: Settings) -> ExtractionResult:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ExtractionFailed("This image file appears to be corrupted.", internal_detail=str(exc)) from exc

    ocr_result = run_ocr(image, max_pixels=settings.max_image_pixels)
    return ExtractionResult(
        pages=[
            ExtractedPage(
                page_number=1,
                text=ocr_result.text,
                extraction_method=ExtractionMethod.OCR,
                ocr_confidence=ocr_result.confidence,
                is_low_quality=not ocr_result.succeeded or not ocr_result.text.strip(),
            )
        ]
    )


def extract_txt(data: bytes) -> ExtractionResult:
    text = data.decode("utf-8", errors="ignore").strip()
    return ExtractionResult(
        pages=[
            ExtractedPage(
                page_number=1,
                text=text,
                extraction_method=ExtractionMethod.PLAIN_TEXT,
                is_low_quality=not text,
            )
        ]
    )


def extract(*, extension: str, data: bytes, settings: Settings) -> ExtractionResult:
    if extension == ".pdf":
        return extract_pdf(data, settings=settings)
    if extension in (".jpg", ".jpeg", ".png"):
        return extract_image(data, settings=settings)
    if extension == ".txt":
        return extract_txt(data)
    raise ExtractionFailed(f"No extractor available for '{extension}' files.")
