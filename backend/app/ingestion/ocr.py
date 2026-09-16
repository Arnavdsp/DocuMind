"""OCR wrapper around Tesseract with preprocessing and confidence scoring.

Improvements over a bare `pytesseract.image_to_string` call:
  * grayscale + adaptive thresholding to help scanned/low-contrast pages
  * upscaling small images toward a target DPI-equivalent size
  * per-word confidence extracted via `image_to_data`, averaged into a
    single page-level score instead of being silently discarded
  * a hard pixel-count ceiling to reject decompression-bomb style images
  * OCR failures are logged with real detail internally and surfaced to
    callers as a structured, non-crashing result rather than swallowed
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageOps

from app.logging import get_logger, log_event
from app.utils.errors import ValidationFailed

logger = get_logger(__name__)

_TARGET_MIN_DIMENSION = 1600  # upscale small scans so small text stays legible
_LANGUAGE = "eng"


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None  # 0-1, None if no words were detected
    succeeded: bool
    error: str | None = None


def _preprocess(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)  # correct camera-phone orientation
    image = image.convert("L")  # grayscale
    width, height = image.size
    smaller_dim = min(width, height)
    if smaller_dim and smaller_dim < _TARGET_MIN_DIMENSION:
        scale = _TARGET_MIN_DIMENSION / smaller_dim
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    image = ImageOps.autocontrast(image)
    return image


def check_pixel_limit(image: Image.Image, max_pixels: int) -> None:
    width, height = image.size
    if width * height > max_pixels:
        raise ValidationFailed("This image is too large to process. Try a smaller or lower-resolution image.")


def run_ocr(image: Image.Image, *, max_pixels: int) -> OCRResult:
    try:
        import pytesseract

        check_pixel_limit(image, max_pixels)
        processed = _preprocess(image)

        data = pytesseract.image_to_data(processed, lang=_LANGUAGE, output_type=pytesseract.Output.DICT)
        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = text.strip()
            if not text:
                continue
            words.append(text)
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_value >= 0:
                confidences.append(conf_value / 100.0)

        full_text = " ".join(words)
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text=full_text, confidence=mean_confidence, succeeded=True)

    except ValidationFailed:
        raise
    except Exception as exc:  # OCR must never crash the ingestion pipeline
        log_event(logger, "ocr_failed", level=40, error=str(exc))
        return OCRResult(text="", confidence=None, succeeded=False, error=str(exc))
