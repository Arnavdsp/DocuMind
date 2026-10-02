from __future__ import annotations

from PIL import Image

from app.ingestion.ocr import run_ocr


def test_run_ocr_on_blank_image_does_not_crash():
    blank = Image.new("RGB", (400, 200), color="white")
    result = run_ocr(blank, max_pixels=40_000_000)
    assert result.succeeded is True
    assert result.text == "" or result.text.isspace() is False


def test_run_ocr_rejects_oversized_image():
    huge = Image.new("RGB", (10, 10), color="white")
    import pytest

    from app.utils.errors import ValidationFailed

    with pytest.raises(ValidationFailed):
        run_ocr(huge, max_pixels=1)  # absurdly small ceiling forces rejection path
