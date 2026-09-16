from __future__ import annotations

import pytest

from app.config import get_settings
from app.ingestion.validation import sanitize_filename, validate_upload
from app.utils.errors import FileTooLarge, UnsupportedFileType, ValidationFailed
from tests.conftest import read_fixture


def test_sanitize_filename_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\weird\\path\\report.pdf") == "report.pdf"


def test_sanitize_filename_strips_unsafe_characters():
    assert sanitize_filename("my report<script>.txt") == "my_report_script_.txt"


def test_validate_upload_rejects_empty_file():
    settings = get_settings()
    with pytest.raises(ValidationFailed):
        validate_upload(filename="empty.txt", declared_content_type="text/plain", data=b"", settings=settings)


def test_validate_upload_rejects_oversized_file():
    settings = get_settings()
    huge = b"a" * (settings.max_upload_bytes + 1)
    with pytest.raises(FileTooLarge):
        validate_upload(filename="big.txt", declared_content_type="text/plain", data=huge, settings=settings)


def test_validate_upload_rejects_unsupported_extension():
    settings = get_settings()
    with pytest.raises(UnsupportedFileType):
        validate_upload(
            filename="script.exe", declared_content_type="application/octet-stream", data=b"MZ\x90\x00", settings=settings
        )


def test_validate_upload_rejects_spoofed_extension():
    """A .pdf filename whose content is not actually a PDF must be rejected
    by magic-byte inspection, not accepted on extension alone."""
    settings = get_settings()
    data = read_fixture("fake.pdf")
    with pytest.raises(ValidationFailed):
        validate_upload(filename="fake.pdf", declared_content_type="application/pdf", data=data, settings=settings)


def test_validate_upload_accepts_genuine_pdf():
    settings = get_settings()
    data = read_fixture("sample.pdf")
    result = validate_upload(filename="sample.pdf", declared_content_type="application/pdf", data=data, settings=settings)
    assert result.extension == ".pdf"
    assert result.content_type == "application/pdf"


def test_validate_upload_accepts_txt():
    settings = get_settings()
    data = read_fixture("sample.txt")
    result = validate_upload(filename="sample.txt", declared_content_type="text/plain", data=data, settings=settings)
    assert result.extension == ".txt"
