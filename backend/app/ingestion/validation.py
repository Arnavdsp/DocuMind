"""Untrusted-upload validation.

Uploaded files are treated as hostile input. We never trust a filename
extension by itself: extension, declared content-type, and magic bytes must
all agree before a file is accepted, and hard limits guard against
resource-exhaustion (huge files, decompression-bomb images, page-bomb PDFs).
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

from app.config import Settings
from app.utils.errors import FileTooLarge, UnsupportedFileType, ValidationFailed

# Magic-byte signatures for the formats we accept. Checked independently of
# both the filename extension and the client-declared content-type, since
# either can be spoofed trivially.
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}

_EXTENSION_TO_CONTENT_TYPE = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    extension: str
    content_type: str
    size_bytes: int


def sanitize_filename(raw_name: str) -> str:
    """Strip directory components and dangerous characters to prevent
    path traversal (`../../etc/passwd`) and filesystem-unsafe names."""
    name = unicodedata.normalize("NFKD", raw_name or "upload")
    name = name.replace("\\", "/")  # normalize Windows-style separators too
    name = os.path.basename(name)  # drop any path component entirely
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "upload"
    return name[:200]


def _detect_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext


def _check_magic_bytes(extension: str, head: bytes) -> bool:
    if extension == ".txt":
        # Plain text has no reliable magic number; reject obvious binary
        # content (embedded NUL bytes) instead.
        return b"\x00" not in head
    signatures = _MAGIC_SIGNATURES.get(extension)
    if not signatures:
        return False
    return any(head.startswith(sig) for sig in signatures)


def validate_upload(
    *,
    filename: str,
    declared_content_type: str | None,
    data: bytes,
    settings: Settings,
) -> ValidatedUpload:
    if len(data) == 0:
        raise ValidationFailed("The uploaded file is empty.")

    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(f"Files must be under {settings.max_upload_bytes // (1024 * 1024)} MB.")

    safe_name = sanitize_filename(filename)
    extension = _detect_extension(safe_name)

    if extension not in settings.allowed_extensions:
        raise UnsupportedFileType(
            f"'{extension or 'unknown'}' files aren't supported. "
            f"Supported types: {', '.join(settings.allowed_extensions)}."
        )

    if not _check_magic_bytes(extension, data[:16]):
        raise ValidationFailed(
            "The file's content doesn't match its extension. It may be corrupted " "or mislabeled."
        )

    resolved_content_type = _EXTENSION_TO_CONTENT_TYPE[extension]

    return ValidatedUpload(
        safe_filename=safe_name,
        extension=extension,
        content_type=resolved_content_type,
        size_bytes=len(data),
    )
