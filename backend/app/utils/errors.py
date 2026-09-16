"""Application exception hierarchy.

Every exception carries a stable `error_code` and a `user_message` that is
safe to show directly to end users. Internal detail (stack traces, raw
exception text) is logged, never returned in the HTTP response body.
"""

from __future__ import annotations


class AppError(Exception):
    error_code: str = "internal_error"
    http_status: int = 500
    user_message: str = "Something went wrong while processing your request."

    def __init__(self, user_message: str | None = None, *, internal_detail: str | None = None):
        self.user_message = user_message or self.user_message
        self.internal_detail = internal_detail
        super().__init__(self.internal_detail or self.user_message)


class ValidationFailed(AppError):
    error_code = "validation_failed"
    http_status = 422
    user_message = "The uploaded file could not be validated."


class UnsupportedFileType(AppError):
    error_code = "unsupported_file_type"
    http_status = 415
    user_message = "This file type isn't supported yet."


class FileTooLarge(AppError):
    error_code = "file_too_large"
    http_status = 413
    user_message = "This file exceeds the maximum allowed size."


class DocumentTooLarge(AppError):
    error_code = "document_too_large"
    http_status = 413
    user_message = "This document has more pages than the system currently supports."


class DocumentNotFound(AppError):
    error_code = "document_not_found"
    http_status = 404
    user_message = "We couldn't find that document."


class DocumentNotReady(AppError):
    error_code = "document_not_ready"
    http_status = 409
    user_message = "This document is still being processed."


class ExtractionFailed(AppError):
    error_code = "extraction_failed"
    http_status = 422
    user_message = "The document's content could not be extracted."


class ModelUnavailable(AppError):
    error_code = "model_unavailable"
    http_status = 503
    user_message = "The requested capability is temporarily unavailable. Try again shortly."


class TranslationFailed(AppError):
    error_code = "translation_failed"
    http_status = 502
    user_message = "Translation failed. Please try again."


class JobNotFound(AppError):
    error_code = "job_not_found"
    http_status = 404
    user_message = "We couldn't find that job."
