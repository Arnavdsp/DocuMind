"""Translation provider abstraction.

Replaces a direct, hard-coded GoogleTranslator("en", "hi") call with:
  * a provider interface so the backing service is configurable
    (`TRANSLATION_PROVIDER`), not hard-wired into call sites
  * language detection when the source language isn't specified
  * sentence-aware chunking that never splits mid-sentence/mid-word, unlike
    naive fixed-width character slicing
  * retry with exponential backoff for transient provider errors
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod

from app.logging import get_logger, log_event
from app.utils.errors import TranslationFailed

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SAFE_CHUNK_CHARS = 4000  # stays under typical provider per-call limits


class TranslationProvider(ABC):
    name: str

    @abstractmethod
    def detect_language(self, text: str) -> str: ...

    @abstractmethod
    def translate(self, text: str, *, source: str, target: str) -> str: ...


def _sentence_aware_chunks(text: str, max_chars: int) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class GoogleTranslateProvider(TranslationProvider):
    name = "google"

    def detect_language(self, text: str) -> str:
        try:
            from deep_translator import single_detection

            return single_detection(text[:500], api_key=None) or "en"
        except Exception:
            return "en"  # safe default; explicit source overrides this anyway

    def translate(self, text: str, *, source: str, target: str) -> str:
        from deep_translator import GoogleTranslator

        chunks = _sentence_aware_chunks(text, _SAFE_CHUNK_CHARS)
        translated_parts: list[str] = []
        translator = GoogleTranslator(source=source or "auto", target=target)

        for chunk in chunks:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    translated_parts.append(translator.translate(chunk) or "")
                    break
                except Exception as exc:  # transient network/provider errors
                    last_error = exc
                    log_event(logger, "translation_retry", level=30, attempt=attempt, error=str(exc))
                    time.sleep(0.5 * (2**attempt))
            else:
                raise TranslationFailed(internal_detail=str(last_error))

        return " ".join(p for p in translated_parts if p)


class NullTranslationProvider(TranslationProvider):
    """Used when TRANSLATION_PROVIDER=none — keeps the API contract intact
    while making it explicit that no external translation call will occur.
    """

    name = "none"

    def detect_language(self, text: str) -> str:
        return "en"

    def translate(self, text: str, *, source: str, target: str) -> str:
        raise TranslationFailed("Translation is disabled on this deployment.")


def get_translation_provider(provider_name: str) -> TranslationProvider:
    if provider_name == "google":
        return GoogleTranslateProvider()
    return NullTranslationProvider()
