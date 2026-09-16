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
    def detect_language(self, text: str) -> str | None:
        """ISO 639-1 code, or None when the language could not be determined.

        None is a real answer. Returning a plausible default here would put a
        fabricated fact on TranslateResponse.source_language, which the UI
        renders as measured.
        """
        ...

    @abstractmethod
    def translate(self, text: str, *, source: str, target: str) -> str: ...


def detect_language_offline(text: str) -> str | None:
    """Identify the language locally, with no network call and no API key.

    Previously this went through `deep_translator.single_detection`, which
    requires a detectlanguage.com API key. It was called with `api_key=None`,
    so it raised on every invocation and the caller's bare `except` returned
    "en" — meaning source language was hardcoded to English, silently, and
    reported as though it had been measured.

    py3langid carries its model as bundled data, so this satisfies NFR-8
    (no runtime network) and costs nothing.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        import py3langid

        code, _confidence = py3langid.classify(stripped[:2000])
        return code or None
    except Exception as exc:  # detector missing or failed — say so, don't guess
        log_event(logger, "language_detection_unavailable", level=30, error=str(exc))
        return None


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

    def detect_language(self, text: str) -> str | None:
        return detect_language_offline(text)

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

    def detect_language(self, text: str) -> str | None:
        # Detection does not require a provider, so it still works here even
        # though translation is disabled.
        return detect_language_offline(text)

    def translate(self, text: str, *, source: str, target: str) -> str:
        raise TranslationFailed("Translation is disabled on this deployment.")


def get_translation_provider(provider_name: str) -> TranslationProvider:
    if provider_name == "google":
        return GoogleTranslateProvider()
    return NullTranslationProvider()
