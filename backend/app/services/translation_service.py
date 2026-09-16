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
from dataclasses import dataclass

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


class GroqTranslationProvider(TranslationProvider):
    """Translate with the generation model already configured for this app.

    Why this exists: `GoogleTranslateProvider` uses deep_translator's
    unofficial endpoint, which rate-limits by source IP. Hugging Face Spaces
    share egress IPs, so it returns "too many requests" there regardless of
    how little this app sends — measured, not assumed.

    The deployment plan's alternative was a local model (m2m100_418M at
    ~1.9 GB, or opus-mt pairs at ~300 MB each). This costs no extra disk, no
    extra download, stays at $0, and avoids the CC-BY-NC licence trap that
    rules out NLLB for an MIT repo. The trade-off, stated plainly: an
    instruction-tuned general model is not a dedicated NMT model, and its
    output should not be presented as one.
    """

    name = "groq"

    _SYSTEM = (
        "You are a translation engine. Translate the user's text into {target}. "
        "Output ONLY the translation — no preamble, no notes, no quotes, no "
        "explanation. Preserve numbers, names, units and formatting exactly. "
        "If a passage is already in the target language, return it unchanged."
    )

    def __init__(self, model_service):
        self._model_service = model_service

    def detect_language(self, text: str) -> str | None:
        return detect_language_offline(text)

    def translate(self, text: str, *, source: str, target: str) -> str:
        if not text.strip():
            return ""
        try:
            return self._model_service.generate(
                self._SYSTEM.format(target=target),
                text,
                # Generous headroom: translations run longer than their source
                # in many target languages, and a clipped translation would be
                # exactly the silent truncation this pipeline exists to avoid.
                max_new_tokens=max(1024, len(text.split()) * 4),
                temperature=0.0,
            ).strip()
        except Exception as exc:
            raise TranslationFailed(internal_detail=str(exc)) from exc


def get_translation_provider(provider_name: str, model_service=None) -> TranslationProvider:
    if provider_name == "groq":
        if model_service is None:
            raise TranslationFailed(
                internal_detail="the groq translation provider needs a model service"
            )
        return GroqTranslationProvider(model_service)
    if provider_name == "google":
        return GoogleTranslateProvider()
    return NullTranslationProvider()


@dataclass(frozen=True)
class TranslationResult:
    """What translation actually did, not what it was asked to do.

    `TranslateResponse.truncated` used to be passed `False` unconditionally by
    the route — accurate, but unmeasured, which is the same class of defect as
    a fabricated number. These counts make it an observation.
    """

    text: str
    segments_total: int
    segments_translated: int

    @property
    def content_dropped(self) -> bool:
        return self.segments_translated != self.segments_total


def translate_document(
    provider: TranslationProvider, text: str, *, source: str, target: str
) -> TranslationResult:
    """Segment on sentence boundaries, translate each, and count both ends.

    Added alongside `TranslationProvider.translate` rather than changing its
    signature: that method's string return is pinned by existing tests, and
    the rule here is that existing tests are not edited to make new code fit.

    Each segment is passed to the provider individually. The provider chunks
    internally too, but a segment already inside the limit yields exactly one
    chunk, so the counts stay meaningful.
    """
    segments = _sentence_aware_chunks(text, _SAFE_CHUNK_CHARS)
    if not segments:
        return TranslationResult(text="", segments_total=0, segments_translated=0)

    translated: list[str] = []
    for index, segment in enumerate(segments):
        # The Google endpoint documents ~5 requests/second. Pace between
        # segments rather than relying on the retry path to absorb a limit we
        # can simply not exceed.
        if index:
            time.sleep(0.25)
        piece = provider.translate(segment, source=source, target=target)
        if piece and piece.strip():
            translated.append(piece)

    return TranslationResult(
        text=" ".join(translated),
        segments_total=len(segments),
        segments_translated=len(translated),
    )
