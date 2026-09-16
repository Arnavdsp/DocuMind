from __future__ import annotations

import sys
import types

import pytest

from app.services.translation_service import (
    NullTranslationProvider,
    _sentence_aware_chunks,
    get_translation_provider,
)
from app.utils.errors import TranslationFailed


def test_sentence_aware_chunking_never_splits_mid_sentence():
    text = "First sentence here. Second sentence here. " * 200
    chunks = _sentence_aware_chunks(text, max_chars=500)
    for chunk in chunks:
        assert chunk.strip().endswith(".")
    assert "".join(chunks).replace(" ", "") .count("First") == text.count("First")


def test_null_provider_raises_translation_failed():
    provider = NullTranslationProvider()
    with pytest.raises(TranslationFailed):
        provider.translate("hello", source="en", target="hi")
    assert provider.detect_language("hello") == "en"


def test_get_translation_provider_factory():
    assert get_translation_provider("none").name == "none"
    assert get_translation_provider("google").name == "google"


def test_google_provider_translate_chunks_and_joins(monkeypatch):
    """Exercises the retry/chunking logic without making a real network
    call: `deep_translator` is faked so the test is deterministic and
    doesn't depend on external services being reachable."""
    calls: list[str] = []

    class FakeGoogleTranslator:
        def __init__(self, source, target):
            self.source = source
            self.target = target

        def translate(self, text):
            calls.append(text)
            return f"[{self.target}]{text}"

    fake_module = types.ModuleType("deep_translator")
    fake_module.GoogleTranslator = FakeGoogleTranslator
    fake_module.single_detection = lambda text, api_key=None: "en"
    monkeypatch.setitem(sys.modules, "deep_translator", fake_module)

    from app.services.translation_service import GoogleTranslateProvider

    provider = GoogleTranslateProvider()
    result = provider.translate("Hello world. This is a test.", source="en", target="hi")
    assert calls  # at least one chunk was sent
    assert result.startswith("[hi]")


def test_google_provider_retries_then_raises_on_persistent_failure(monkeypatch):
    class AlwaysFailsTranslator:
        def __init__(self, source, target):
            pass

        def translate(self, text):
            raise RuntimeError("simulated provider outage")

    fake_module = types.ModuleType("deep_translator")
    fake_module.GoogleTranslator = AlwaysFailsTranslator
    monkeypatch.setitem(sys.modules, "deep_translator", fake_module)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)  # skip real backoff delay

    from app.services.translation_service import GoogleTranslateProvider

    provider = GoogleTranslateProvider()
    with pytest.raises(TranslationFailed):
        provider.translate("Hello world.", source="en", target="hi")
