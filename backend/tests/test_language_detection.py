"""Offline source-language detection (D5).

The previous implementation called `deep_translator.single_detection` with
`api_key=None`. That function requires a detectlanguage.com key, so it raised
on every call and the caller's bare `except` returned "en" — source language
was hardcoded to English and reported as though it had been measured.

These tests pin the property that broke: an undetermined language must be
distinguishable from a detected one.
"""

from __future__ import annotations

from app.services.translation_service import (
    GoogleTranslateProvider,
    NullTranslationProvider,
    detect_language_offline,
    get_translation_provider,
)

ENGLISH = "The sampling frequency was 20 MHz during the experiment and remained stable."
FRENCH = "La frequence d echantillonnage etait de 20 MHz pendant toute l experience realisee."
GERMAN = "Die Abtastfrequenz betrug waehrend des gesamten Experiments zwanzig Megahertz."


def test_detects_english():
    assert detect_language_offline(ENGLISH) == "en"


def test_detects_a_non_english_language():
    """The original defect was invisible precisely because English was the
    hardcoded answer — only a non-English input can catch it."""
    assert detect_language_offline(FRENCH) != "en"


def test_distinguishes_two_non_english_languages():
    assert detect_language_offline(FRENCH) != detect_language_offline(GERMAN)


def test_empty_text_returns_none_not_a_default():
    assert detect_language_offline("") is None
    assert detect_language_offline("   \n  ") is None


def test_detection_requires_no_network_or_api_key(monkeypatch):
    """Guards NFR-8. If detection ever reaches deep_translator again, this
    fails rather than silently degrading back to the hardcoded default."""

    def explode(*args, **kwargs):
        raise AssertionError("language detection must not call a remote service")

    import deep_translator

    monkeypatch.setattr(deep_translator, "single_detection", explode, raising=False)
    assert detect_language_offline(ENGLISH) == "en"


def test_both_providers_detect_without_translating():
    """Detection does not depend on a translation provider being enabled."""
    assert GoogleTranslateProvider().detect_language(FRENCH) != "en"
    assert NullTranslationProvider().detect_language(FRENCH) != "en"


def test_null_provider_still_detects_when_translation_is_disabled():
    provider = get_translation_provider("none")
    assert provider.name == "none"
    assert provider.detect_language(ENGLISH) == "en"


def test_detector_failure_returns_none_rather_than_guessing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fail_py3langid(name, *args, **kwargs):
        if name == "py3langid":
            raise ImportError("simulated: detector unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_py3langid)
    assert detect_language_offline(ENGLISH) is None


# --- content-loss detection (FR-22) -------------------------------------------


def test_length_ratio_is_reported_as_a_measurement():
    from app.services.translation_service import TranslationResult

    result = TranslationResult(text="abcd", segments_total=1, segments_translated=1, input_chars=8)
    assert result.length_ratio == 0.5


def test_no_input_reports_a_null_ratio_not_zero():
    from app.services.translation_service import TranslationResult

    assert TranslationResult(text="", segments_total=0, segments_translated=0).length_ratio is None


def test_missing_segment_is_content_dropped():
    from app.services.translation_service import TranslationResult

    result = TranslationResult(text="x", segments_total=3, segments_translated=2, input_chars=10)
    assert result.content_dropped
    assert result.dropped_reason == "segment_missing"


def test_implausibly_short_output_is_caught_even_with_every_segment_present():
    """The real failure: a model returns every segment but collapses content
    inside them. Measured at 5,520 chars in and 1,686 out on a repetitive
    document, with segments 2/2 — segment counting alone called that clean."""
    from app.services.translation_service import TranslationResult

    result = TranslationResult(text="x" * 1686, segments_total=2, segments_translated=2, input_chars=5520)
    assert result.content_dropped
    assert result.dropped_reason == "output_length_implausible"


def test_a_normal_translation_is_not_flagged():
    from app.services.translation_service import TranslationResult

    result = TranslationResult(text="y" * 1050, segments_total=2, segments_translated=2, input_chars=1000)
    assert not result.content_dropped
    assert result.dropped_reason is None
