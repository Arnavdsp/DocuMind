"""Structured-summary integrity (FR-21).

Regression cover for a real defect found by running an 80-page document
through the map-reduce path:

The reduce step emits a JSON object carrying up to 10 findings and 15
numbers. At `generation_max_new_tokens = 500` that object was cut mid-string,
`json.loads` failed, and the fallback put `cleaned[:1500]` straight into
`executive_summary` — so the user was shown a raw `{"executive_summary":"...`
blob as prose, with key_findings and important_numbers silently emptied.

Silent degradation that *looks* like output is worse than a loud failure, and
these tests pin both halves of the fix: a bigger budget for structured calls,
and a fallback that salvages prose rather than leaking JSON syntax.
"""

from __future__ import annotations

from app.config import Settings
from app.services.summarization_service import (
    _parse_structured_output,
    _salvage_executive_summary,
    _structured_budget,
)

# What a truncated structured response actually looks like: a complete first
# field, then the cut.
TRUNCATED = (
    '{"executive_summary":"The document reports a series of measured values for a '
    'configuration under test, presented in sequential sections.","key_findings":'
    '["Finding 1 reports 7.2 percent","Finding 2 reports 10.2 perc'
)


# --- the budget ----------------------------------------------------------------


def test_structured_budget_exceeds_the_plain_generation_budget():
    settings = Settings()
    assert _structured_budget(settings) > settings.generation_max_new_tokens


def test_structured_budget_never_falls_below_the_generation_budget():
    """Configuring one upward must not starve the other."""
    settings = Settings(generation_max_new_tokens=4000, summarize_max_new_tokens=100)
    assert _structured_budget(settings) == 4000


# --- salvage -------------------------------------------------------------------


def test_salvage_recovers_prose_from_truncated_json():
    salvaged = _salvage_executive_summary(TRUNCATED)
    assert salvaged is not None
    assert salvaged.startswith("The document reports")
    assert "{" not in salvaged and '"key_findings"' not in salvaged


def test_salvage_handles_escaped_quotes():
    payload = '{"executive_summary":"The report calls this a \\"hybrid\\" configuration.","key_'
    assert _salvage_executive_summary(payload) == 'The report calls this a "hybrid" configuration.'


def test_salvage_returns_none_when_there_is_nothing_to_recover():
    assert _salvage_executive_summary("not json at all") is None
    assert _salvage_executive_summary('{"key_findings":["a"]') is None


# --- the parse path -------------------------------------------------------------


def test_truncated_json_does_not_leak_into_the_summary():
    """The actual defect: raw JSON rendered to the user as prose."""
    result = _parse_structured_output(TRUNCATED)
    assert not result.executive_summary.lstrip().startswith("{")
    assert '"executive_summary"' not in result.executive_summary
    assert result.executive_summary.startswith("The document reports")


def test_unsalvageable_json_says_so_rather_than_showing_syntax():
    result = _parse_structured_output('{"key_findings":["a","b"],"important_numb')
    assert not result.executive_summary.lstrip().startswith("{")
    assert "could not be assembled" in result.executive_summary


def test_plain_prose_still_passes_through():
    """A model that ignores the format instruction entirely is not a failure —
    its prose is still a usable summary."""
    result = _parse_structured_output("This document describes a retrieval evaluation.")
    assert result.executive_summary == "This document describes a retrieval evaluation."


def test_well_formed_json_is_parsed_fully():
    payload = (
        '{"executive_summary":"A summary.","key_findings":["one","two"],'
        '"important_numbers":["71.2%"],"methodology":null,"limitations":null}'
    )
    result = _parse_structured_output(payload)
    assert result.executive_summary == "A summary."
    assert result.key_findings == ["one", "two"]
    assert result.important_numbers == ["71.2%"]
    assert result.methodology is None


def test_fenced_json_is_parsed():
    payload = '```json\n{"executive_summary":"Fenced.","key_findings":[]}\n```'
    assert _parse_structured_output(payload).executive_summary == "Fenced."


def test_empty_output_reports_rather_than_returning_blank():
    assert _parse_structured_output("").executive_summary == "No summary could be generated."
