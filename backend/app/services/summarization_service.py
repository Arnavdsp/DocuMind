"""Summarization strategy.

Short documents are summarized directly. Long documents go through a
map-reduce pipeline (page-group chunk summaries -> synthesis of those
summaries) instead of being truncated to an arbitrary character count and
having the rest silently discarded, as the original implementation did.
"""

from __future__ import annotations

import json
import re

from app.config import Settings
from app.ingestion.extractors import ExtractedPage
from app.schemas.summary import StructuredSummary
from app.services.model_service import ModelService

# Below this word count, summarize the whole document in one call.
_DIRECT_SUMMARY_WORD_THRESHOLD = 1800
# Group pages into batches of roughly this many words per map-step call.
_MAP_STEP_WORD_BUDGET = 1500

_STRUCTURED_INSTRUCTIONS = (
    "Summarize the provided document text. Preserve technical terminology, "
    "numbers, named claims, methodology, and limitations exactly as stated — "
    "never invent details that aren't in the text. "
    "Respond with ONLY a JSON object (no markdown fences, no commentary) matching "
    'this shape: {"executive_summary": string, "key_findings": [string], '
    '"important_numbers": [string], "methodology": string or null, '
    '"limitations": string or null}. If the document doesn\'t have a distinct '
    "methodology or limitations section, use null for that field."
)

_MAP_INSTRUCTIONS = (
    "Write a dense, factual summary of this excerpt in 3-5 sentences. Preserve "
    "specific numbers, technical terms, and claims verbatim where possible. "
    "Do not add information that isn't present in the excerpt."
)


def _group_pages_by_word_budget(pages: list[ExtractedPage], word_budget: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_words = 0
    for page in pages:
        words = len(page.text.split())
        if current and current_words + words > word_budget:
            groups.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(page.text)
        current_words += words
    if current:
        groups.append("\n\n".join(current))
    return groups


def _structured_budget(settings: Settings) -> int:
    """Token budget for a structured-JSON call.

    Never below the plain generation budget, so configuring one upward cannot
    accidentally starve the other.
    """
    return max(
        getattr(settings, "summarize_max_new_tokens", 0) or 0,
        settings.generation_max_new_tokens,
    )


def _parse_structured_output(raw: str) -> StructuredSummary:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return StructuredSummary(
            executive_summary=str(data.get("executive_summary", "")).strip() or cleaned[:800],
            key_findings=[str(x) for x in data.get("key_findings", [])][:10],
            important_numbers=[str(x) for x in data.get("important_numbers", [])][:15],
            methodology=data.get("methodology") or None,
            limitations=data.get("limitations") or None,
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        # The model did not return parseable JSON. Small and mock models will
        # not reliably follow a format instruction, and a long structured
        # object can be cut mid-string by the token budget.
        #
        # The previous behaviour here was to put `cleaned[:1500]` straight into
        # executive_summary — which, when the output was truncated JSON, meant
        # rendering a raw `{"executive_summary":"...` blob to the user as
        # though it were prose, with key_findings and important_numbers
        # silently emptied. Salvage the prose instead, and say so when it
        # cannot be salvaged.
        salvaged = _salvage_executive_summary(cleaned)
        if salvaged:
            return StructuredSummary(executive_summary=salvaged)
        if cleaned.lstrip().startswith("{"):
            return StructuredSummary(
                executive_summary=(
                    "The summary could not be assembled: the model returned malformed "
                    "or truncated structured output. Raw output withheld rather than "
                    "shown as prose."
                )
            )
        return StructuredSummary(executive_summary=cleaned[:1500] or "No summary could be generated.")


def _salvage_executive_summary(cleaned: str) -> str | None:
    """Pull the executive_summary string out of truncated JSON.

    A cut-off object still usually carries a complete first field. Recovering
    it is strictly better than discarding the call, and strictly better than
    showing the reader JSON syntax.
    """
    match = re.search(r'"executive_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"').strip() or None
    except json.JSONDecodeError:
        return None


def summarize_document(
    pages: list[ExtractedPage],
    *,
    model_service: ModelService,
    settings: Settings,
) -> tuple[StructuredSummary, str]:
    full_text = "\n\n".join(p.text for p in pages if p.text)
    total_words = len(full_text.split())

    if total_words == 0:
        return (
            StructuredSummary(executive_summary="This document has no extractable text to summarize."),
            "direct",
        )

    if total_words <= _DIRECT_SUMMARY_WORD_THRESHOLD:
        raw = model_service.generate(
            _STRUCTURED_INSTRUCTIONS,
            full_text,
            max_new_tokens=_structured_budget(settings),
            temperature=settings.generation_temperature,
        )
        return _parse_structured_output(raw), "direct"

    # Map step: summarize page groups independently.
    groups = _group_pages_by_word_budget(pages, _MAP_STEP_WORD_BUDGET)
    group_summaries = [
        model_service.generate(
            _MAP_INSTRUCTIONS, group, max_new_tokens=220, temperature=settings.generation_temperature
        )
        for group in groups
    ]

    # Reduce step: synthesize the group summaries into one structured summary.
    synthesis_input = "\n\n".join(f"Section {i + 1} summary: {s}" for i, s in enumerate(group_summaries))
    raw = model_service.generate(
        _STRUCTURED_INSTRUCTIONS,
        synthesis_input,
        max_new_tokens=_structured_budget(settings),
        temperature=settings.generation_temperature,
    )
    return _parse_structured_output(raw), "map_reduce"
