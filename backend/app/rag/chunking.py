"""Page-aware chunking with metadata sufficient for citations.

Chunks are built by first splitting each page into sentences, then packing
sentences into chunks up to a target token budget (approximated via a
whitespace-word heuristic — good enough for chunk sizing; the embedding
model's own tokenizer governs the true truncation point). Chunking never
crosses a page boundary, so every chunk can be attributed to exactly one
page for citation purposes, and a sliding overlap keeps context continuous
between adjacent chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.extractors import ExtractedPage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    page_number: int | None
    section: str | None
    text: str
    start_offset: int
    end_offset: int
    token_estimate: int


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences with their (start, end) offsets in the original, unstripped text."""
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for sentence in _split_sentences(text):
        start = text.find(sentence, pos)
        if start < 0:  # cannot happen for substrings in order; keep offsets monotonic anyway
            start = pos
        end = start + len(sentence)
        spans.append((sentence, start, end))
        pos = end
    return spans


def _estimate_tokens(text: str) -> int:
    # A word roughly maps to ~1.3 tokens for common English tokenizers;
    # this is only used for chunk-size budgeting, not billing or truncation.
    return max(1, int(len(text.split()) * 1.3))


def chunk_page(
    page: ExtractedPage,
    *,
    document_id: str,
    target_tokens: int,
    overlap_tokens: int,
    section: str | None = None,
) -> list[Chunk]:
    spans = _sentence_spans(page.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        chunk_id = f"{document_id}:p{page.page_number}:{len(chunks)}"
        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page.page_number,
                section=section,
                text=" ".join(s for s, _, _ in current),
                # offsets index into page.text, so a citation can slice the original span
                start_offset=current[0][1],
                end_offset=current[-1][2],
                token_estimate=current_tokens,
            )
        )

    for span in spans:
        sentence_tokens = _estimate_tokens(span[0])

        if current and current_tokens + sentence_tokens > target_tokens:
            flush()
            # keep the tail of the previous chunk for overlap continuity
            overlap: list[tuple[str, int, int]] = []
            overlap_count = 0
            for prev in reversed(current):
                t = _estimate_tokens(prev[0])
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_count += t
            current = overlap
            current_tokens = overlap_count

        current.append(span)
        current_tokens += sentence_tokens

    flush()
    return chunks


def chunk_document(
    pages: list[ExtractedPage],
    *,
    document_id: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError(
            f"overlap_tokens must be in [0, target_tokens), got {overlap_tokens} with target {target_tokens}"
        )
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(
            chunk_page(
                page,
                document_id=document_id,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return all_chunks
