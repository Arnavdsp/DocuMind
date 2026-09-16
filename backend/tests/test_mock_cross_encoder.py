"""Deterministic mock cross-encoder (FR-10, GAP-10).

Before this existed, `MockModelService.get_cross_encoder` raised, so every
assertion about reranking needed a GPU and a multi-gigabyte download — which
meant, in practice, that none were written. These run with no GPU and no
network, and cover the properties rerank tests actually depend on: stable
ordering, a total order under ties, and bounded output.

The scores themselves are meaningless as a quality signal. Nothing here
asserts they are good, only that they are deterministic and correctly shaped.
"""

from __future__ import annotations

from app.rag.chunking import Chunk
from app.rag.reranker import CrossEncoderReranker, build_reranker
from app.rag.vector_store import ScoredChunk
from app.services.model_service import MockModelService

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=f"doc:c{idx}",
        page_number=idx + 1,
        section=None,
        text=text,
        start_offset=0,
        end_offset=len(text),
        token_estimate=len(text.split()),
    )


def _candidates(*texts: str) -> list[ScoredChunk]:
    # Uniform incoming score, so any reordering is attributable to the
    # cross-encoder rather than to the retrieval score leaking through.
    return [ScoredChunk(chunk=_chunk(i, t), score=0.5) for i, t in enumerate(texts)]


# --- the scorer ---------------------------------------------------------------


def test_mock_backend_returns_a_cross_encoder_instead_of_raising():
    encoder = MockModelService().get_cross_encoder(MODEL)
    assert hasattr(encoder, "predict")
    assert encoder.model_name == MODEL


def test_scores_are_labelled_as_mock():
    """The score's kind travels with it; a mock score must never read as real."""
    assert MockModelService().get_cross_encoder(MODEL).score_kind == "mock_cross_encoder"


def test_predict_returns_one_score_per_pair():
    encoder = MockModelService().get_cross_encoder(MODEL)
    scores = encoder.predict([("q", "a"), ("q", "b"), ("q", "c")])
    assert len(scores) == 3


def test_scores_are_deterministic_across_instances():
    pairs = [("sampling frequency", "The sampling frequency was 20 MHz.")]
    first = MockModelService().get_cross_encoder(MODEL).predict(pairs)
    second = MockModelService().get_cross_encoder(MODEL).predict(pairs)
    assert first == second


def test_scores_are_bounded_to_the_unit_interval():
    """Citation.relevance_score is clamped to [0, 1]; emitting outside that
    range would fabricate values through the clamp (see D2 in the phase log)."""
    encoder = MockModelService().get_cross_encoder(MODEL)
    pairs = [
        ("short", "short"),
        ("a much longer query with many repeated repeated tokens", "a much longer query with many repeated repeated tokens"),
        ("nothing", "shared"),
    ]
    for score in encoder.predict(pairs):
        assert 0.0 <= score <= 1.0


def test_a_passage_sharing_query_terms_outscores_one_that_does_not():
    encoder = MockModelService().get_cross_encoder(MODEL)
    relevant, irrelevant = encoder.predict(
        [
            ("sampling frequency experiment", "The sampling frequency was 20 MHz during the experiment."),
            ("sampling frequency experiment", "Bananas are a good source of potassium."),
        ]
    )
    assert relevant > irrelevant


def test_empty_query_or_passage_scores_zero():
    encoder = MockModelService().get_cross_encoder(MODEL)
    assert encoder.predict([("", "some passage")]) == [0.0]
    assert encoder.predict([("some query", "")]) == [0.0]


def test_identical_passages_score_identically_regardless_of_position():
    """Tie-breaking must depend on content, not on list index — otherwise
    rerank ordering would shift under an unrelated retrieval change."""
    encoder = MockModelService().get_cross_encoder(MODEL)
    scores = encoder.predict([("query", "same text"), ("other", "x"), ("query", "same text")])
    assert scores[0] == scores[2]


# --- through the reranker ------------------------------------------------------


def test_cross_encoder_reranker_reorders_candidates_on_the_mock_backend():
    candidates = _candidates(
        "Bananas are a good source of potassium and fiber.",
        "The sampling frequency was 20 MHz during the experiment.",
    )
    reranked = CrossEncoderReranker(MODEL, MockModelService()).rerank(
        "what sampling frequency was used in the experiment", candidates, top_k=2
    )
    assert reranked[0].chunk.text.startswith("The sampling frequency")


def test_cross_encoder_reranker_respects_top_k():
    candidates = _candidates("alpha one", "beta two", "gamma three", "delta four")
    reranked = CrossEncoderReranker(MODEL, MockModelService()).rerank("alpha", candidates, top_k=2)
    assert len(reranked) == 2


def test_cross_encoder_reranker_handles_no_candidates():
    assert CrossEncoderReranker(MODEL, MockModelService()).rerank("q", [], top_k=4) == []


def test_rerank_ordering_is_stable_across_runs():
    candidates = _candidates("alpha one", "beta two", "gamma three", "alpha one")
    rerank = lambda: [  # noqa: E731
        sc.chunk.chunk_id
        for sc in CrossEncoderReranker(MODEL, MockModelService()).rerank("alpha", candidates, top_k=4)
    ]
    assert rerank() == rerank()


def test_configuring_a_reranker_model_no_longer_breaks_the_mock_backend():
    """Previously this path raised ModelUnavailable from inside rerank(),
    surfacing as an uncaught 500 rather than a degraded result."""
    reranker = build_reranker(reranker_model=MODEL, model_service=MockModelService())
    assert isinstance(reranker, CrossEncoderReranker)
    result = reranker.rerank("alpha", _candidates("alpha one", "beta two"), top_k=1)
    assert len(result) == 1


def test_default_config_still_selects_the_lexical_reranker():
    """reranker_model defaults to None; this change must not alter that."""
    from app.rag.reranker import LexicalOverlapReranker

    reranker = build_reranker(reranker_model=None, model_service=MockModelService())
    assert isinstance(reranker, LexicalOverlapReranker)
