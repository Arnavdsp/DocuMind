"""Groq generation backend.

Every test here is network-free: the HTTP layer is stubbed, so these assert
the *contract* — truthful backend naming, loud failure on a missing key,
retry only on retryable statuses — rather than that Groq is reachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.services.model_service import GroqModelService, MockModelService
from app.utils.errors import ModelUnavailable


class _StubResponse:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = str(self._payload)

    def json(self) -> dict:
        return self._payload


def _ok(content: str) -> _StubResponse:
    return _StubResponse(200, {"choices": [{"message": {"content": content}}]})


def _settings(**overrides) -> Settings:
    base = {"groq_api_key": "test-key-not-a-real-credential", "groq_max_retries": 3}
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff is real time; tests must not pay for it."""
    monkeypatch.setattr("app.services.model_service._sleep_backoff", lambda *a, **k: None)


def _patch_post(monkeypatch, responses: list):
    """Stub httpx.post with a queue of responses; records the calls made."""
    import httpx

    calls: list[dict] = []
    queue = list(responses)

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# --- identity ---------------------------------------------------------------


def test_backend_name_reports_the_groq_model():
    service = GroqModelService(_settings(groq_model="openai/gpt-oss-20b"), MockModelService())
    assert "groq:openai/gpt-oss-20b" in service.backend_name


def test_backend_name_discloses_a_mock_embedding_delegate():
    """Generation can be real while retrieval scores are meaningless. The
    client must be able to tell — model_used is rendered on screen."""
    service = GroqModelService(_settings(), MockModelService())
    assert service.backend_name == "groq:openai/gpt-oss-20b (embeddings: mock)"


def test_device_info_names_both_halves():
    info = GroqModelService(_settings(), MockModelService()).device_info
    assert "groq api" in info["generation"]
    assert "device" in info


# --- delegation --------------------------------------------------------------


def test_embed_is_delegated_to_the_local_service():
    local = MockModelService()
    service = GroqModelService(_settings(), local)
    vectors = service.embed(["hybrid retrieval"])
    assert isinstance(vectors, np.ndarray)
    np.testing.assert_array_equal(vectors, local.embed(["hybrid retrieval"]))


def test_extractive_qa_raises_rather_than_fabricating_a_span():
    service = GroqModelService(_settings(), MockModelService())
    with pytest.raises(ModelUnavailable):
        service.extractive_qa("question", "context")


# --- generation --------------------------------------------------------------


def test_generate_returns_the_message_content(monkeypatch):
    calls = _patch_post(monkeypatch, [_ok("  71.2 percent.  ")])
    service = GroqModelService(_settings(), MockModelService())

    answer = service.generate("sys", "user", max_new_tokens=100, temperature=0.0)

    assert answer == "71.2 percent."
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_generate_sends_reasoning_effort_when_configured(monkeypatch):
    calls = _patch_post(monkeypatch, [_ok("ok")])
    service = GroqModelService(_settings(groq_reasoning_effort="low"), MockModelService())
    service.generate("sys", "user", max_new_tokens=100, temperature=0.0)
    assert calls[0]["json"]["reasoning_effort"] == "low"


def test_generate_omits_reasoning_effort_when_unset(monkeypatch):
    calls = _patch_post(monkeypatch, [_ok("ok")])
    service = GroqModelService(_settings(groq_reasoning_effort=None), MockModelService())
    service.generate("sys", "user", max_new_tokens=100, temperature=0.0)
    assert "reasoning_effort" not in calls[0]["json"]


def test_missing_key_raises_instead_of_falling_back(monkeypatch):
    """A silent fallback to another backend would make model_used a lie."""
    _patch_post(monkeypatch, [])
    service = GroqModelService(_settings(groq_api_key=None), MockModelService())
    with pytest.raises(ModelUnavailable):
        service.generate("sys", "user", max_new_tokens=100, temperature=0.0)


def test_rate_limit_is_retried_then_succeeds(monkeypatch, no_sleep):
    calls = _patch_post(
        monkeypatch, [_StubResponse(429, {}, {"retry-after": "0"}), _ok("recovered")]
    )
    service = GroqModelService(_settings(), MockModelService())
    assert service.generate("sys", "user", max_new_tokens=100, temperature=0.0) == "recovered"
    assert len(calls) == 2


def test_server_error_is_retried(monkeypatch, no_sleep):
    calls = _patch_post(monkeypatch, [_StubResponse(503), _ok("recovered")])
    service = GroqModelService(_settings(), MockModelService())
    assert service.generate("sys", "user", max_new_tokens=100, temperature=0.0) == "recovered"
    assert len(calls) == 2


def test_auth_failure_is_not_retried(monkeypatch, no_sleep):
    """401 will never change on retry; burning the budget on it is wrong."""
    calls = _patch_post(monkeypatch, [_StubResponse(401, {"error": "invalid api key"})])
    service = GroqModelService(_settings(), MockModelService())
    with pytest.raises(ModelUnavailable):
        service.generate("sys", "user", max_new_tokens=100, temperature=0.0)
    assert len(calls) == 1


def test_exhausted_retries_raise(monkeypatch, no_sleep):
    calls = _patch_post(monkeypatch, [_StubResponse(429) for _ in range(3)])
    service = GroqModelService(_settings(groq_max_retries=3), MockModelService())
    with pytest.raises(ModelUnavailable):
        service.generate("sys", "user", max_new_tokens=100, temperature=0.0)
    assert len(calls) == 3


def test_unexpected_response_shape_raises(monkeypatch, no_sleep):
    _patch_post(monkeypatch, [_StubResponse(200, {"choices": []})])
    service = GroqModelService(_settings(), MockModelService())
    with pytest.raises(ModelUnavailable):
        service.generate("sys", "user", max_new_tokens=100, temperature=0.0)


def test_api_key_is_not_present_in_the_raised_message(monkeypatch, no_sleep):
    """Errors reach logs and error handlers; the credential must not ride along."""
    secret = "gsk_secret_value_that_must_not_leak"
    _patch_post(monkeypatch, [_StubResponse(401, {"error": "nope"})])
    service = GroqModelService(_settings(groq_api_key=secret), MockModelService())
    with pytest.raises(ModelUnavailable) as excinfo:
        service.generate("sys", "user", max_new_tokens=100, temperature=0.0)
    assert secret not in str(excinfo.value)
    assert secret not in repr(excinfo.value)
