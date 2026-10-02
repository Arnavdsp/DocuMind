from __future__ import annotations

from tests.conftest import read_fixture
from tests.test_api_documents import _upload, _wait_until_ready


def test_ask_returns_grounded_answer_with_citations(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    ask_resp = client.post(
        f"/api/documents/{document_id}/ask",
        json={"question": "What was the sampling frequency used in the experiment?"},
    )
    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert body["question"]
    assert "grounding" in body
    assert 0.0 <= body["relevance_score"] <= 1.0
    assert body["model_used"] == "mock"


def test_ask_abstains_on_unrelated_question(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    ask_resp = client.post(
        f"/api/documents/{document_id}/ask",
        json={"question": "What is the boiling point of liquid nitrogen on Jupiter's moon Europa?"},
    )
    body = ask_resp.json()
    assert body["grounding"] in ("none", "weak")


def test_ask_on_unknown_document_returns_404(client):
    resp = client.post("/api/documents/does-not-exist/ask", json={"question": "anything"})
    assert resp.status_code == 404


def test_ask_rejects_empty_question(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    ask_resp = client.post(f"/api/documents/{document_id}/ask", json={"question": ""})
    assert ask_resp.status_code == 422  # pydantic min_length validation
