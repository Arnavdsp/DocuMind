from __future__ import annotations

from tests.conftest import read_fixture
from tests.test_api_documents import _upload, _wait_until_ready


def test_summarize_returns_structured_summary(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    summary_resp = client.post(f"/api/documents/{document_id}/summarize", json={})
    assert summary_resp.status_code == 200
    body = summary_resp.json()
    assert body["summary"]["executive_summary"]
    assert body["strategy"] == "direct"
    assert body["cached"] is False


def test_summarize_is_cached_on_second_call(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    client.post(f"/api/documents/{document_id}/summarize", json={})
    second = client.post(f"/api/documents/{document_id}/summarize", json={})
    assert second.json()["cached"] is True


def test_summarize_force_refresh_bypasses_cache(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    client.post(f"/api/documents/{document_id}/summarize", json={})
    refreshed = client.post(f"/api/documents/{document_id}/summarize", json={"force_refresh": True})
    assert refreshed.json()["cached"] is False


def test_summarize_on_unknown_document_returns_404(client):
    summary_resp = client.post("/api/documents/definitely-not-a-real-id/summarize", json={})
    assert summary_resp.status_code == 404
