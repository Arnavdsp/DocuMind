from __future__ import annotations

import time

from tests.conftest import read_fixture


def _upload(client, filename, content, content_type):
    return client.post(
        "/api/documents", files={"file": (filename, content, content_type)}
    )


def _wait_until_ready(client, document_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/documents/{document_id}")
        if resp.json()["status"] in ("ready", "failed"):
            return resp.json()
        time.sleep(0.1)
    raise TimeoutError("document never reached a terminal state")


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readiness_endpoint(client):
    resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_upload_txt_document_and_poll_to_ready(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    assert resp.status_code == 201
    body = resp.json()
    document_id = body["document"]["document_id"]

    document = _wait_until_ready(client, document_id)
    assert document["status"] == "ready"
    assert document["metrics"]["word_count"] > 0
    assert document["metrics"]["page_count"] == 1


def test_upload_rejects_invalid_extension(client):
    resp = _upload(client, "malware.exe", b"MZ\x90\x00", "application/octet-stream")
    assert resp.status_code == 415
    assert resp.json()["detail"]["error_code"] == "unsupported_file_type"


def test_upload_rejects_spoofed_pdf(client):
    resp = _upload(client, "fake.pdf", read_fixture("fake.pdf"), "application/pdf")
    assert resp.status_code == 422


def test_get_unknown_document_returns_404(client):
    resp = client.get("/api/documents/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "document_not_found"


def test_list_documents_includes_uploaded_document(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    listing = client.get("/api/documents")
    ids = [d["document_id"] for d in listing.json()["documents"]]
    assert document_id in ids


def test_delete_document_removes_it(client):
    resp = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = resp.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    delete_resp = client.delete(f"/api/documents/{document_id}")
    assert delete_resp.status_code == 204
    assert client.get(f"/api/documents/{document_id}").status_code == 404


def test_reuploading_identical_bytes_is_a_cache_hit(client):
    first = _upload(client, "sample.txt", read_fixture("sample.txt"), "text/plain")
    document_id = first.json()["document"]["document_id"]
    _wait_until_ready(client, document_id)

    second = _upload(client, "sample_again.txt", read_fixture("sample.txt"), "text/plain")
    assert second.json()["document"]["document_id"] == document_id
