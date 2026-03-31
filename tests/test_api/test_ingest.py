from __future__ import annotations


def test_ingest_rejects_invalid_url(client):
    resp = client.post(
        "/api/ingest",
        json={"repo_url": "not-a-github-url", "branch": "main"},
    )
    # Should fail in URL parsing
    assert resp.status_code in (400, 422, 500)


def test_query_requires_fields(client):
    resp = client.post("/api/query", json={})
    assert resp.status_code == 422
