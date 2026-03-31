from __future__ import annotations


def test_query_endpoint_requires_repo(client):
    resp = client.post("/api/query", json={"query": "find main function"})
    assert resp.status_code == 422


def test_query_endpoint_validation(client):
    resp = client.post(
        "/api/query",
        json={"query": "test", "repo": "owner/repo", "top_k": 100},
    )
    # top_k > 50 should fail validation
    assert resp.status_code == 422
