from __future__ import annotations

import hashlib
import hmac
import json


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_webhook_rejects_invalid_signature(client):
    payload = {"ref": "refs/heads/main", "repository": {"full_name": "owner/repo"}, "commits": []}
    body = json.dumps(payload).encode()

    resp = client.post(
        "/api/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=invalidsignature",
            "X-GitHub-Event": "push",
        },
    )
    assert resp.status_code == 401


def test_webhook_accepts_valid_signature(client):
    payload = {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": "a" * 40,
        "repository": {"full_name": "owner/repo"},
        "commits": [],
    }
    body = json.dumps(payload).encode()

    secret = "test-secret"
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    resp = client.post(
        "/api/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": "push",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["task_id"]


def test_webhook_ignores_non_push(client):
    payload = {"action": "opened"}
    body = json.dumps(payload).encode()

    secret = "test-secret"
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    resp = client.post(
        "/api/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": "pull_request",
        },
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "ignored"
