from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Set test env vars before importing the app
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("GITHUB_TOKEN", "ghp-test-token")
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/chroma_test")
os.environ.setdefault("APP_ENV", "development")


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c
