from __future__ import annotations

from app.services.rag.chunker import chunk_code


SAMPLE_PYTHON = '''
import os

def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
'''

SAMPLE_JS = '''
import express from 'express';

function greet(name) {
    return `Hello, ${name}!`;
}

class UserService {
    constructor(db) {
        this.db = db;
    }

    async getUser(id) {
        return this.db.find(id);
    }
}

export default UserService;
'''

SAMPLE_TS = '''
interface User {
    id: number;
    name: string;
    email: string;
}

export function createUser(name: string, email: string): User {
    return { id: Date.now(), name, email };
}

export class UserRepository {
    private users: User[] = [];

    add(user: User): void {
        this.users.push(user);
    }

    findById(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}
'''


def test_chunk_python():
    docs = chunk_code("src/app.py", SAMPLE_PYTHON, "test/repo")
    assert len(docs) > 0

    chunk_types = {d.metadata["chunk_type"] for d in docs}
    assert "function" in chunk_types or "method" in chunk_types

    # Should have function and class chunks
    names = {d.metadata.get("function_name") or d.metadata.get("class_name") for d in docs}
    assert "hello" in names or any("hello" in (d.metadata.get("function_name", "")) for d in docs)

    # Verify metadata
    for doc in docs:
        assert doc.metadata["repo"] == "test/repo"
        assert doc.metadata["file_path"] == "src/app.py"
        assert doc.metadata["language"] == "python"


def test_chunk_javascript():
    docs = chunk_code("src/app.js", SAMPLE_JS, "test/repo")
    assert len(docs) > 0
    languages = {d.metadata["language"] for d in docs}
    assert "javascript" in languages


def test_chunk_typescript():
    docs = chunk_code("src/app.ts", SAMPLE_TS, "test/repo")
    assert len(docs) > 0
    languages = {d.metadata["language"] for d in docs}
    assert "typescript" in languages


def test_chunk_unsupported_language():
    docs = chunk_code("config.yaml", "key: value\nother: data", "test/repo")
    assert len(docs) == 1
    assert docs[0].metadata["chunk_type"] == "module"
    assert docs[0].metadata["language"] == "unknown"


def test_chunk_metadata_fields():
    docs = chunk_code("test.py", SAMPLE_PYTHON, "owner/repo")
    required_fields = {
        "chunk_id", "repo", "file_path", "function_name",
        "class_name", "chunk_type", "language", "start_line", "end_line",
    }
    for doc in docs:
        for field in required_fields:
            assert field in doc.metadata, f"Missing metadata field: {field}"


def test_chunk_with_commit_id():
    """commit_id propagates to all chunk metadata when provided."""
    docs = chunk_code("test.py", SAMPLE_PYTHON, "owner/repo", commit_id="abc123def")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["commit_id"] == "abc123def"


def test_chunk_without_commit_id():
    """Omitting commit_id defaults to empty string."""
    docs = chunk_code("test.py", SAMPLE_PYTHON, "owner/repo")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["commit_id"] == ""
