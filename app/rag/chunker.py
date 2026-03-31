from __future__ import annotations

import tiktoken
from langchain_core.documents import Document

from app.services.parser import CodeChunk, detect_language, parse_file
from app.utils.logging import get_logger

logger = get_logger(__name__)

MAX_CHUNK_TOKENS = 500
OVERLAP_LINES = 5

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def _split_large_chunk(chunk: CodeChunk, repo: str, file_path: str) -> list[Document]:
    """Split a chunk that exceeds MAX_CHUNK_TOKENS into overlapping segments."""
    lines = chunk.code.splitlines(keepends=True)
    documents: list[Document] = []
    start = 0
    part = 0

    while start < len(lines):
        end = start
        current = ""
        while end < len(lines):
            candidate = current + lines[end]
            if _count_tokens(candidate) > MAX_CHUNK_TOKENS and end > start:
                break
            current = candidate
            end += 1

        documents.append(
            _make_document(
                code=current,
                chunk=chunk,
                repo=repo,
                file_path=file_path,
                start_line=chunk.start_line + start,
                end_line=chunk.start_line + end - 1,
                part=part,
            )
        )
        part += 1
        start = max(start + 1, end - OVERLAP_LINES)

    return documents


def _make_document(
    code: str,
    chunk: CodeChunk,
    repo: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    part: int | None = None,
    commit_id: str = "",
) -> Document:
    """Build a LangChain Document with rich metadata for retrieval and filtering."""
    chunk_id = f"{repo}:{file_path}:{chunk.name}"
    if part is not None:
        chunk_id += f":part{part}"

    return Document(
        page_content=code,
        metadata={
            "chunk_id": chunk_id,
            "repo": repo,
            "file_path": file_path,
            "function_name": chunk.name if chunk.chunk_type in ("function", "method") else "",
            "class_name": chunk.parent_class or (chunk.name if chunk.chunk_type == "class" else ""),
            "chunk_type": chunk.chunk_type,
            "language": chunk.language,
            "start_line": start_line or chunk.start_line,
            "end_line": end_line or chunk.end_line,
            "docstring": chunk.docstring,
            "commit_id": commit_id,
        },
    )


def chunk_code(file_path: str, content: str, repo: str, commit_id: str = "") -> list[Document]:
    """Parse and chunk a source file into LangChain Documents.

    Uses tree-sitter AST parsing for supported languages (Python, JS, TS) to
    extract function-level and class-level chunks. Falls back to full-file
    chunking for unsupported languages.

    Args:
        file_path: Path of the file within the repo.
        content: Raw source code content.
        repo: Repository identifier (owner/name).
        commit_id: Git commit SHA for traceability.

    Returns:
        List of LangChain Document objects with metadata.
    """
    language = detect_language(file_path)

    if language is None:
        # Unsupported language — treat as plain text, single chunk
        return [
            Document(
                page_content=content,
                metadata={
                    "chunk_id": f"{repo}:{file_path}:<module>",
                    "repo": repo,
                    "file_path": file_path,
                    "function_name": "",
                    "class_name": "",
                    "chunk_type": "module",
                    "language": "unknown",
                    "start_line": 1,
                    "end_line": len(content.splitlines()),
                    "docstring": "",
                    "commit_id": commit_id,
                },
            )
        ]

    parsed_chunks = parse_file(content, language)
    documents: list[Document] = []

    for chunk in parsed_chunks:
        token_count = _count_tokens(chunk.code)

        if token_count > MAX_CHUNK_TOKENS:
            documents.extend(_split_large_chunk(chunk, repo, file_path))
        else:
            documents.append(_make_document(
                code=chunk.code, chunk=chunk, repo=repo,
                file_path=file_path, commit_id=commit_id,
            ))

    logger.info(
        "chunked_file",
        file_path=file_path,
        language=language,
        num_chunks=len(documents),
    )
    return documents
