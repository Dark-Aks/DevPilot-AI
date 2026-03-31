from __future__ import annotations

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from app.rag.retriever import retrieve, format_context, _tokenize, _keyword_score


@patch("app.rag.retriever.get_vectorstore")
def test_retrieve_calls_vectorstore(mock_get_vs):
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        Document(
            page_content="def foo(): pass",
            metadata={
                "chunk_id": "repo:f.py:foo",
                "file_path": "f.py",
                "function_name": "foo",
                "class_name": "",
                "chunk_type": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
            },
        )
    ]
    mock_get_vs.return_value = mock_store

    results = retrieve("find foo", "owner/repo", top_k=5, use_reranking=False, use_cache=False)
    assert len(results) == 1
    assert results[0].metadata["function_name"] == "foo"
    mock_store.similarity_search.assert_called_once()


@patch("app.rag.retriever.get_vectorstore")
def test_retrieve_with_reranking(mock_get_vs):
    """Test that re-ranking reorders results by combined score."""
    mock_store = MagicMock()
    # Return two docs — one with a keyword match, one without
    mock_store.similarity_search.return_value = [
        Document(
            page_content="def calculate_tax(amount): return amount * 0.1",
            metadata={
                "chunk_id": "repo:tax.py:calculate_tax",
                "file_path": "tax.py",
                "function_name": "calculate_tax",
                "class_name": "",
                "chunk_type": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
            },
        ),
        Document(
            page_content="def helper(): return None",
            metadata={
                "chunk_id": "repo:util.py:helper",
                "file_path": "util.py",
                "function_name": "helper",
                "class_name": "",
                "chunk_type": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
            },
        ),
    ]
    mock_get_vs.return_value = mock_store

    results = retrieve("calculate_tax", "owner/repo", top_k=2, use_reranking=True, use_cache=False)
    assert len(results) >= 1
    # The function matching the keyword should rank first
    assert results[0].metadata["function_name"] == "calculate_tax"


def test_format_context_empty():
    result = format_context([])
    assert "No relevant" in result


def test_format_context_with_docs():
    docs = [
        Document(
            page_content="def bar(): return 42",
            metadata={
                "file_path": "src/utils.py",
                "function_name": "bar",
                "class_name": "",
                "start_line": 10,
                "end_line": 11,
            },
        )
    ]
    result = format_context(docs)
    assert "src/utils.py" in result
    assert "bar" in result
    assert "def bar" in result


# ── Keyword scoring unit tests ──

def test_tokenize_splits_identifiers():
    tokens = _tokenize("def calculateTax(amount)")
    assert "def" in tokens
    assert "calculatetax" in tokens
    assert "amount" in tokens


def test_keyword_score_full_match():
    score = _keyword_score(["def", "foo"], "def foo(): pass")
    assert score > 0.0


def test_keyword_score_no_match():
    score = _keyword_score(["xyz", "abc"], "def foo(): pass")
    assert score == 0.0


def test_keyword_score_empty_query():
    score = _keyword_score([], "def foo(): pass")
    assert score == 0.0
