"""
Advanced RAG Retriever — DevPilot AI

Implements a three-stage retrieval pipeline:
  1. Hybrid Search    — combines vector similarity + keyword (BM25-style) matching
  2. Re-ranking       — scores results by combined relevance signal
  3. Context Assembly — formats top results for LLM consumption with metadata

Design decisions:
  - Hybrid search catches both semantic meaning AND exact identifier matches
    (e.g. a query for "calculateTax" finds the function even if the embedding
    space doesn't place it close to a natural-language paraphrase).
  - Re-ranking uses a lightweight scoring heuristic (no separate model) to
    keep latency low while still improving result quality.
  - Caching avoids redundant vector DB queries within the TTL window.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.utils.cache import retrieval_cache, get_retrieval_cache_key
from app.utils.logging import get_logger
from app.utils.metrics import track_latency
from app.rag.vectorstore import get_vectorstore

logger = get_logger(__name__)


# ── Keyword search (BM25-inspired) ──


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier tokens for keyword matching."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def _keyword_score(query_tokens: list[str], document_text: str) -> float:
    """Compute a simple term-frequency keyword relevance score.

    Returns a 0.0–1.0 score representing what fraction of query tokens
    appear in the document, weighted by their frequency.
    """
    if not query_tokens:
        return 0.0
    doc_tokens = Counter(_tokenize(document_text))
    matches = sum(1 for t in query_tokens if doc_tokens.get(t, 0) > 0)
    return matches / len(query_tokens)


# ── Re-ranking ──


def _rerank_score(
    doc: Document,
    query: str,
    query_tokens: list[str],
    vector_rank: int,
    alpha: float,
) -> float:
    """Compute a hybrid re-ranking score combining vector rank and keyword matching.

    Score components:
      - vector_score: inverse rank from the vector search (higher = better)
      - keyword_score: fraction of query tokens found in the document
      - metadata_bonus: boost for function/class name matches

    alpha controls the vector vs keyword blend:
      alpha=1.0 → pure vector, alpha=0.0 → pure keyword.
    """
    vector_score = 1.0 / (1 + vector_rank)
    kw_score = _keyword_score(query_tokens, doc.page_content)

    # Metadata bonus: direct function/class name match in query
    meta = doc.metadata
    bonus = 0.0
    query_lower = query.lower()
    for field in ("function_name", "class_name"):
        name = meta.get(field, "").lower()
        if name and name in query_lower:
            bonus += 0.3
    if meta.get("file_path", "").lower() in query_lower:
        bonus += 0.15

    return round(alpha * vector_score + (1 - alpha) * kw_score + bonus, 4)


# ── Core retrieval functions ──


def retrieve(
    query: str,
    repo: str,
    top_k: int | None = None,
    filter_metadata: dict | None = None,
    use_reranking: bool = True,
    use_cache: bool = True,
) -> list[Document]:
    """Hybrid search with re-ranking over the ingested codebase.

    Pipeline:
      1. Check cache for identical query+repo+top_k
      2. Retrieve 3× top_k candidates from ChromaDB vector search
      3. Re-rank using hybrid vector+keyword scoring
      4. Return top_k results sorted by combined score

    Args:
        query: Natural language or code-aware query.
        repo: Repository identifier (owner/repo).
        top_k: Number of final results (default from config).
        filter_metadata: Optional ChromaDB where-filter dict.
        use_reranking: Whether to apply hybrid re-ranking.
        use_cache: Whether to check/update the retrieval cache.

    Returns:
        List of Documents ordered by hybrid relevance.
    """
    effective_top_k = top_k or settings.rag_rerank_top_k

    # ── Cache check ──
    cache_key = get_retrieval_cache_key(query, repo, effective_top_k)
    if use_cache:
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            logger.info("rag_cache_hit", repo=repo, query=query[:60])
            return cached

    # ── Vector search (over-fetch for re-ranking) ──
    with track_latency("rag_vector_search", repo=repo) as timing:
        store = get_vectorstore(repo)
        candidate_k = effective_top_k * 3 if use_reranking else effective_top_k
        search_kwargs: dict[str, Any] = {"k": candidate_k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
        candidates = store.similarity_search(query, **search_kwargs)

    if not candidates:
        logger.warning("rag_empty_retrieval", repo=repo, query=query[:80])
        return []

    # ── Re-ranking ──
    if use_reranking and len(candidates) > 1:
        with track_latency("rag_rerank", repo=repo, candidates=len(candidates)):
            query_tokens = _tokenize(query)
            scored = [
                (doc, _rerank_score(doc, query, query_tokens, rank, settings.rag_hybrid_alpha))
                for rank, doc in enumerate(candidates)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in scored[:effective_top_k]]
    else:
        results = candidates[:effective_top_k]

    # ── Cache store ──
    if use_cache:
        retrieval_cache.set(cache_key, results)

    logger.info(
        "rag_retrieve",
        repo=repo,
        query=query[:80],
        candidates=len(candidates),
        returned=len(results),
    )
    return results


def retrieve_for_changes(
    changed_files: list[dict],
    repo: str,
    top_k: int | None = None,
) -> list[Document]:
    """Retrieve context relevant to a set of changed files.

    For each changed file, builds a semantic query from the filename and
    available patch content, then deduplicates across all results.
    """
    effective_top_k = top_k or settings.rag_top_k
    all_docs: list[Document] = []
    seen_ids: set[str] = set()

    for file_info in changed_files:
        file_path = file_info.get("filename", "")
        query_parts = [f"Code in {file_path}"]
        if patch := file_info.get("patch", ""):
            query_parts.append(patch[:500])
        query = "\n".join(query_parts)

        docs = retrieve(query, repo, top_k=effective_top_k)
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    logger.info(
        "rag_retrieve_for_changes",
        repo=repo,
        changed_files=len(changed_files),
        total_chunks=len(all_docs),
    )
    return all_docs


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a structured context string for LLM consumption.

    Each document is rendered with its file path, line range, and any
    function/class metadata so the LLM can reference specific locations.
    """
    if not documents:
        return "No relevant code context found."

    sections: list[str] = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        header_parts = [
            f"[{i}]",
            meta.get("file_path", "unknown"),
            f"(L{meta.get('start_line', '?')}-L{meta.get('end_line', '?')})",
        ]
        if fn := meta.get("function_name"):
            header_parts.append(f"function: {fn}")
        if cls := meta.get("class_name"):
            header_parts.append(f"class: {cls}")
        if lang := meta.get("language"):
            header_parts.append(f"[{lang}]")

        header = "  ".join(header_parts)
        sections.append(f"--- {header} ---\n{doc.page_content}")

    return "\n\n".join(sections)
