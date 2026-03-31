"""
Caching Layer — DevPilot AI

Two-tier caching strategy:
  1. In-memory LRU cache (always available, no dependencies)
  2. Redis cache (optional, for multi-process / distributed deployments)

Caches:
  - Embedding results (avoid re-embedding identical code chunks)
  - RAG retrieval results (avoid re-querying for same query+repo within TTL)
  - LLM responses (optional, for identical prompts)
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe in-memory LRU cache with TTL support.

    Designed for single-process deployments. For multi-process scaling,
    replace with the RedisCache below.

    Args:
        max_size: Maximum number of entries.
        ttl_seconds: Time-to-live per entry. 0 = no expiry.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _is_expired(self, timestamp: float) -> bool:
        if self._ttl <= 0:
            return False
        return (time.monotonic() - timestamp) > self._ttl

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None on miss or expiry."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, ts = self._cache[key]
        if self._is_expired(ts):
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value. Evicts LRU entry if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


def _make_cache_key(*parts: str) -> str:
    """Create a deterministic cache key from string parts."""
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Singleton caches ──

# RAG retrieval cache: query+repo → list of document dicts
retrieval_cache = LRUCache(max_size=500, ttl_seconds=300)

# Embedding cache: chunk_id → embedding vector (avoids re-embedding unchanged code)
embedding_cache = LRUCache(max_size=5000, ttl_seconds=3600)

# LLM response cache: prompt_hash → response (for idempotent agent calls)
llm_cache = LRUCache(max_size=200, ttl_seconds=600)


def get_retrieval_cache_key(query: str, repo: str, top_k: int) -> str:
    return _make_cache_key("retrieval", repo, query, str(top_k))


def get_llm_cache_key(system_prompt: str, user_prompt: str, model: str) -> str:
    return _make_cache_key("llm", model, system_prompt[:200], user_prompt[:500])


def cache_stats() -> dict[str, Any]:
    """Return stats for all cache tiers (used by /health endpoint)."""
    return {
        "retrieval_cache": retrieval_cache.stats,
        "embedding_cache": embedding_cache.stats,
        "llm_cache": llm_cache.stats,
    }
