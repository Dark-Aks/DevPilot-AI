from __future__ import annotations

import time

from app.core.cache import LRUCache, cache_stats


def test_lru_cache_set_and_get():
    cache = LRUCache(max_size=5, ttl_seconds=60)
    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"


def test_lru_cache_miss():
    cache = LRUCache(max_size=5, ttl_seconds=60)
    assert cache.get("missing") is None


def test_lru_cache_eviction():
    cache = LRUCache(max_size=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_ttl_expiry():
    cache = LRUCache(max_size=5, ttl_seconds=0.1)
    cache.set("x", "val")
    time.sleep(0.15)
    assert cache.get("x") is None


def test_lru_cache_hit_miss_stats():
    cache = LRUCache(max_size=5, ttl_seconds=60)
    cache.set("k", "v")
    cache.get("k")      # hit
    cache.get("other")   # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_cache_stats_returns_all_caches():
    stats = cache_stats()
    assert "retrieval" in stats
    assert "embedding" in stats
    assert "llm" in stats
