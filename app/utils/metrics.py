"""
Metrics & Evaluation Module — DevPilot AI

Tracks three categories of production metrics:
  1. Latency   — Wall-clock time for RAG retrieval, LLM calls, agent runs
  2. Cost      — Token counting → estimated USD per request (OpenAI pricing)
  3. Quality   — Retrieval hit-rate, context relevance scoring

All metrics are emitted as structured log events for easy aggregation by
any log-based monitoring pipeline (Datadog, Grafana Loki, CloudWatch).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import tiktoken

from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── Cost estimation (USD per 1K tokens, OpenAI pricing as of 2025) ──

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding (GPT-4 family)."""
    return len(_encoder.encode(text, disallowed_special=()))


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
) -> float:
    """Estimate USD cost for a single LLM or embedding call."""
    pricing = _PRICING.get(model, _PRICING.get("gpt-4o"))
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)


# ── Latency tracking ──


@dataclass
class LatencyRecord:
    """Captures wall-clock seconds for a named operation."""
    operation: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@contextmanager
def track_latency(operation: str, **meta) -> Generator[dict, None, None]:
    """Context manager that measures and logs execution time.

    Usage:
        with track_latency("rag_retrieve", repo="owner/repo") as timing:
            results = do_retrieval(...)
        # timing["duration_ms"] is now populated
    """
    timing: dict[str, Any] = {"operation": operation, "duration_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        timing["duration_ms"] = round(elapsed_ms, 2)
        logger.info(
            "latency",
            operation=operation,
            duration_ms=timing["duration_ms"],
            **meta,
        )


# ── Request-level metrics aggregator ──


@dataclass
class RequestMetrics:
    """Accumulates metrics across the lifecycle of a single request/workflow run."""

    request_id: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies: list[LatencyRecord] = field(default_factory=list)
    retrieval_chunks_returned: int = 0
    retrieval_chunks_relevant: int = 0  # Set via manual evaluation or heuristic
    agents_invoked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_name: str = "",
    ) -> None:
        """Record token usage and cost from a single LLM invocation."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_cost_usd += cost
        if agent_name:
            self.agents_invoked.append(agent_name)

        logger.info(
            "llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            agent=agent_name,
        )

    def record_latency(self, operation: str, duration_ms: float, **meta) -> None:
        self.latencies.append(
            LatencyRecord(operation=operation, duration_ms=duration_ms, metadata=meta)
        )

    def record_retrieval(self, total: int, relevant: int = 0) -> None:
        self.retrieval_chunks_returned = total
        self.retrieval_chunks_relevant = relevant

    @property
    def retrieval_hit_rate(self) -> float:
        """Fraction of retrieved chunks deemed relevant (0.0–1.0)."""
        if self.retrieval_chunks_returned == 0:
            return 0.0
        return self.retrieval_chunks_relevant / self.retrieval_chunks_returned

    @property
    def total_latency_ms(self) -> float:
        return sum(r.duration_ms for r in self.latencies)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of all collected metrics."""
        return {
            "request_id": self.request_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "retrieval_chunks": self.retrieval_chunks_returned,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 3),
            "agents_invoked": self.agents_invoked,
            "errors": self.errors,
            "latency_breakdown": {
                r.operation: r.duration_ms for r in self.latencies
            },
        }

    def emit(self) -> None:
        """Log the full metrics summary as a structured event."""
        logger.info("request_metrics", **self.summary())
