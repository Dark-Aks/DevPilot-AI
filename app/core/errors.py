"""
Error Handling — DevPilot AI

Defines custom exception hierarchy and resilience utilities:
  - Domain exceptions (retrieval, agent, GitHub, config)
  - Graceful fallback decorator for agent functions
  - Circuit breaker pattern for external API calls
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ── Custom Exceptions ──


class DevPilotError(Exception):
    """Base exception for all DevPilot errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class RetrievalError(DevPilotError):
    """Raised when RAG retrieval fails or returns empty results."""
    pass


class AgentError(DevPilotError):
    """Raised when an agent fails to produce valid output."""
    pass


class GitHubAPIError(DevPilotError):
    """Raised when a GitHub API call fails after retries."""
    pass


class LLMError(DevPilotError):
    """Raised when the LLM provider returns an error or times out."""
    pass


class ConfigurationError(DevPilotError):
    """Raised when required configuration is missing or invalid."""
    pass


# ── Resilience Utilities ──


def agent_fallback(agent_name: str, default_key: str, default_value: Any):
    """Decorator that wraps an agent function with error handling.

    If the agent raises ANY exception, logs the error and returns a
    fallback dict so the workflow can continue without crashing.

    Usage:
        @agent_fallback("review", "review_findings", [])
        async def run_review(state, llm):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "agent_fallback_triggered",
                    agent=agent_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return {
                    default_key: default_value,
                    "errors": [f"{agent_name}: {type(e).__name__}: {str(e)}"],
                }

        return wrapper

    return decorator


class CircuitBreaker:
    """Simple circuit breaker for external service calls.

    States:
      CLOSED  — normal operation, calls pass through
      OPEN    — too many failures, calls are rejected immediately
      HALF_OPEN — after cooldown, allow one probe call

    Args:
        failure_threshold: Failures before opening the circuit.
        recovery_timeout: Seconds to wait before half-open probe.
        name: Identifier for logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                name=self.name,
                failures=self._failure_count,
            )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        if self.state == "OPEN":
            raise DevPilotError(
                f"Circuit breaker '{self.name}' is OPEN — service unavailable",
                details={"breaker": self.name, "failures": self._failure_count},
            )

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


# ── Timeout wrapper ──


async def with_timeout(coro, timeout_seconds: float, operation: str = ""):
    """Wrap an awaitable with a timeout, raising LLMError on expiry."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        msg = f"Operation '{operation}' timed out after {timeout_seconds}s"
        logger.error("timeout", operation=operation, timeout_s=timeout_seconds)
        raise LLMError(msg, details={"operation": operation, "timeout": timeout_seconds})
