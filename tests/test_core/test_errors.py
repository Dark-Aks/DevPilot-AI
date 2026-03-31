from __future__ import annotations

import asyncio
import time

import pytest

from app.utils.errors import (
    DevPilotError,
    RetrievalError,
    AgentError,
    CircuitBreaker,
    agent_fallback,
    with_timeout,
    LLMError,
)


# ── Exception hierarchy ──


def test_custom_exceptions_inherit_from_base():
    assert issubclass(RetrievalError, DevPilotError)
    assert issubclass(AgentError, DevPilotError)


def test_devpilot_error_stores_details():
    err = DevPilotError("fail", details={"key": "val"})
    assert str(err) == "fail"
    assert err.details == {"key": "val"}


# ── agent_fallback decorator ──


@pytest.mark.asyncio
async def test_agent_fallback_returns_result_on_success():
    @agent_fallback("test_agent", "output", "default")
    async def good_agent(state, llm):
        return {"output": "real_result"}

    result = await good_agent({}, None)
    assert result["output"] == "real_result"


@pytest.mark.asyncio
async def test_agent_fallback_returns_default_on_error():
    @agent_fallback("test_agent", "output", "default_value")
    async def bad_agent(state, llm):
        raise RuntimeError("boom")

    result = await bad_agent({}, None)
    assert result["output"] == "default_value"
    assert any("test_agent" in e for e in result["errors"])


# ── CircuitBreaker ──


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="test")
    assert cb.state == "CLOSED"


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60, name="test")
    cb.record_failure()
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "OPEN"


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60, name="test")
    cb.record_failure()
    cb.record_success()
    assert cb.state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60, name="test")
    cb.record_failure()
    assert cb.state == "OPEN"

    async def dummy():
        return "ok"

    with pytest.raises(DevPilotError, match="OPEN"):
        await cb.call(dummy)


@pytest.mark.asyncio
async def test_circuit_breaker_allows_half_open_probe():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, name="test")
    cb.record_failure()
    assert cb.state == "OPEN"
    time.sleep(0.06)
    assert cb.state == "HALF_OPEN"

    async def ok_func():
        return "recovered"

    result = await cb.call(ok_func)
    assert result == "recovered"
    assert cb.state == "CLOSED"


# ── with_timeout ──


@pytest.mark.asyncio
async def test_with_timeout_returns_on_success():
    async def fast():
        return 42

    result = await with_timeout(fast(), timeout_seconds=1.0, operation="test")
    assert result == 42


@pytest.mark.asyncio
async def test_with_timeout_raises_on_expiry():
    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(LLMError, match="timed out"):
        await with_timeout(slow(), timeout_seconds=0.05, operation="slow_op")
