from __future__ import annotations

from app.utils.metrics import count_tokens, estimate_cost, RequestMetrics, track_latency


def test_count_tokens_returns_positive():
    n = count_tokens("def hello(): pass")
    assert n > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_estimate_cost_gpt4o():
    cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    assert cost > 0


def test_estimate_cost_unknown_model_uses_fallback():
    cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
    assert cost > 0


def test_request_metrics_summary():
    m = RequestMetrics(request_id="test-123")
    m.record_llm_call("gpt-4o", input_tokens=100, output_tokens=50, agent_name="review")
    m.record_retrieval(total=10, relevant=7)

    summary = m.summary()
    assert summary["request_id"] == "test-123"
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50
    assert summary["total_cost_usd"] > 0
    assert summary["retrieval_chunks"] == 10
    assert summary["retrieval_hit_rate"] == 0.7
    assert "review" in summary["agents_invoked"]


def test_request_metrics_hit_rate_zero_division():
    m = RequestMetrics(request_id="empty")
    assert m.retrieval_hit_rate == 0.0


def test_track_latency_records_duration():
    with track_latency("test_op") as timing:
        _ = sum(range(1000))
    assert timing["duration_ms"] > 0
    assert timing["operation"] == "test_op"
