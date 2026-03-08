"""Tests for metrics collection system."""

from __future__ import annotations

import pytest

from agentlet.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    LLMUsageMetrics,
    MetricsRegistry,
    ToolMetrics,
    get_metrics_registry,
    set_metrics_registry,
)


def test_counter_increment():
    """Test counter increments correctly."""
    counter = Counter("test_counter", "Test counter")

    counter.inc()
    assert counter.get() == 1.0

    counter.inc(5.0)
    assert counter.get() == 6.0


def test_counter_reset():
    """Test counter reset returns previous value."""
    counter = Counter("test_counter", "Test counter")
    counter.inc(10.0)

    prev = counter.reset()

    assert prev == 10.0
    assert counter.get() == 0.0


def test_gauge_set():
    """Test gauge set and get."""
    gauge = Gauge("test_gauge", "Test gauge")

    gauge.set(42.0)
    assert gauge.get() == 42.0

    gauge.set(100.0)
    assert gauge.get() == 100.0


def test_gauge_increment_decrement():
    """Test gauge inc/dec."""
    gauge = Gauge("test_gauge", "Test gauge")

    gauge.set(10.0)
    gauge.inc(5.0)
    assert gauge.get() == 15.0

    gauge.dec(3.0)
    assert gauge.get() == 12.0


def test_histogram_observe():
    """Test histogram records observations."""
    hist = Histogram("test_hist", "Test histogram")

    hist.observe(0.01)
    hist.observe(0.05)
    hist.observe(0.1)

    stats = hist.get_stats()
    assert stats["count"] == 3
    assert stats["min"] == 0.01
    assert stats["max"] == 0.1


def test_histogram_bucket_counts():
    """Test histogram bucket counts."""
    hist = Histogram("test_hist", "Test histogram", buckets=[0.01, 0.1, 1.0])

    hist.observe(0.005)  # le_0.01
    hist.observe(0.05)   # le_0.1
    hist.observe(0.5)    # le_1.0

    buckets = hist.get_bucket_counts()
    assert buckets["le_0.01"] == 1
    assert buckets["le_0.1"] == 2
    assert buckets["le_1.0"] == 3
    assert buckets["le_inf"] == 3


def test_histogram_percentiles():
    """Test histogram percentile calculations."""
    hist = Histogram("test_hist", "Test histogram")

    # Add 100 values
    for i in range(100):
        hist.observe(float(i))

    stats = hist.get_stats()
    assert stats["count"] == 100
    assert stats["p50"] == 50.0
    assert 93 <= stats["p95"] <= 96
    assert 98 <= stats["p99"] <= 99


def test_llm_metrics_record_request():
    """Test LLM metrics recording."""
    metrics = LLMUsageMetrics()

    result = metrics.record_request(
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        latency_seconds=1.5,
    )

    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 500
    assert result["total_tokens"] == 1500
    assert result["latency_seconds"] == 1.5
    assert result["cost_usd"] > 0

    assert metrics.requests_total.get() == 1.0
    assert metrics.tokens_input_total.get() == 1000.0
    assert metrics.tokens_output_total.get() == 500.0


def test_llm_metrics_cost_calculation_openai():
    """Test cost calculation for OpenAI models."""
    metrics = LLMUsageMetrics()

    gpt4_result = metrics.record_request("gpt-4o", 1000000, 500000, 1.0)
    assert gpt4_result["cost_usd"] > 0

    # GPT-4o-mini should be cheaper
    mini_result = metrics.record_request("gpt-4o-mini", 1000000, 500000, 1.0)
    assert mini_result["cost_usd"] < gpt4_result["cost_usd"]


def test_llm_metrics_cost_calculation_unknown_model():
    """Test cost calculation for unknown model returns 0."""
    metrics = LLMUsageMetrics()

    result = metrics.record_request("unknown-model", 1000, 500, 1.0)
    assert result["cost_usd"] == 0.0


def test_llm_metrics_summary():
    """Test LLM metrics summary."""
    metrics = LLMUsageMetrics()

    metrics.record_request("gpt-4o", 1000, 500, 1.0)
    metrics.record_request("gpt-4o", 2000, 1000, 2.0)

    summary = metrics.get_summary()

    assert summary["requests"]["total"] == 2.0
    assert summary["tokens"]["input"] == 3000.0
    assert summary["tokens"]["output"] == 1500.0
    assert summary["cost_usd"] > 0
    assert summary["latency"]["count"] == 2


def test_tool_metrics_record():
    """Test tool metrics recording."""
    metrics = ToolMetrics()

    metrics.record_execution("Read", 0.1, error=False)
    metrics.record_execution("Write", 0.2, error=True)
    metrics.record_execution("Read", 0.15, error=False)

    summary = metrics.get_summary()

    assert summary["executions_total"] == 3
    assert summary["errors_total"] == 1
    assert summary["error_rate"] == 1.0 / 3.0


def test_metrics_registry():
    """Test metrics registry aggregates all metrics."""
    registry = MetricsRegistry()

    registry.llm.record_request("gpt-4o", 1000, 500, 1.0)
    registry.tools.record_execution("Read", 0.1)
    registry.sessions_started.inc()
    registry.approvals_requested.inc()
    registry.approvals_granted.inc()

    all_metrics = registry.get_all_metrics()

    assert all_metrics["llm"]["requests"]["total"] == 1.0
    assert all_metrics["tools"]["executions_total"] == 1.0
    assert all_metrics["sessions"]["started"] == 1.0
    assert all_metrics["approvals"]["requested"] == 1.0
    assert all_metrics["approvals"]["granted"] == 1.0
    assert "timestamp" in all_metrics


def test_metrics_registry_prometheus_export():
    """Test Prometheus export format."""
    registry = MetricsRegistry()

    registry.llm.record_request("gpt-4o", 1000, 500, 1.0)
    registry.llm.requests_total.inc()

    output = registry.export_prometheus()

    assert "# HELP llm_requests_total" in output
    assert "# TYPE llm_requests_total counter" in output
    assert "llm_requests_total 2.0" in output
    assert "# HELP llm_latency_seconds" in output
    assert "# TYPE llm_latency_seconds histogram" in output


def test_global_registry_singleton():
    """Test global registry is a singleton."""
    reg1 = get_metrics_registry()
    reg2 = get_metrics_registry()

    assert reg1 is reg2


def test_global_registry_override():
    """Test global registry can be overridden."""
    original = get_metrics_registry()
    new_registry = MetricsRegistry()

    set_metrics_registry(new_registry)

    assert get_metrics_registry() is new_registry

    # Restore
    set_metrics_registry(original)
    assert get_metrics_registry() is original


def test_histogram_empty_stats():
    """Test histogram stats when empty."""
    hist = Histogram("test_hist", "Test histogram")

    stats = hist.get_stats()

    assert stats["count"] == 0
    assert stats["sum"] == 0.0
    assert stats["avg"] == 0.0


def test_metrics_thread_safety():
    """Test basic thread safety of counter."""
    import threading

    counter = Counter("test_counter", "Test counter")

    def increment():
        for _ in range(1000):
            counter.inc()

    threads = [threading.Thread(target=increment) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.get() == 10000.0


def test_llm_metrics_active_requests():
    """Test active requests gauge."""
    metrics = LLMUsageMetrics()

    metrics.active_requests.set(5)
    assert metrics.active_requests.get() == 5.0

    metrics.active_requests.inc()
    assert metrics.active_requests.get() == 6.0

    metrics.active_requests.dec()
    assert metrics.active_requests.get() == 5.0
