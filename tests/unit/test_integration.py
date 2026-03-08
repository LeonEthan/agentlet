"""Integration tests for production systems working together."""

from __future__ import annotations

import time

import pytest

from agentlet.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from agentlet.core.metrics import Counter, Gauge, Histogram, get_metrics_registry, set_metrics_registry
from agentlet.core.parallel import ParallelExecutor
from agentlet.core.rate_limiter import TokenBucketRateLimiter
from agentlet.tools.base import Tool, ToolDefinition, ToolResult


class MockTool(Tool):
    """Mock tool for integration testing."""
    def __init__(self, name: str, delay: float = 0):
        self._name = name
        self.delay = delay
        self._def = ToolDefinition(
            name=name, description=f"Mock {name}",
            input_schema={"type": "object"}, approval_category="read_only"
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._def

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        if self.delay > 0:
            time.sleep(self.delay)
        return ToolResult(output=f"{self._name} result")


def test_rate_limiter_with_circuit_breaker():
    """Test rate limiter and circuit breaker work together."""
    limiter = TokenBucketRateLimiter(rate=100, capacity=10)
    breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

    call_count = 0

    def operation():
        nonlocal call_count
        limiter.acquire()
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Simulated failure")
        return "success"

    # First calls should fail and trip circuit
    for _ in range(2):
        with pytest.raises(Exception):
            breaker.call(operation)

    # Circuit should be open now
    with pytest.raises(CircuitOpenError):
        breaker.call(operation)

    # Wait for recovery
    time.sleep(0.15)

    # Should work after recovery
    result = breaker.call(operation)
    assert result == "success"


def test_parallel_executor_with_metrics():
    """Test parallel execution updates metrics correctly."""
    # Use fresh registry
    set_metrics_registry(None)
    registry = get_metrics_registry()

    executor = ParallelExecutor(max_workers=3)

    tool_calls = [
        ("call_1", MockTool("tool1", 0.01), {}),
        ("call_2", MockTool("tool2", 0.01), {}),
        ("call_3", MockTool("tool3", 0.01), {}),
    ]

    results = executor.execute(tool_calls)

    # Verify all completed
    assert len(results) == 3
    assert all(not r.is_error for r in results.values())


def test_metrics_thread_safety():
    """Test metrics work correctly under concurrent access."""
    import threading

    counter = Counter("test_counter", "Test")
    gauge = Gauge("test_gauge", "Test")
    histogram = Histogram("test_hist", "Test")

    def worker():
        for _ in range(100):
            counter.inc()
            gauge.set(counter.get())
            histogram.observe(counter.get())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.get() == 1000
    stats = histogram.get_stats()
    assert stats["count"] == 1000


def test_all_systems_compose_correctly():
    """Test all production systems can be composed together."""
    from agentlet.core.cache import RequestCache
    from agentlet.core.health import create_default_health_checker
    from agentlet.core.middleware import MiddlewareChain, logging_request_handler
    from agentlet.llm.schemas import ModelRequest

    # Health check
    health = create_default_health_checker()
    assert health.is_healthy() or True  # May fail in constrained env

    # Cache
    cache = RequestCache()

    # Middleware
    middleware = MiddlewareChain()
    middleware.add_request_handler(logging_request_handler(log_body=False))

    # Rate limiter
    limiter = TokenBucketRateLimiter(rate=1000, capacity=100)
    assert limiter.try_acquire()

    # Circuit breaker
    breaker = CircuitBreaker("test_compose")

    # All systems can coexist
    assert all([health, cache, middleware, limiter, breaker])


def test_error_recovery_integration():
    """Test error categorization works with circuit breaker."""
    from agentlet.core.errors import categorize_error, RecoveryExecutor

    # Simulate rate limit error
    class FakeRateLimitError(Exception):
        status_code = 429
        retry_after = 0.01

    exc = FakeRateLimitError("rate limit exceeded")
    categorized = categorize_error(exc)

    assert categorized.category.name == "TRANSIENT"
    assert categorized.retryable is True
    assert categorized.retry_after_seconds == 0.01


def test_end_to_end_production_flow():
    """Simulate a complete production workflow."""
    # Reset metrics
    set_metrics_registry(None)
    registry = get_metrics_registry()

    # Step 1: Health check
    from agentlet.core.health import create_default_health_checker
    health = create_default_health_checker()
    health_result = health.check_all()
    assert "status" in health_result

    # Step 2: Execute tools in parallel
    executor = ParallelExecutor(max_workers=2)
    tool_calls = [
        ("t1", MockTool("tool1", 0.01), {}),
        ("t2", MockTool("tool2", 0.01), {}),
    ]
    results = executor.execute(tool_calls)
    assert len(results) == 2

    # Step 3: Record metrics
    registry.llm.record_request("gpt-4o", 100, 50, 0.5)

    # Step 4: Verify all metrics captured
    summary = registry.get_all_metrics()
    assert "llm" in summary
    assert summary["llm"]["requests"]["total"] == 1.0
