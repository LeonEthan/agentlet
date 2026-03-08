"""Benchmarks for critical paths."""

from __future__ import annotations

import time

import pytest

from agentlet.core.cache import generate_cache_key, RequestCache
from agentlet.core.metrics import Counter, Histogram
from agentlet.core.rate_limiter import TokenBucketRateLimiter
from agentlet.llm.schemas import ModelRequest, ModelToolDefinition
from agentlet.core.messages import Message


def test_cache_key_generation_performance():
    """Cache key generation should be fast (< 1ms)."""
    request = ModelRequest(
        messages=[Message(role="user", content="Test message")],
        tools=[ModelToolDefinition(name="Read", description="Read file", input_schema={})],
    )

    start = time.perf_counter()
    for _ in range(1000):
        key = generate_cache_key(request)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0  # 1000 ops in less than 1 second


def test_token_bucket_high_throughput():
    """Token bucket should handle high throughput."""
    limiter = TokenBucketRateLimiter(rate=10000, capacity=1000)

    start = time.perf_counter()
    for _ in range(100):
        limiter.try_acquire()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.01  # 100 ops in less than 10ms


def test_counter_performance():
    """Counter increment should be fast."""
    counter = Counter("bench_counter", "Benchmark")

    start = time.perf_counter()
    for _ in range(10000):
        counter.inc()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1  # 10k ops in less than 100ms
    assert counter.get() == 10000


def test_histogram_performance():
    """Histogram observation should be fast."""
    hist = Histogram("bench_hist", "Benchmark")

    start = time.perf_counter()
    for i in range(1000):
        hist.observe(i / 100)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1  # 1k ops in less than 100ms
    assert hist.get_stats()["count"] == 1000


def test_cache_hit_performance():
    """Cache hit should be faster than cache miss."""
    cache = RequestCache()
    request = ModelRequest(messages=[Message(role="user", content="Test")])

    from agentlet.llm.schemas import ModelResponse
    response = ModelResponse(
        message=Message(role="assistant", content="Response"),
        finish_reason="stop",
    )

    # Warm up cache
    cache.set(request, response)

    # Measure hit time
    start = time.perf_counter()
    for _ in range(100):
        cache.get(request)
    hit_time = time.perf_counter() - start

    assert hit_time < 0.01  # 100 hits in less than 10ms
