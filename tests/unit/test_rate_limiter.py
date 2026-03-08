"""Tests for rate limiting."""

from __future__ import annotations

import time

import pytest

from agentlet.core.rate_limiter import (
    AdaptiveRateLimiter,
    RateLimitedClient,
    TokenBucketRateLimiter,
)


def test_token_bucket_allows_burst():
    """Test token bucket allows burst up to capacity."""
    limiter = TokenBucketRateLimiter(rate=1.0, capacity=5.0)

    # Should allow 5 immediate acquisitions
    for _ in range(5):
        assert limiter.try_acquire() is True

    # 6th should fail
    assert limiter.try_acquire() is False


def test_token_bucket_refills_over_time():
    """Test tokens refill over time."""
    limiter = TokenBucketRateLimiter(rate=10.0, capacity=1.0)

    # Use the token
    limiter.acquire()
    assert limiter.try_acquire() is False

    # Wait for refill
    time.sleep(0.15)

    # Should have a token now
    assert limiter.try_acquire() is True


def test_token_bucket_blocks_when_empty():
    """Test acquire blocks when tokens exhausted."""
    limiter = TokenBucketRateLimiter(rate=100.0, capacity=1.0)

    # Use the token
    limiter.acquire()

    # Next acquire should be fast since rate is high
    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start

    # Should have waited about 0.01s (1/100)
    assert elapsed < 0.1


def test_adaptive_rate_limiter_starts_at_target():
    """Test adaptive limiter starts at target rate."""
    limiter = AdaptiveRateLimiter(target_rate=5.0)

    # Should allow burst at initial rate
    for _ in range(5):
        assert limiter.try_acquire() is True


def test_adaptive_rate_limiter_reduces_on_error():
    """Test adaptive limiter reduces rate on rate limit errors."""
    limiter = AdaptiveRateLimiter(target_rate=10.0, min_rate=0.1)

    # Simulate rate limit error
    limiter.on_rate_limit_error()

    # Rate should be reduced
    assert limiter._current_rate == 5.0  # 50% of 10

    # Another error
    limiter.on_rate_limit_error()
    assert limiter._current_rate == 2.5  # 50% of 5


def test_adaptive_rate_limiter_increases_on_success():
    """Test adaptive limiter increases rate on success."""
    limiter = AdaptiveRateLimiter(target_rate=10.0)

    # Reduce rate first
    limiter.on_rate_limit_error()
    assert limiter._current_rate == 5.0

    # Success should increase rate
    limiter.on_success()
    assert limiter._current_rate == 5.25  # 5% increase


def test_adaptive_rate_limiter_respects_min_rate():
    """Test adaptive limiter doesn't go below minimum."""
    limiter = AdaptiveRateLimiter(target_rate=1.0, min_rate=0.1)

    # Multiple errors
    for _ in range(10):
        limiter.on_rate_limit_error()

    # Should not go below min_rate
    assert limiter._current_rate >= 0.1


def test_adaptive_rate_limiter_respects_target_rate():
    """Test adaptive limiter doesn't exceed target."""
    limiter = AdaptiveRateLimiter(target_rate=1.0)

    # Start below target
    limiter._current_rate = 0.5
    limiter._limiter.rate = 0.5

    # Multiple successes
    for _ in range(100):
        limiter.on_success()

    # Should not exceed target
    assert limiter._current_rate <= 1.0


def test_rate_limited_client():
    """Test rate limited client wrapper."""

    class FakeClient:
        def __init__(self):
            self.call_count = 0

        def complete(self, request):
            self.call_count += 1
            return f"response-{self.call_count}"

    fake = FakeClient()
    limiter = TokenBucketRateLimiter(rate=1000.0, capacity=10.0)
    client = RateLimitedClient(fake, limiter)

    result = client.complete("request")

    assert result == "response-1"
    assert fake.call_count == 1


def test_rate_limiter_acquire_timing():
    """Test that acquire properly waits."""
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=1.0)

    # First acquire should be immediate
    start = time.monotonic()
    limiter.acquire()
    first_elapsed = time.monotonic() - start
    assert first_elapsed < 0.01

    # Second acquire should wait ~0.5s
    limiter.acquire()
    second_elapsed = time.monotonic() - start
    assert second_elapsed >= 0.4  # Some tolerance
