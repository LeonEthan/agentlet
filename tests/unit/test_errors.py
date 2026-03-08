"""Tests for error categorization and recovery system."""

from __future__ import annotations

import time

import pytest

from agentlet.core.errors import (
    CategorizedError,
    CircuitOpenError,
    ErrorAggregator,
    ErrorCategory,
    RecoveryAction,
    RecoveryExecutor,
    categorize_error,
)


class FakeRateLimitError(Exception):
    """Simulated rate limit error."""

    def __init__(self, message="Rate limit exceeded"):
        super().__init__(message)
        self.status_code = 429
        self.retry_after = 2.0


class FakeAuthError(Exception):
    """Simulated authentication error."""

    def __init__(self, message="Invalid API key"):
        super().__init__(message)
        self.status_code = 401


class FakeTimeoutError(Exception):
    """Simulated timeout error."""

    def __init__(self, message="Request timed out"):
        super().__init__(message)


class FakeValidationError(Exception):
    """Simulated validation error."""

    def __init__(self, message="Invalid request"):
        super().__init__(message)
        self.status_code = 400


def test_categorize_rate_limit_error():
    """Test rate limit errors are categorized as transient."""
    exc = FakeRateLimitError()
    categorized = categorize_error(exc)

    assert categorized.category == ErrorCategory.TRANSIENT
    assert categorized.action == RecoveryAction.RETRY_ADAPTIVE
    assert categorized.retryable is True
    assert categorized.retry_after_seconds == 2.0


def test_categorize_auth_error():
    """Test auth errors are not retryable."""
    exc = FakeAuthError()
    categorized = categorize_error(exc)

    assert categorized.category == ErrorCategory.AUTH
    assert categorized.action == RecoveryAction.FAIL_USER
    assert categorized.retryable is False


def test_categorize_timeout_error():
    """Test timeout errors are retryable with backoff."""
    exc = TimeoutError("Connection timed out")
    categorized = categorize_error(exc)

    assert categorized.category == ErrorCategory.TIMEOUT
    assert categorized.action == RecoveryAction.RETRY_BACKOFF
    assert categorized.retryable is True


def test_categorize_validation_error():
    """Test validation errors fail fast."""
    exc = FakeValidationError()
    categorized = categorize_error(exc)

    assert categorized.category == ErrorCategory.VALIDATION
    assert categorized.action == RecoveryAction.FAIL_FAST
    assert categorized.retryable is False


def test_categorize_by_message_patterns():
    """Test categorization by message substring matching."""
    test_cases = [
        ("Rate limit exceeded", ErrorCategory.TRANSIENT),
        ("Service unavailable", ErrorCategory.TRANSIENT),
        ("Authentication failed", ErrorCategory.AUTH),
        ("Invalid token", ErrorCategory.AUTH),
        ("Request timeout", ErrorCategory.TIMEOUT),
        ("Disk full", ErrorCategory.RESOURCE),
        ("Invalid parameter", ErrorCategory.VALIDATION),
    ]

    for message, expected_category in test_cases:
        exc = Exception(message)
        categorized = categorize_error(exc)
        assert categorized.category == expected_category, f"Failed for: {message}"


def test_categorize_preserves_context():
    """Test context is preserved in categorized error."""
    exc = Exception("test")
    context = {"operation": "test_op", "attempt": 3}

    categorized = categorize_error(exc, context=context)

    assert categorized.context["operation"] == "test_op"
    assert categorized.context["attempt"] == 3


def test_categorize_unknown_error_is_retryable():
    """Test unknown errors are conservatively marked retryable."""
    exc = Exception("Something weird happened")
    categorized = categorize_error(exc)

    assert categorized.category == ErrorCategory.UNKNOWN
    assert categorized.retryable is True


def test_recovery_executor_succeeds_on_first_try():
    """Test executor returns immediately on success."""
    executor = RecoveryExecutor(max_retries=2)

    call_count = 0

    def operation():
        nonlocal call_count
        call_count += 1
        return "success"

    result = executor.execute(operation, "test_op")

    assert result == "success"
    assert call_count == 1


def test_recovery_executor_retries_on_transient_error():
    """Test executor retries transient errors."""
    executor = RecoveryExecutor(max_retries=2, base_delay=0.01)

    call_count = 0

    def operation():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise FakeRateLimitError()
        return "success"

    result = executor.execute(operation, "test_op")

    assert result == "success"
    assert call_count == 2


def test_recovery_executor_gives_up_after_max_retries():
    """Test executor raises after exhausting retries."""
    executor = RecoveryExecutor(max_retries=1, base_delay=0.01)

    def operation():
        raise FakeRateLimitError()

    with pytest.raises(FakeRateLimitError):
        executor.execute(operation, "test_op")


def test_recovery_executor_fails_fast_on_auth_error():
    """Test executor doesn't retry auth errors."""
    executor = RecoveryExecutor(max_retries=3)

    call_count = 0

    def operation():
        nonlocal call_count
        call_count += 1
        raise FakeAuthError()

    with pytest.raises(FakeAuthError):
        executor.execute(operation, "test_op")

    assert call_count == 1  # No retries


def test_categorized_error_is_frozen():
    """Test categorized errors are frozen (field reassignment prevented)."""
    exc = Exception("test")
    original_context = {"key": "value"}
    categorized = categorize_error(exc, context=original_context)

    # Should be able to read context
    assert categorized.context["key"] == "value"

    # Frozen dataclass prevents field reassignment
    with pytest.raises(AttributeError):
        categorized.context = {}

    # Original dict can be modified but doesn't affect internal copy
    original_context["new"] = "added"
    assert "new" not in categorized.context


def test_error_aggregator_tracks_errors():
    """Test aggregator tracks error frequencies."""
    aggregator = ErrorAggregator(window_size=10)

    # Record some errors
    for _ in range(3):
        aggregator.record(categorize_error(FakeRateLimitError()))
    for _ in range(2):
        aggregator.record(categorize_error(FakeAuthError()))

    stats = aggregator.get_stats(window_seconds=60)

    assert stats["total"] == 5
    assert stats["by_category"]["TRANSIENT"] == 3
    assert stats["by_category"]["AUTH"] == 2


def test_error_aggregator_respects_window():
    """Test aggregator respects time window."""
    aggregator = ErrorAggregator(window_size=10)

    # Record errors
    aggregator.record(categorize_error(FakeRateLimitError()))

    # Get stats with very small window (should exclude recent)
    stats = aggregator.get_stats(window_seconds=0.001)

    # Wait a tiny bit then check
    time.sleep(0.01)
    stats = aggregator.get_stats(window_seconds=0.001)

    assert stats["total"] == 0  # Old error excluded


def test_error_aggregator_health_check():
    """Test health check based on transient rate."""
    aggregator = ErrorAggregator(window_size=100)

    # All transient errors - should be unhealthy at low threshold
    for _ in range(10):
        aggregator.record(categorize_error(FakeRateLimitError()))

    assert aggregator.is_healthy(threshold=0.05) is False

    # Mix errors - mostly non-transient should be healthy
    aggregator2 = ErrorAggregator(window_size=100)
    aggregator2.record(categorize_error(FakeAuthError()))  # Non-transient
    aggregator2.record(categorize_error(FakeAuthError()))  # Non-transient
    aggregator2.record(categorize_error(FakeRateLimitError()))  # Transient

    # 1/3 transient rate - should be healthy at 0.5 threshold
    assert aggregator2.is_healthy(threshold=0.5) is True


def test_categorized_error_includes_user_message():
    """Test user-friendly messages are provided."""
    exc = FakeAuthError()
    categorized = categorize_error(exc)

    assert "credentials" in categorized.user_message.lower()
    assert len(categorized.user_message) > 10  # Substantial message


def test_recovery_executor_uses_retry_after_header():
    """Test executor respects server-provided retry-after."""
    executor = RecoveryExecutor(max_retries=1, base_delay=10.0)

    start_time = time.monotonic()

    def operation():
        raise FakeRateLimitError()  # Has retry_after=2.0

    with pytest.raises(FakeRateLimitError):
        executor.execute(operation, "test_op")

    elapsed = time.monotonic() - start_time

    # Should use retry_after=2.0, not base_delay=10.0
    assert elapsed < 5.0  # Much less than 10s base delay


def test_error_categories_are_distinct():
    """Test all error categories have unique auto values."""
    categories = list(ErrorCategory)
    values = [c.value for c in categories]

    assert len(values) == len(set(values))  # All unique


def test_recovery_actions_are_distinct():
    """Test all recovery actions have unique auto values."""
    actions = list(RecoveryAction)
    values = [a.value for a in actions]

    assert len(values) == len(set(values))  # All unique
