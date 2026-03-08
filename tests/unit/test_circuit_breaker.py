"""Tests for circuit breaker pattern."""

from __future__ import annotations

import time

import pytest

from agentlet.core.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    OperationError,
)


def test_circuit_breaker_starts_closed():
    """Test that circuit breaker starts in CLOSED state."""
    breaker = CircuitBreaker("test")

    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_successful_call():
    """Test successful operation passes through."""
    breaker = CircuitBreaker("test")

    result = breaker.call(lambda: "success")

    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_tracks_failures():
    """Test that failures are tracked."""
    breaker = CircuitBreaker("test", failure_threshold=3)

    # Fail twice - should stay closed
    for _ in range(2):
        with pytest.raises(OperationError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_opens_after_threshold():
    """Test circuit opens after failure threshold exceeded."""
    breaker = CircuitBreaker("test", failure_threshold=2)

    # Fail twice to exceed threshold
    for _ in range(2):
        with pytest.raises(OperationError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    assert breaker.state == CircuitState.OPEN

    # Next call should fail fast with CircuitOpenError
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not execute")


def test_circuit_breaker_half_open_after_timeout():
    """Test circuit enters half-open after recovery timeout."""
    breaker = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.01,
    )

    # Open the circuit
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout
    time.sleep(0.02)

    # Should now be half-open
    result = breaker.call(lambda: "success")
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_half_open_failure_reopens():
    """Test that failure in half-open reopens circuit."""
    breaker = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.01,
    )

    # Open the circuit
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    # Wait for timeout
    time.sleep(0.02)

    # Fail in half-open
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))

    assert breaker.state == CircuitState.OPEN


def test_circuit_breaker_half_open_limit():
    """Test half-open call limit."""
    breaker = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.01,
        half_open_max_calls=2,
    )

    # Open the circuit
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    time.sleep(0.02)

    # Use up half-open calls
    breaker._state = CircuitState.HALF_OPEN
    breaker._half_open_calls = 2

    # Next call should fail fast
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not execute")


def test_circuit_breaker_force_close():
    """Test manual reset of circuit."""
    breaker = CircuitBreaker("test", failure_threshold=1)

    # Open the circuit
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    assert breaker.state == CircuitState.OPEN

    # Force close
    breaker.force_close()

    assert breaker.state == CircuitState.CLOSED

    # Should work normally again
    result = breaker.call(lambda: "success")
    assert result == "success"


def test_circuit_breaker_get_stats():
    """Test statistics reporting."""
    breaker = CircuitBreaker("test", failure_threshold=5)

    stats = breaker.get_stats()

    assert stats["name"] == "test"
    assert stats["state"] == "CLOSED"
    assert stats["failure_count"] == 0
    assert stats["failure_threshold"] == 5


def test_circuit_breaker_retry_after():
    """Test retry after timing."""
    breaker = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=10.0,
    )

    # Open the circuit
    with pytest.raises(OperationError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    # Check retry after is positive
    with pytest.raises(CircuitOpenError) as exc_info:
        breaker.call(lambda: "fail")

    assert exc_info.value.retry_after is not None
    assert exc_info.value.retry_after > 0
    assert exc_info.value.circuit_name == "test"


def test_circuit_breaker_preserves_function_arguments():
    """Test that function arguments are passed through."""
    breaker = CircuitBreaker("test")

    def func(a, b, c=None):
        return (a, b, c)

    result = breaker.call(func, 1, 2, c=3)

    assert result == (1, 2, 3)


def test_circuit_breaker_multiple_circuits_independent():
    """Test that multiple circuit breakers are independent."""
    breaker1 = CircuitBreaker("service1", failure_threshold=1)
    breaker2 = CircuitBreaker("service2", failure_threshold=1)

    # Open breaker1
    with pytest.raises(OperationError):
        breaker1.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

    assert breaker1.state == CircuitState.OPEN
    assert breaker2.state == CircuitState.CLOSED

    # breaker2 should still work
    result = breaker2.call(lambda: "success")
    assert result == "success"
