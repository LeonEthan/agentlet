"""Tests for retry_with_backoff decorator."""

import pytest
from unittest.mock import Mock, call

from agentlet.core.types import retry_with_backoff


class RetryableError(Exception):
    """Exception that should trigger retry."""


class NonRetryableError(Exception):
    """Exception that should not trigger retry."""


def test_retry_succeeds_after_transient_failures():
    """Test that retry succeeds after transient failures."""
    mock_func = Mock(side_effect=[RetryableError(), RetryableError(), "success"])

    @retry_with_backoff(
        max_retries=3,
        base_delay=0.01,
        retryable_exceptions=(RetryableError,),
    )
    def func():
        return mock_func()

    result = func()
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_exhausted_raises_last_exception():
    """Test that retry raises last exception when max retries exhausted."""
    mock_func = Mock(side_effect=RetryableError("failure"))

    @retry_with_backoff(
        max_retries=2,
        base_delay=0.01,
        retryable_exceptions=(RetryableError,),
    )
    def func():
        return mock_func()

    with pytest.raises(RetryableError, match="failure"):
        func()
    assert mock_func.call_count == 3  # initial + 2 retries


def test_non_retryable_exception_raises_immediately():
    """Test that non-retryable exceptions are raised immediately."""
    mock_func = Mock(side_effect=NonRetryableError("immediate"))

    @retry_with_backoff(
        max_retries=3,
        base_delay=0.01,
        retryable_exceptions=(RetryableError,),
    )
    def func():
        return mock_func()

    with pytest.raises(NonRetryableError, match="immediate"):
        func()
    assert mock_func.call_count == 1


def test_success_on_first_call_no_retry():
    """Test that successful calls don't trigger retry."""
    mock_func = Mock(return_value="success")

    @retry_with_backoff(
        max_retries=3,
        base_delay=0.01,
        retryable_exceptions=(RetryableError,),
    )
    def func():
        return mock_func()

    result = func()
    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_preserves_function_metadata():
    """Test that decorator preserves function metadata."""

    @retry_with_backoff(max_retries=1, base_delay=0.01)
    def my_function():
        """My docstring."""
        return "result"

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."
