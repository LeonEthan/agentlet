"""Circuit breaker pattern for resilient external service calls.

Prevents cascading failures when LLM APIs are experiencing issues by
failing fast when error threshold is exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import Callable, TypeVar

from agentlet.core.types import get_logger

logger = get_logger("agentlet.circuit_breaker")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing fast
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for resilient service calls.

    Args:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open
        half_open_max_calls: Max calls allowed in half-open state

    Usage:
        breaker = CircuitBreaker("openai", failure_threshold=5)

        try:
            result = breaker.call(lambda: model.complete(request))
        except CircuitOpenError:
            # Circuit is open, fail fast
            return cached_response
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _last_failure_time: float | None = field(default=None, repr=False)
    _half_open_calls: int = field(default=0, repr=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def call(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute operation with circuit breaker protection.

        Args:
            operation: Callable to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result from operation

        Raises:
            CircuitOpenError: If circuit is open
            OperationError: If operation fails (and circuit may trip)
        """
        self._update_state()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN - failing fast",
                circuit_name=self.name,
                retry_after=self._time_until_retry(),
            )

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' HALF_OPEN limit reached",
                    circuit_name=self.name,
                )
            self._half_open_calls += 1

        try:
            result = operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise OperationError(f"Operation failed: {e}") from e

    def _update_state(self) -> None:
        """Check if we should transition from OPEN to HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return

        if self._last_failure_time is None:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            return

        elapsed = monotonic() - self._last_failure_time
        if elapsed >= self.recovery_timeout:
            logger.info(
                f"Circuit '{self.name}' entering HALF_OPEN state",
                elapsed_seconds=elapsed,
            )
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0

    def _on_success(self) -> None:
        """Handle successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit '{self.name}' recovery confirmed - CLOSING")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self._failure_count += 1
        self._last_failure_time = monotonic()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(
                f"Circuit '{self.name}' recovery failed - OPENING",
                failure_count=self._failure_count,
            )
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit '{self.name}' threshold exceeded - OPENING",
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
            )
            self._state = CircuitState.OPEN

    def _time_until_retry(self) -> float:
        """Calculate seconds until retry allowed."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = monotonic() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    def force_close(self) -> None:
        """Manually reset circuit to CLOSED state."""
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_stats(self) -> dict[str, object]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "time_until_retry": self._time_until_retry(),
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        message: str,
        circuit_name: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.circuit_name = circuit_name
        self.retry_after = retry_after


class OperationError(Exception):
    """Raised when protected operation fails."""
    pass


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "OperationError",
]
