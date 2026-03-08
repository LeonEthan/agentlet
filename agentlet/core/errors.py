"""Error categorization and smart recovery for resilient API operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, TypeVar

from agentlet.core.types import deep_copy_json, get_logger

if TYPE_CHECKING:
    from agentlet.core.circuit_breaker import CircuitBreaker

logger = get_logger("agentlet.errors")
T = TypeVar("T")


class ErrorCategory(Enum):
    """Categorized error types for intelligent recovery decisions."""
    TRANSIENT = auto()
    AUTH = auto()
    VALIDATION = auto()
    RESOURCE = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()
    FATAL = auto()


class RecoveryAction(Enum):
    """Recommended recovery actions based on error categorization."""
    RETRY_IMMEDIATE = auto()
    RETRY_BACKOFF = auto()
    RETRY_ADAPTIVE = auto()
    FAIL_CIRCUIT = auto()
    FAIL_FAST = auto()
    FAIL_USER = auto()
    FAIL_ESCALATE = auto()


@dataclass(frozen=True, slots=True)
class CategorizedError:
    """Structured error with categorization and recovery guidance."""

    original: Exception
    category: ErrorCategory
    action: RecoveryAction
    user_message: str
    retryable: bool = False
    retry_after_seconds: float | None = None
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = deep_copy_json(self.context) if self.context else {}
        if not isinstance(copied, dict):
            copied = {}
        object.__setattr__(self, "context", copied)


# Error patterns: (exception_type, substrings) -> (category, action, message)
_ERROR_PATTERNS: list[tuple[type[Exception] | None, list[str], ErrorCategory, RecoveryAction, str]] = [
    (None, ["rate limit", "ratelimit", "too many requests", "429"], ErrorCategory.TRANSIENT, RecoveryAction.RETRY_ADAPTIVE, "Rate limit exceeded. Adjusting request pace..."),
    (None, ["service unavailable", "503", "gateway timeout", "504", "temporarily unavailable", "overloaded", "throttle"], ErrorCategory.TRANSIENT, RecoveryAction.RETRY_BACKOFF, "Service temporarily unavailable. Retrying..."),
    (TimeoutError, [], ErrorCategory.TIMEOUT, RecoveryAction.RETRY_BACKOFF, "Request timed out. Retrying with backoff..."),
    (None, ["timeout", "timed out", "deadline exceeded", "read timeout", "connection timeout", "request timeout"], ErrorCategory.TIMEOUT, RecoveryAction.RETRY_BACKOFF, "Request timed out. Retrying with backoff..."),
    (None, ["authentication", "authorization", "unauthorized", "401", "forbidden", "403", "invalid api key", "invalid token", "api key expired", "credentials"], ErrorCategory.AUTH, RecoveryAction.FAIL_USER, "Authentication failed. Please check your API credentials."),
    (None, ["quota exceeded", "insufficient quota", "resource exhausted", "disk full", "no space", "memory limit"], ErrorCategory.RESOURCE, RecoveryAction.FAIL_ESCALATE, "Resource limit exceeded. Please check your quota or try again later."),
    (ValueError, [], ErrorCategory.VALIDATION, RecoveryAction.FAIL_FAST, "Invalid request. Please check your input and try again."),
    (None, ["validation", "invalid", "malformed", "schema", "required field", "bad request", "400", "unprocessable", "422"], ErrorCategory.VALIDATION, RecoveryAction.FAIL_FAST, "Invalid request. Please check your input."),
]

# HTTP status code -> (category, action, message)
_STATUS_CODES: dict[int, tuple[ErrorCategory, RecoveryAction, str]] = {
    429: (ErrorCategory.TRANSIENT, RecoveryAction.RETRY_ADAPTIVE, "Rate limit exceeded. Adjusting request pace..."),
    500: (ErrorCategory.TRANSIENT, RecoveryAction.RETRY_BACKOFF, "Service temporarily unavailable. Retrying..."),
    502: (ErrorCategory.TRANSIENT, RecoveryAction.RETRY_BACKOFF, "Service temporarily unavailable. Retrying..."),
    503: (ErrorCategory.TRANSIENT, RecoveryAction.RETRY_BACKOFF, "Service temporarily unavailable. Retrying..."),
    504: (ErrorCategory.TRANSIENT, RecoveryAction.RETRY_BACKOFF, "Service temporarily unavailable. Retrying..."),
    401: (ErrorCategory.AUTH, RecoveryAction.FAIL_USER, "Authentication failed. Please check your credentials."),
    403: (ErrorCategory.AUTH, RecoveryAction.FAIL_USER, "Authentication failed. Please check your credentials."),
    400: (ErrorCategory.VALIDATION, RecoveryAction.FAIL_FAST, "Invalid request. Please check your input."),
    408: (ErrorCategory.TIMEOUT, RecoveryAction.RETRY_BACKOFF, "Request timeout. Retrying..."),
}

RETRYABLE_ACTIONS = {RecoveryAction.RETRY_IMMEDIATE, RecoveryAction.RETRY_BACKOFF, RecoveryAction.RETRY_ADAPTIVE}


def _extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP status code from exception if present."""
    for attr in ["status_code", "code", "response_code", "http_status"]:
        if hasattr(exc, attr):
            try:
                code = int(getattr(exc, attr))
                if 100 <= code < 600:
                    return code
            except (TypeError, ValueError):
                continue
    if hasattr(exc, "response"):
        response = getattr(exc, "response")
        if hasattr(response, "status_code"):
            try:
                return int(response.status_code)
            except (TypeError, ValueError):
                pass
    return None


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract retry-after duration from exception if present."""
    for attr in ["retry_after", "retry_after_seconds", "wait_seconds"]:
        if hasattr(exc, attr):
            try:
                return float(getattr(exc, attr))
            except (TypeError, ValueError):
                continue
    if hasattr(exc, "response"):
        response = getattr(exc, "response")
        if hasattr(response, "headers"):
            headers = getattr(response, "headers", {})
            retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
            if retry_after:
                try:
                    return float(retry_after)
                except (TypeError, ValueError):
                    pass
    return None


def categorize_error(exc: Exception, context: dict[str, object] | None = None) -> CategorizedError:
    """Categorize an exception and determine recovery strategy."""
    error_str = f"{type(exc).__name__}: {exc}".lower()
    exc_type = type(exc)

    # Check for explicit status codes
    status_code = _extract_status_code(exc)
    if status_code and status_code in _STATUS_CODES:
        cat, action, msg = _STATUS_CODES[status_code]
        return CategorizedError(
            original=exc, category=cat, action=action, user_message=msg,
            retryable=action in RETRYABLE_ACTIONS,
            retry_after_seconds=_extract_retry_after(exc) if status_code == 429 else None,
            context=context or {},
        )

    # Pattern-based categorization
    for exception_type, substrings, category, action, message in _ERROR_PATTERNS:
        if exception_type is not None and issubclass(exc_type, exception_type):
            return CategorizedError(
                original=exc, category=category, action=action, user_message=message,
                retryable=action in RETRYABLE_ACTIONS, context=context or {},
            )
        for substr in substrings:
            if substr in error_str:
                return CategorizedError(
                    original=exc, category=category, action=action, user_message=message,
                    retryable=action in RETRYABLE_ACTIONS, context=context or {},
                )

    # Default: unknown but retryable
    return CategorizedError(
        original=exc, category=ErrorCategory.UNKNOWN, action=RecoveryAction.RETRY_BACKOFF,
        user_message=f"An unexpected error occurred: {exc}", retryable=True, context=context or {},
    )


@dataclass
class RecoveryExecutor:
    """Executes recovery strategies for categorized errors."""

    circuit_breaker: CircuitBreaker | None = None
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0

    def execute(self, operation: Callable[[], T], operation_name: str = "operation") -> T:
        """Execute operation with automatic error recovery."""
        from time import sleep
        import random

        last_error: CategorizedError | None = None

        for attempt in range(self.max_retries + 1):
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                raise CircuitOpenError(f"Circuit breaker open for {operation_name}. Service temporarily unavailable.")

            try:
                result = operation()
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                return result
            except Exception as exc:
                categorized = categorize_error(exc, context={"attempt": attempt, "operation": operation_name})
                last_error = categorized

                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {categorized.category.name}",
                    category=categorized.category.name, action=categorized.action.name, attempt=attempt,
                )

                if not categorized.retryable or attempt >= self.max_retries:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise categorized.original

                delay = self._calculate_delay(categorized, attempt)
                if delay > self.max_delay:
                    logger.error(f"Delay {delay:.1f}s exceeds max {self.max_delay:.1f}s, giving up")
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                    raise categorized.original

                logger.debug(f"Retrying {operation_name} in {delay:.2f}s")
                sleep(delay)

        if last_error:
            raise last_error.original
        raise RuntimeError("Recovery loop exited without result or error")

    def _calculate_delay(self, categorized: CategorizedError, attempt: int) -> float:
        """Calculate delay before next retry."""
        if categorized.retry_after_seconds is not None:
            return categorized.retry_after_seconds
        import random
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self.max_delay)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class ErrorAggregator:
    """Aggregates errors for pattern detection and alerting."""

    window_size: int = 100

    def __post_init__(self):
        self._errors: list[tuple[float, ErrorCategory]] = []
        self._lock = __import__("threading").Lock()

    def record(self, categorized: CategorizedError) -> None:
        """Record a categorized error."""
        from time import time
        with self._lock:
            self._errors.append((time(), categorized.category))
            if len(self._errors) > self.window_size:
                self._errors = self._errors[-self.window_size:]

    def get_stats(self, window_seconds: float = 300.0) -> dict[str, object]:
        """Get error statistics for the specified time window."""
        from time import time
        from collections import Counter

        cutoff = time() - window_seconds
        with self._lock:
            recent = [cat for ts, cat in self._errors if ts >= cutoff]
        counts = Counter(recent)
        total = len(recent)
        return {
            "total": total,
            "by_category": {cat.name: count for cat, count in counts.items()},
            "transient_rate": counts.get(ErrorCategory.TRANSIENT, 0) / max(total, 1),
            "error_rate": total / max(window_seconds, 1),
        }

    def is_healthy(self, threshold: float = 0.1) -> bool:
        """Check if error rate is within acceptable threshold."""
        return self.get_stats(window_seconds=60.0)["transient_rate"] < threshold


__all__ = [
    "CategorizedError",
    "CircuitOpenError",
    "ErrorAggregator",
    "ErrorCategory",
    "RecoveryAction",
    "RecoveryExecutor",
    "categorize_error",
]
