"""Rate limiting for API calls to prevent quota exhaustion."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Protocol

from agentlet.core.types import get_logger

logger = get_logger("agentlet.rate_limiter")


class RateLimiter(Protocol):
    """Protocol for rate limiting implementations."""

    def acquire(self) -> None:
        """Acquire permission to proceed. Blocks if necessary."""

    def try_acquire(self) -> bool:
        """Try to acquire permission without blocking. Returns success."""


@dataclass
class TokenBucketRateLimiter:
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining average rate.

    Args:
        rate: Tokens per second
        capacity: Maximum bucket size (burst capacity)
    """

    rate: float = 10.0  # tokens per second
    capacity: float = 10.0  # bucket capacity

    _tokens: float = field(default=None, repr=False)  # type: ignore
    _last_update: float = field(default_factory=monotonic, repr=False)

    def __post_init__(self) -> None:
        if self._tokens is None:
            self._tokens = self.capacity  # Start with full bucket

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        self._add_tokens()

        if self._tokens < 1.0:
            # Need to wait for tokens
            needed = 1.0 - self._tokens
            wait_time = needed / self.rate
            logger.debug(f"Rate limit: waiting {wait_time:.3f}s")
            sleep(wait_time)
            self._add_tokens()

        self._tokens -= 1.0

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking."""
        self._add_tokens()

        if self._tokens < 1.0:
            return False

        self._tokens -= 1.0
        return True


@dataclass
class AdaptiveRateLimiter:
    """Rate limiter that adapts based on API responses.

    Automatically reduces rate when rate limit errors occur,
    and gradually increases back to target rate on success.
    """

    target_rate: float = 10.0
    min_rate: float = 0.1
    capacity: float = 10.0

    _current_rate: float = field(default=0.0, repr=False)
    _limiter: TokenBucketRateLimiter = field(default=None, repr=False)  # type: ignore

    def __post_init__(self) -> None:
        if self._current_rate == 0:
            self._current_rate = self.target_rate
        if self._limiter is None:
            self._limiter = TokenBucketRateLimiter(
                rate=self._current_rate,
                capacity=self.capacity,
            )

    def acquire(self) -> None:
        """Acquire permission to proceed."""
        self._limiter.acquire()

    def try_acquire(self) -> bool:
        """Try to acquire permission without blocking."""
        return self._limiter.try_acquire()

    def on_success(self) -> None:
        """Call when request succeeds - gradually increase rate."""
        if self._current_rate < self.target_rate:
            self._current_rate = min(
                self.target_rate,
                self._current_rate * 1.05,  # 5% increase
            )
            self._limiter.rate = self._current_rate
            logger.debug(f"Rate increased to {self._current_rate:.2f}/s")

    def on_rate_limit_error(self) -> None:
        """Call when rate limit error occurs - reduce rate."""
        self._current_rate = max(
            self.min_rate,
            self._current_rate * 0.5,  # 50% decrease
        )
        self._limiter.rate = self._current_rate
        logger.warning(f"Rate limited, reduced to {self._current_rate:.2f}/s")


@dataclass
class RateLimitedClient:
    """Wrapper that adds rate limiting to any ModelClient."""

    client: object
    limiter: RateLimiter = field(default_factory=lambda: TokenBucketRateLimiter())

    def complete(self, request):
        """Complete with rate limiting."""
        self.limiter.acquire()
        return self.client.complete(request)


__all__ = [
    "AdaptiveRateLimiter",
    "RateLimitedClient",
    "RateLimiter",
    "TokenBucketRateLimiter",
]
