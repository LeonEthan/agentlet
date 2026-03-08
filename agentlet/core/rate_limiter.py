"""Rate limiting for API calls to prevent quota exhaustion."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Protocol

from agentlet.core.types import get_logger

logger = get_logger("agentlet.rate_limiter")


class RateLimiter(Protocol):
    def acquire(self) -> None: ...
    def try_acquire(self) -> bool: ...


@dataclass
class TokenBucketRateLimiter:
    """Token bucket rate limiter."""
    rate: float = 10.0
    capacity: float = 10.0
    _tokens: float = field(default=None, repr=False)  # type: ignore
    _last_update: float = field(default_factory=monotonic, repr=False)

    def __post_init__(self) -> None:
        if self._tokens is None:
            self._tokens = self.capacity

    def _add_tokens(self) -> None:
        now = monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last_update) * self.rate)
        self._last_update = now

    def acquire(self) -> None:
        self._add_tokens()
        if self._tokens < 1.0:
            wait_time = (1.0 - self._tokens) / self.rate
            logger.debug(f"Rate limit: waiting {wait_time:.3f}s")
            sleep(wait_time)
            self._add_tokens()
        self._tokens -= 1.0

    def try_acquire(self) -> bool:
        self._add_tokens()
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


@dataclass
class AdaptiveRateLimiter:
    """Rate limiter that adapts based on API responses."""
    target_rate: float = 10.0
    min_rate: float = 0.1
    capacity: float = 10.0
    _current_rate: float = field(default=0.0, repr=False)
    _limiter: TokenBucketRateLimiter = field(default=None, repr=False)  # type: ignore

    def __post_init__(self) -> None:
        if self._current_rate == 0:
            self._current_rate = self.target_rate
        if self._limiter is None:
            self._limiter = TokenBucketRateLimiter(rate=self._current_rate, capacity=self.capacity)

    def acquire(self) -> None:
        self._limiter.acquire()

    def try_acquire(self) -> bool:
        return self._limiter.try_acquire()

    def on_success(self) -> None:
        if self._current_rate < self.target_rate:
            self._current_rate = min(self.target_rate, self._current_rate * 1.05)
            self._limiter.rate = self._current_rate
            logger.debug(f"Rate increased to {self._current_rate:.2f}/s")

    def on_rate_limit_error(self) -> None:
        self._current_rate = max(self.min_rate, self._current_rate * 0.5)
        self._limiter.rate = self._current_rate
        logger.warning(f"Rate limited, reduced to {self._current_rate:.2f}/s")


@dataclass
class RateLimitedClient:
    """Wrapper that adds rate limiting to any ModelClient."""
    client: object
    limiter: RateLimiter = field(default_factory=lambda: TokenBucketRateLimiter())

    def complete(self, request):
        self.limiter.acquire()
        return self.client.complete(request)


__all__ = [
    "AdaptiveRateLimiter", "RateLimitedClient", "RateLimiter", "TokenBucketRateLimiter",
]
