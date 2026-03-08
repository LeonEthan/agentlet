"""Request caching and deduplication for LLM calls.

Avoids redundant API calls for identical requests within a time window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol

from agentlet.core.types import JSONObject, get_logger
from agentlet.llm.schemas import ModelRequest, ModelResponse

logger = get_logger("agentlet.cache")


def _make_hashable(obj: object) -> object:
    """Convert nested structure to hashable form."""
    if isinstance(obj, dict):
        return tuple((k, _make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    return obj


def generate_cache_key(request: ModelRequest) -> str:
    """Generate deterministic cache key for a request.

    The key is based on message content and tool definitions,
    ignoring metadata that doesn't affect the response.
    """
    # Extract relevant parts for caching
    cache_data: JSONObject = {
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "name": m.name,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in m.tool_calls
                ] if m.tool_calls else None,
            }
            for m in request.messages
        ],
        "tools": [
            {"name": t.name, "description": t.description, "schema": t.input_schema}
            for t in sorted(request.tools, key=lambda t: t.name)
        ] if request.tools else None,
        "tool_choice": request.tool_choice.mode if request.tool_choice else None,
    }

    # Create deterministic hash
    hashable = _make_hashable(cache_data)
    json_str = json.dumps(hashable, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:32]


@dataclass
class CacheEntry:
    """Single cached response with metadata."""

    response: ModelResponse
    created_at: float = field(default_factory=monotonic)
    hit_count: int = 0


class CacheBackend(Protocol):
    """Protocol for cache storage backends."""

    def get(self, key: str) -> CacheEntry | None:
        """Get entry by key. Returns None if not found or expired."""

    def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry under key."""

    def delete(self, key: str) -> None:
        """Remove entry by key."""

    def clear(self) -> None:
        """Remove all entries."""


@dataclass
class InMemoryCacheBackend:
    """Simple in-memory cache backend with TTL support."""

    default_ttl_seconds: float | None = None
    _storage: dict[str, CacheEntry] = field(default_factory=dict)

    def get(self, key: str) -> CacheEntry | None:
        """Get entry if it exists and hasn't expired."""
        entry = self._storage.get(key)
        if entry is None:
            return None

        # Check TTL
        if self.default_ttl_seconds is not None:
            age = monotonic() - entry.created_at
            if age > self.default_ttl_seconds:
                del self._storage[key]
                return None

        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry."""
        self._storage[key] = entry

    def delete(self, key: str) -> None:
        """Remove entry."""
        self._storage.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        self._storage.clear()


@dataclass
class RequestCache:
    """Request cache with hit tracking and TTL.

    Usage:
        cache = RequestCache(ttl_seconds=300)  # 5 minute cache

        # Check cache
        cached = cache.get(request)
        if cached:
            return cached

        # Store result
        response = model.complete(request)
        cache.set(request, response)
    """

    backend: CacheBackend = field(default_factory=InMemoryCacheBackend)
    hits: int = 0
    misses: int = 0

    def get(self, request: ModelRequest) -> ModelResponse | None:
        """Get cached response if available."""
        key = generate_cache_key(request)
        entry = self.backend.get(key)

        if entry is None:
            self.misses += 1
            return None

        entry.hit_count += 1
        self.hits += 1
        logger.debug("Cache hit", key=key, hit_count=entry.hit_count)
        return entry.response

    def set(self, request: ModelRequest, response: ModelResponse) -> None:
        """Cache a response."""
        key = generate_cache_key(request)
        entry = CacheEntry(response=response)
        self.backend.set(key, entry)
        logger.debug("Cache store", key=key)

    def invalidate(self, request: ModelRequest) -> None:
        """Remove a specific request from cache."""
        key = generate_cache_key(request)
        self.backend.delete(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.backend.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def get_stats(self) -> dict[str, object]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


class CachingModelClient:
    """Wrapper that adds caching to any ModelClient."""

    def __init__(self, client, cache: RequestCache | None = None) -> None:
        self._client = client
        self._cache = cache or RequestCache()

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete with caching."""
        # Check cache first
        cached = self._cache.get(request)
        if cached is not None:
            return cached

        # Fetch and cache
        response = self._client.complete(request)
        self._cache.set(request, response)
        return response

    @property
    def cache_stats(self) -> dict[str, object]:
        """Get cache statistics."""
        return self._cache.get_stats()


__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CachingModelClient",
    "InMemoryCacheBackend",
    "RequestCache",
    "generate_cache_key",
]
