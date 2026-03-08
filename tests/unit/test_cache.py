"""Tests for request caching and deduplication."""

from __future__ import annotations

import pytest

from agentlet.core.cache import (
    CacheEntry,
    CachingModelClient,
    InMemoryCacheBackend,
    RequestCache,
    generate_cache_key,
)
from agentlet.core.messages import Message
from agentlet.llm.schemas import ModelRequest, ModelResponse, ModelToolDefinition


class FakeModelClient:
    """Fake client for testing caching."""

    def __init__(self) -> None:
        self.call_count = 0
        self.responses: list[ModelResponse] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        response = ModelResponse(
            message=Message(role="assistant", content=f"Response {self.call_count}"),
            finish_reason="stop",
        )
        self.responses.append(response)
        return response


def test_generate_cache_key_deterministic():
    """Test that cache keys are deterministic for identical requests."""
    request1 = ModelRequest(messages=(Message(role="user", content="hello"),))
    request2 = ModelRequest(messages=(Message(role="user", content="hello"),))

    key1 = generate_cache_key(request1)
    key2 = generate_cache_key(request2)

    assert key1 == key2
    assert len(key1) == 32  # SHA256 truncated to 32 chars


def test_generate_cache_key_different_content():
    """Test that different content produces different keys."""
    request1 = ModelRequest(messages=(Message(role="user", content="hello"),))
    request2 = ModelRequest(messages=(Message(role="user", content="world"),))

    key1 = generate_cache_key(request1)
    key2 = generate_cache_key(request2)

    assert key1 != key2


def test_generate_cache_key_with_tools():
    """Test cache key generation with tools."""
    tools = (
        ModelToolDefinition(
            name="Read",
            description="Read a file",
            input_schema={"type": "object"},
        ),
    )
    request = ModelRequest(
        messages=(Message(role="user", content="read file"),),
        tools=tools,
    )

    key = generate_cache_key(request)

    assert len(key) == 32


def test_in_memory_cache_backend_store_and_get():
    """Test basic store and retrieve."""
    backend = InMemoryCacheBackend()
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )
    entry = CacheEntry(response=response)

    backend.set("key1", entry)
    retrieved = backend.get("key1")

    assert retrieved is not None
    assert retrieved.response == response


def test_in_memory_cache_backend_missing_key():
    """Test retrieving missing key returns None."""
    backend = InMemoryCacheBackend()

    result = backend.get("nonexistent")

    assert result is None


def test_in_memory_cache_backend_ttl_expiration():
    """Test that entries expire after TTL."""
    backend = InMemoryCacheBackend(default_ttl_seconds=0.01)
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )
    entry = CacheEntry(response=response)

    backend.set("key1", entry)

    # Should exist immediately
    assert backend.get("key1") is not None

    # Wait for expiration
    import time
    time.sleep(0.02)

    # Should be expired now
    assert backend.get("key1") is None


def test_in_memory_cache_backend_delete():
    """Test deleting entries."""
    backend = InMemoryCacheBackend()
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )
    entry = CacheEntry(response=response)

    backend.set("key1", entry)
    backend.delete("key1")

    assert backend.get("key1") is None


def test_in_memory_cache_backend_clear():
    """Test clearing all entries."""
    backend = InMemoryCacheBackend()
    entry = CacheEntry(
        response=ModelResponse(
            message=Message(role="assistant", content="Hi"),
            finish_reason="stop",
        )
    )

    backend.set("key1", entry)
    backend.set("key2", entry)
    backend.clear()

    assert backend.get("key1") is None
    assert backend.get("key2") is None


def test_request_cache_hit():
    """Test cache hit."""
    cache = RequestCache()
    request = ModelRequest(messages=(Message(role="user", content="hello"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )

    cache.set(request, response)
    cached = cache.get(request)

    assert cached == response
    assert cache.hits == 1
    assert cache.misses == 0


def test_request_cache_miss():
    """Test cache miss."""
    cache = RequestCache()
    request = ModelRequest(messages=(Message(role="user", content="hello"),))

    cached = cache.get(request)

    assert cached is None
    assert cache.hits == 0
    assert cache.misses == 1


def test_request_cache_hit_rate():
    """Test hit rate calculation."""
    cache = RequestCache()
    request = ModelRequest(messages=(Message(role="user", content="hello"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )

    assert cache.hit_rate == 0.0

    cache.get(request)  # Miss
    assert cache.hit_rate == 0.0

    cache.set(request, response)
    cache.get(request)  # Hit
    assert cache.hit_rate == 0.5


def test_request_cache_stats():
    """Test cache statistics."""
    cache = RequestCache()
    request = ModelRequest(messages=(Message(role="user", content="hello"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )

    cache.set(request, response)
    cache.get(request)

    stats = cache.get_stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 1.0


def test_request_cache_invalidate():
    """Test invalidating specific request."""
    cache = RequestCache()
    request = ModelRequest(messages=(Message(role="user", content="hello"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )

    cache.set(request, response)
    cache.invalidate(request)

    assert cache.get(request) is None


def test_caching_model_client_avoids_duplicate_calls():
    """Test that caching wrapper avoids duplicate API calls."""
    fake = FakeModelClient()
    cache = RequestCache()
    client = CachingModelClient(fake, cache)

    request = ModelRequest(messages=(Message(role="user", content="hello"),))

    # First call hits the API
    response1 = client.complete(request)
    assert fake.call_count == 1

    # Second call uses cache
    response2 = client.complete(request)
    assert fake.call_count == 1  # No new call
    assert response1 == response2


def test_caching_model_client_different_requests():
    """Test that different requests are cached separately."""
    fake = FakeModelClient()
    cache = RequestCache()
    client = CachingModelClient(fake, cache)

    request1 = ModelRequest(messages=(Message(role="user", content="hello"),))
    request2 = ModelRequest(messages=(Message(role="user", content="world"),))

    client.complete(request1)
    client.complete(request2)

    assert fake.call_count == 2


def test_caching_model_client_stats():
    """Test caching client exposes stats."""
    fake = FakeModelClient()
    client = CachingModelClient(fake)

    request = ModelRequest(messages=(Message(role="user", content="hello"),))

    client.complete(request)
    client.complete(request)

    stats = client.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1
