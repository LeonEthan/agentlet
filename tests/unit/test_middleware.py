"""Tests for request/response middleware."""

from __future__ import annotations

import pytest

from agentlet.core.messages import Message
from agentlet.core.middleware import (
    MiddlewareChain,
    MiddlewareClient,
    header_injection_middleware,
    logging_request_handler,
    logging_response_handler,
    timing_middleware,
)
from agentlet.llm.schemas import ModelRequest, ModelResponse, TokenUsage


class FakeModelClient:
    """Fake client for testing middleware."""

    def __init__(self, response: ModelResponse | None = None) -> None:
        self.response = response or ModelResponse(
            message=Message(role="assistant", content="Hello"),
            finish_reason="stop",
        )
        self.last_request: ModelRequest | None = None
        self.call_count = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        self.call_count += 1
        return self.response


def test_middleware_chain_processes_requests():
    """Test that request handlers are applied in order."""
    chain = MiddlewareChain()

    calls = []

    def handler1(request: ModelRequest) -> ModelRequest:
        calls.append("h1")
        return request

    def handler2(request: ModelRequest) -> ModelRequest:
        calls.append("h2")
        return request

    chain.add_request_handler(handler1).add_request_handler(handler2)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    result = chain.process_request(request)

    assert calls == ["h1", "h2"]
    assert result == request


def test_middleware_chain_processes_responses():
    """Test that response handlers are applied in reverse order."""
    chain = MiddlewareChain()

    calls = []

    def handler1(request: ModelRequest, response: ModelResponse) -> ModelResponse:
        calls.append("h1")
        return response

    def handler2(request: ModelRequest, response: ModelResponse) -> ModelResponse:
        calls.append("h2")
        return response

    chain.add_response_handler(handler1).add_response_handler(handler2)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )
    result = chain.process_response(request, response)

    # Response handlers run in reverse order
    assert calls == ["h2", "h1"]
    assert result == response


def test_middleware_chain_can_transform_request():
    """Test that request handlers can modify requests."""
    chain = MiddlewareChain()

    def add_metadata(request: ModelRequest) -> ModelRequest:
        from dataclasses import replace

        new_metadata = dict(request.metadata)
        new_metadata["added"] = True
        return replace(request, metadata=new_metadata)

    chain.add_request_handler(add_metadata)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    result = chain.process_request(request)

    assert result.metadata.get("added") is True


def test_middleware_chain_can_transform_response():
    """Test that response handlers can modify responses."""
    chain = MiddlewareChain()

    def add_usage(request: ModelRequest, response: ModelResponse) -> ModelResponse:
        from dataclasses import replace

        if response.usage is None:
            return replace(response, usage=TokenUsage(input_tokens=10, output_tokens=5))
        return response

    chain.add_response_handler(add_usage)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )
    result = chain.process_response(request, response)

    assert result.usage is not None
    assert result.usage.input_tokens == 10


def test_middleware_client_wraps_underlying_client():
    """Test MiddlewareClient wraps a ModelClient."""
    fake = FakeModelClient()
    client = MiddlewareClient(fake)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    response = client.complete(request)

    assert fake.call_count == 1
    assert response.finish_reason == "stop"


def test_middleware_client_applies_chain():
    """Test that MiddlewareClient applies middleware chain."""
    fake = FakeModelClient()
    chain = MiddlewareChain()

    # Add a request handler that modifies the request
    def add_tag(request: ModelRequest) -> ModelRequest:
        from dataclasses import replace

        new_metadata = dict(request.metadata)
        new_metadata["tag"] = "test"
        return replace(request, metadata=new_metadata)

    chain.add_request_handler(add_tag)
    client = MiddlewareClient(fake, chain)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    client.complete(request)

    # The underlying client should see the modified request
    assert fake.last_request is not None
    assert fake.last_request.metadata.get("tag") == "test"


def test_logging_request_handler():
    """Test logging request handler creation."""
    handler = logging_request_handler(log_body=False)

    request = ModelRequest(
        messages=(Message(role="user", content="test"),),
        tools=(),
    )
    result = handler(request)

    # Should pass through unchanged
    assert result == request


def test_logging_response_handler():
    """Test logging response handler creation."""
    handler = logging_response_handler(log_body=False)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hello"),
        finish_reason="stop",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    result = handler(request, response)

    # Should pass through unchanged
    assert result == response


def test_timing_middleware():
    """Test timing middleware creates handlers."""
    request_handler, response_handler = timing_middleware()

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    response = ModelResponse(
        message=Message(role="assistant", content="Hi"),
        finish_reason="stop",
    )

    # Apply request handler
    processed_request = request_handler(request)
    assert processed_request == request

    # Apply response handler (should log timing)
    result = response_handler(request, response)
    assert result == response


def test_header_injection_middleware():
    """Test header injection middleware."""
    handler = header_injection_middleware({"X-Custom": "value"})

    request = ModelRequest(
        messages=(Message(role="user", content="test"),),
        metadata={},
    )
    result = handler(request)

    assert result.metadata["headers"]["X-Custom"] == "value"


def test_header_injection_preserves_existing():
    """Test header injection preserves existing headers."""
    handler = header_injection_middleware({"X-New": "new"})

    request = ModelRequest(
        messages=(Message(role="user", content="test"),),
        metadata={"headers": {"X-Existing": "old"}},
    )
    result = handler(request)

    assert result.metadata["headers"]["X-Existing"] == "old"
    assert result.metadata["headers"]["X-New"] == "new"


def test_middleware_chain_error_handler():
    """Test error handlers are called."""
    chain = MiddlewareChain()

    errors_caught = []

    def error_handler(request: ModelRequest, error: Exception) -> None:
        errors_caught.append((request, error))

    chain.add_error_handler(error_handler)

    request = ModelRequest(messages=(Message(role="user", content="test"),))
    error = ValueError("test error")

    chain.handle_error(request, error)

    assert len(errors_caught) == 1
    assert errors_caught[0][0] == request
    assert errors_caught[0][1] == error
