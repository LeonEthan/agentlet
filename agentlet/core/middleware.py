"""Request/response middleware for debugging and transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from agentlet.core.types import Timer, get_logger
from agentlet.llm.schemas import ModelRequest, ModelResponse

logger = get_logger("agentlet.middleware")


class RequestHandler(Protocol):
    def __call__(self, request: ModelRequest) -> ModelRequest: ...


class ResponseHandler(Protocol):
    def __call__(self, request: ModelRequest, response: ModelResponse) -> ModelResponse: ...


class ErrorHandler(Protocol):
    def __call__(self, request: ModelRequest, error: Exception) -> None: ...


@dataclass
class MiddlewareChain:
    """Chain of middleware handlers for request/response processing."""
    request_handlers: list[RequestHandler] = field(default_factory=list)
    response_handlers: list[ResponseHandler] = field(default_factory=list)
    error_handlers: list[ErrorHandler] = field(default_factory=list)

    def add_request_handler(self, handler: RequestHandler) -> "MiddlewareChain":
        self.request_handlers.append(handler)
        return self

    def add_response_handler(self, handler: ResponseHandler) -> "MiddlewareChain":
        self.response_handlers.append(handler)
        return self

    def add_error_handler(self, handler: ErrorHandler) -> "MiddlewareChain":
        self.error_handlers.append(handler)
        return self

    def process_request(self, request: ModelRequest) -> ModelRequest:
        for handler in self.request_handlers:
            request = handler(request)
        return request

    def process_response(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        for handler in reversed(self.response_handlers):
            response = handler(request, response)
        return response

    def handle_error(self, request: ModelRequest, error: Exception) -> None:
        for handler in self.error_handlers:
            handler(request, error)


def logging_request_handler(log_body: bool = False) -> RequestHandler:
    def handler(request: ModelRequest) -> ModelRequest:
        context = {"message_count": len(request.messages), "tool_count": len(request.tools)}
        if log_body:
            context["messages"] = [{"role": m.role, "content": m.content[:100] if m.content else None}
                                   for m in request.messages]
        logger.info("LLM request", **context)
        return request
    return handler


def logging_response_handler(log_body: bool = False) -> ResponseHandler:
    def handler(request: ModelRequest, response: ModelResponse) -> ModelResponse:
        context = {"finish_reason": response.finish_reason, "has_tool_calls": len(response.message.tool_calls) > 0}
        if response.usage:
            context.update({"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens})
        if log_body and response.message.content:
            context["content_preview"] = response.message.content[:100]
        logger.info("LLM response", **context)
        return response
    return handler


def timing_middleware() -> tuple[RequestHandler, ResponseHandler]:
    _timer_store: dict[int, Timer] = {}

    def request_handler(request: ModelRequest) -> ModelRequest:
        timer = Timer()
        timer.__enter__()
        _timer_store[id(request)] = timer
        return request

    def response_handler(request: ModelRequest, response: ModelResponse) -> ModelResponse:
        timer = _timer_store.pop(id(request), None)
        if timer:
            timer.__exit__(None, None, None)
            logger.info("LLM timing", elapsed_seconds=round(timer.elapsed_seconds, 3))
        return response

    return request_handler, response_handler


def header_injection_middleware(headers: dict[str, str]) -> RequestHandler:
    def handler(request: ModelRequest) -> ModelRequest:
        from dataclasses import replace
        new_metadata = dict(request.metadata)
        new_metadata.setdefault("headers", {}).update(headers)
        return replace(request, metadata=new_metadata)
    return handler


class MiddlewareClient:
    """Wrapper for ModelClient that applies middleware chain."""

    def __init__(self, client, middleware: MiddlewareChain | None = None) -> None:
        self._client = client
        self._middleware = middleware or MiddlewareChain()

    def complete(self, request: ModelRequest) -> ModelResponse:
        processed_request = self._middleware.process_request(request)
        try:
            response = self._client.complete(processed_request)
        except Exception as e:
            self._middleware.handle_error(processed_request, e)
            raise
        return self._middleware.process_response(processed_request, response)


__all__ = [
    "ErrorHandler", "MiddlewareChain", "MiddlewareClient", "RequestHandler", "ResponseHandler",
    "header_injection_middleware", "logging_request_handler", "logging_response_handler", "timing_middleware",
]
