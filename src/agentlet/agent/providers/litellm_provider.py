from __future__ import annotations

"""LiteLLM-backed implementation of the provider contract."""

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentlet.agent.context import Message, ToolCall
from agentlet.agent.providers.registry import (
    LLMResponse,
    ProviderConfig,
    ProviderStreamEvent,
    TokenUsage,
)
from agentlet.agent.tools.registry import ToolSpec


CompletionFunc = Callable[..., Awaitable[Any]]
StreamCompletionFunc = Callable[..., Any]


@dataclass
class _ToolCallAccumulator:
    """Incrementally rebuild streamed tool calls from partial chunks."""

    id: str = ""
    name: str = ""
    arguments_parts: list[str] = field(default_factory=list)

    def update(self, raw_call: Any) -> None:
        call_id = _get_attr(raw_call, "id")
        if call_id is not None:
            self.id = str(call_id)

        function = _get_attr(raw_call, "function") or {}
        name = _get_attr(function, "name")
        if name is not None:
            self.name = str(name)

        arguments = _get_attr(function, "arguments")
        if arguments is None:
            return
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)
            return
        self.arguments_parts.append(json.dumps(arguments, ensure_ascii=False))

    def build(self) -> ToolCall:
        return ToolCall(
            id=self.id,
            name=self.name,
            arguments_json="".join(self.arguments_parts) or "{}",
        )


class LiteLLMProvider:
    """Translate agentlet's provider contract to LiteLLM and back."""

    def __init__(
        self,
        config: ProviderConfig,
        completion_func: CompletionFunc | None = None,
        stream_completion_func: StreamCompletionFunc | None = None,
    ) -> None:
        self.config = config
        self._completion_func = completion_func
        self._stream_completion_func = stream_completion_func

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Execute one completion request through LiteLLM.

        The request is assembled from normalized internal messages and optional
        tool schemas so the rest of the codebase never needs to speak LiteLLM's
        request dialect directly.
        """
        request = self._build_request(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        completion = self._completion_func or self._load_completion_func()
        response = await completion(**request)
        return self._normalize_response(response)

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Execute one streamed completion request and normalize its chunks."""
        request = self._build_request(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request["stream"] = True

        completion = self._stream_completion_func or self._load_completion_func()
        raw_stream = completion(**request)
        if inspect.isawaitable(raw_stream):
            raw_stream = await raw_stream

        if not hasattr(raw_stream, "__aiter__"):
            yield ProviderStreamEvent(
                kind="response_complete",
                response=self._normalize_response(raw_stream),
            )
            return

        content_parts: list[str] = []
        tool_calls: dict[int, _ToolCallAccumulator] = {}
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in raw_stream:
            choice = self._read_first_choice(chunk)
            delta = _get_attr(choice, "delta") or _get_attr(choice, "message") or {}

            delta_content = self._coerce_content(_get_attr(delta, "content"))
            if delta_content:
                content_parts.append(delta_content)
                yield ProviderStreamEvent(kind="content_delta", text=delta_content)

            raw_tool_calls = _get_attr(delta, "tool_calls") or []
            for position, raw_call in enumerate(raw_tool_calls):
                raw_index = _get_attr(raw_call, "index")
                try:
                    index = int(raw_index) if raw_index is not None else position
                except (TypeError, ValueError):
                    index = position
                accumulator = tool_calls.setdefault(index, _ToolCallAccumulator())
                accumulator.update(raw_call)

            choice_finish_reason = _get_attr(choice, "finish_reason")
            if choice_finish_reason is not None:
                finish_reason = str(choice_finish_reason)

            chunk_usage = self._read_usage(_get_attr(chunk, "usage"))
            if chunk_usage is not None:
                usage = chunk_usage

        yield ProviderStreamEvent(
            kind="response_complete",
            response=LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tuple(
                    tool_calls[index].build() for index in sorted(tool_calls)
                ),
                finish_reason=finish_reason,
                usage=usage,
            ),
        )

    def _load_completion_func(self) -> CompletionFunc:
        """Import LiteLLM lazily so tests can inject a fake completion function."""
        try:
            from litellm import acompletion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is required to use LiteLLMProvider. Install project dependencies first."
            ) from exc
        return acompletion

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Collapse the raw SDK response into agentlet's stable internal types."""
        choice = self._read_first_choice(response)
        message = _get_attr(choice, "message") or {}
        content = self._coerce_content(_get_attr(message, "content"))
        tool_calls = self._read_tool_calls(_get_attr(message, "tool_calls") or [])
        finish_reason = _get_attr(choice, "finish_reason")
        usage = self._read_usage(_get_attr(response, "usage"))
        return LLMResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
        )

    def _build_request(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build the LiteLLM request payload from normalized inputs."""
        request = {
            "model": model or self.config.model,
            "messages": [message.to_provider_dict() for message in messages],
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
        }

        resolved_max_tokens = (
            self.config.max_tokens if max_tokens is None else max_tokens
        )
        if resolved_max_tokens is not None:
            request["max_tokens"] = resolved_max_tokens
        if self.config.api_key is not None:
            request["api_key"] = self.config.api_key
        if self.config.api_base is not None:
            request["api_base"] = self.config.api_base
        if tools:
            request["tools"] = [tool.to_provider_dict() for tool in tools]
        return request

    def _read_first_choice(self, response: Any) -> Any:
        """Return the first completion choice or fail loudly on malformed output."""
        choices = _get_attr(response, "choices") or []
        if not choices:
            raise RuntimeError("Provider response did not include any choices.")
        return choices[0]

    def _read_tool_calls(self, raw_tool_calls: list[Any]) -> list[ToolCall]:
        """Normalize provider-specific tool call payloads into ToolCall objects."""
        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            function = _get_attr(raw_call, "function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(_get_attr(raw_call, "id") or ""),
                    name=str(_get_attr(function, "name") or ""),
                    arguments_json=self._coerce_arguments_json(
                        _get_attr(function, "arguments")
                    ),
                )
            )
        return tool_calls

    def _read_usage(self, raw_usage: Any) -> TokenUsage | None:
        """Return usage only when all standard token counters are present."""
        if raw_usage is None:
            return None

        prompt_tokens = _get_attr(raw_usage, "prompt_tokens")
        completion_tokens = _get_attr(raw_usage, "completion_tokens")
        total_tokens = _get_attr(raw_usage, "total_tokens")
        if None in (prompt_tokens, completion_tokens, total_tokens):
            return None

        return TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )

    def _coerce_content(self, content: Any) -> str | None:
        """Preserve string content and JSON-encode structured content when needed."""
        if content is None or isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _coerce_arguments_json(self, arguments: Any) -> str:
        """Ensure tool arguments are stored as JSON text for a stable boundary."""
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

def _get_attr(value: Any, key: str) -> Any:
    """Read both dict-style and attribute-style SDK objects transparently."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
