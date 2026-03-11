from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from agentlet.agent.context import Message, ToolCall
from agentlet.agent.providers.registry import LLMResponse, ProviderConfig, TokenUsage
from agentlet.agent.tools.registry import ToolSpec


CompletionFunc = Callable[..., Awaitable[Any]]


class LiteLLMProvider:
    def __init__(
        self,
        config: ProviderConfig,
        completion_func: CompletionFunc | None = None,
    ) -> None:
        self.config = config
        self._completion_func = completion_func

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
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

        completion = self._completion_func or self._load_completion_func()
        response = await completion(**request)
        return self._normalize_response(response)

    def _load_completion_func(self) -> CompletionFunc:
        try:
            from litellm import acompletion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is required to use LiteLLMProvider. Install project dependencies first."
            ) from exc
        return acompletion

    def _normalize_response(self, response: Any) -> LLMResponse:
        choice = self._read_first_choice(response)
        message = self._get_attr(choice, "message") or {}
        content = self._coerce_content(self._get_attr(message, "content"))
        tool_calls = self._read_tool_calls(self._get_attr(message, "tool_calls") or [])
        finish_reason = self._get_attr(choice, "finish_reason")
        usage = self._read_usage(self._get_attr(response, "usage"))
        return LLMResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
        )

    def _read_first_choice(self, response: Any) -> Any:
        choices = self._get_attr(response, "choices") or []
        if not choices:
            raise RuntimeError("Provider response did not include any choices.")
        return choices[0]

    def _read_tool_calls(self, raw_tool_calls: list[Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            function = self._get_attr(raw_call, "function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(self._get_attr(raw_call, "id") or ""),
                    name=str(self._get_attr(function, "name") or ""),
                    arguments_json=self._coerce_arguments_json(
                        self._get_attr(function, "arguments")
                    ),
                )
            )
        return tool_calls

    def _read_usage(self, raw_usage: Any) -> TokenUsage | None:
        if raw_usage is None:
            return None

        prompt_tokens = self._get_attr(raw_usage, "prompt_tokens")
        completion_tokens = self._get_attr(raw_usage, "completion_tokens")
        total_tokens = self._get_attr(raw_usage, "total_tokens")
        if None in (prompt_tokens, completion_tokens, total_tokens):
            return None

        return TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )

    def _coerce_content(self, content: Any) -> str | None:
        if content is None or isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _coerce_arguments_json(self, arguments: Any) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    def _get_attr(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
