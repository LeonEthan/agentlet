"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import JSONObject, TokenUsage, deep_copy_json_object
from agentlet.llm.base import ModelClient
from agentlet.llm.schemas import ModelRequest, ModelResponse, ModelToolDefinition, ToolChoice

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


@runtime_checkable
class AnthropicTransport(Protocol):
    """Callable transport used by the Anthropic adapter."""

    def __call__(self, payload: JSONObject) -> JSONObject:
        """Send one provider request and return the raw provider response."""


@dataclass(frozen=True, slots=True)
class AnthropicModelClient(ModelClient):
    """Concrete adapter for the Anthropic Messages API."""

    model: str
    max_output_tokens: int
    transport: AnthropicTransport
    request_defaults: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0")
        object.__setattr__(
            self,
            "request_defaults",
            deep_copy_json_object(self.request_defaults),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = build_anthropic_request(
            request,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            request_defaults=self.request_defaults,
        )
        response_payload = self.transport(payload)
        return parse_anthropic_response(response_payload)


def build_anthropic_transport(
    *,
    api_key: str,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    timeout_seconds: float = 60.0,
) -> AnthropicTransport:
    """Build a JSON-over-HTTP transport for the Anthropic Messages API."""

    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if not anthropic_version.strip():
        raise ValueError("anthropic_version must not be empty")
    if not base_url.strip():
        raise ValueError("base_url must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    endpoint = f"{base_url.rstrip('/')}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
    }

    def transport(payload: JSONObject) -> JSONObject:
        request_body = json.dumps(payload).encode("utf-8")
        request = Request(
            endpoint,
            data=request_body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"anthropic request failed with HTTP {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"anthropic request failed: {exc.reason}") from exc

        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "anthropic response body was not valid JSON"
            ) from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("anthropic response payload must be a JSON object")
        return response_payload

    return transport


def build_anthropic_request(
    request: ModelRequest,
    *,
    model: str,
    max_output_tokens: int,
    request_defaults: JSONObject | None = None,
) -> JSONObject:
    """Build one Anthropic Messages API payload."""

    if not model.strip():
        raise ValueError("model must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be > 0")

    payload = deep_copy_json_object(request_defaults or {})
    payload["model"] = model
    payload["max_tokens"] = max_output_tokens

    system_parts = [
        message.content
        for message in request.messages
        if message.role == "system" and message.content
    ]
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    payload["messages"] = _build_messages_payload(request.messages)
    if request.tools:
        payload["tools"] = [
            _build_tool_payload(tool)
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = _build_tool_choice_payload(request.tool_choice)
    return payload


def parse_anthropic_response(payload: JSONObject) -> ModelResponse:
    """Normalize one Anthropic response into the shared schema."""

    content_payload = payload.get("content")
    if not isinstance(content_payload, list):
        raise ValueError("anthropic response content must be a list")

    stop_reason = payload.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError("anthropic response stop_reason must be a non-empty string")

    response_metadata: JSONObject = {
        "provider": "anthropic",
        "stop_reason": stop_reason,
    }
    if payload.get("id") is not None:
        response_metadata["response_id"] = str(payload["id"])
    if payload.get("model") is not None:
        response_metadata["model"] = str(payload["model"])
    if payload.get("stop_sequence") is not None:
        response_metadata["stop_sequence"] = str(payload["stop_sequence"])

    content, tool_calls = _parse_assistant_content(content_payload)
    return ModelResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        finish_reason=_normalize_finish_reason(stop_reason),
        usage=_parse_usage(payload.get("usage")),
        metadata=response_metadata,
    )


def _build_messages_payload(messages: tuple[Message, ...]) -> list[JSONObject]:
    payload_messages: list[JSONObject] = []
    pending_tool_results: list[JSONObject] = []

    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            pending_tool_results.append(_build_tool_result_block(message))
            continue
        if message.role == "user" and pending_tool_results:
            combined_content = list(pending_tool_results)
            if message.content:
                combined_content.append({"type": "text", "text": message.content})
            payload_messages.append(
                {
                    "role": "user",
                    "content": combined_content,
                }
            )
            pending_tool_results = []
            continue
        if pending_tool_results:
            payload_messages.append(
                {
                    "role": "user",
                    "content": list(pending_tool_results),
                }
            )
            pending_tool_results = []
        payload_messages.append(_build_message_payload(message))

    if pending_tool_results:
        payload_messages.append(
            {
                "role": "user",
                "content": list(pending_tool_results),
            }
        )

    return payload_messages


def _build_message_payload(message: Message) -> JSONObject:
    if message.role == "user":
        return {"role": "user", "content": message.content}
    if message.role != "assistant":
        raise ValueError(f"unsupported Anthropic message role: {message.role}")

    content: list[JSONObject] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    content.extend(
        _build_tool_use_block(tool_call)
        for tool_call in message.tool_calls
    )
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "assistant", "content": content}


def _build_tool_payload(tool: ModelToolDefinition) -> JSONObject:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": deep_copy_json_object(tool.input_schema),
    }


def _build_tool_choice_payload(tool_choice: ToolChoice) -> JSONObject:
    if tool_choice.mode == "auto":
        return {"type": "auto"}
    if tool_choice.mode == "required":
        return {"type": "any"}
    if tool_choice.mode == "tool":
        return {"type": "tool", "name": tool_choice.tool_name or ""}
    return {"type": "none"}


def _build_tool_use_block(tool_call: ToolCall) -> JSONObject:
    return {
        "type": "tool_use",
        "id": tool_call.id,
        "name": tool_call.name,
        "input": deep_copy_json_object(tool_call.arguments),
    }


def _build_tool_result_block(message: Message) -> JSONObject:
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id or "",
        "content": message.content,
    }


def _parse_assistant_content(content_payload: list[object]) -> tuple[str, tuple[ToolCall, ...]]:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for index, block_payload in enumerate(content_payload):
        block = _require_mapping(block_payload, f"content[{index}]")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"content[{index}].text must be a string")
            text_parts.append(text)
            continue
        if block_type == "tool_use":
            tool_id = block.get("id")
            tool_name = block.get("name")
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError(f"content[{index}].id must be a non-empty string")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"content[{index}].name must be a non-empty string")
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                raise ValueError(f"content[{index}].input must be an object")
            tool_calls.append(
                ToolCall(
                    id=tool_id,
                    name=tool_name,
                    arguments=deep_copy_json_object(tool_input),
                    metadata={"provider_type": "tool_use"},
                )
            )
            continue
        raise ValueError(
            f"unsupported anthropic assistant content block type: {block_type}"
        )

    return "".join(text_parts), tuple(tool_calls)


def _normalize_finish_reason(stop_reason: str) -> str:
    if stop_reason in {"end_turn", "pause_turn", "refusal"}:
        return "stop"
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return stop_reason


def _parse_usage(payload: object) -> TokenUsage | None:
    if payload is None:
        return None
    usage_payload = _require_mapping(payload, "usage")
    return TokenUsage(
        input_tokens=int(usage_payload.get("input_tokens", 0)),
        output_tokens=int(usage_payload.get("output_tokens", 0)),
    )


def _require_mapping(payload: object, path: str) -> JSONObject:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    return payload


__all__ = [
    "AnthropicModelClient",
    "AnthropicTransport",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_ANTHROPIC_VERSION",
    "build_anthropic_request",
    "build_anthropic_transport",
    "parse_anthropic_response",
]
