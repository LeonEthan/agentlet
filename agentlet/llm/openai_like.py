"""OpenAI-compatible model adapter."""

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


@runtime_checkable
class OpenAILikeTransport(Protocol):
    """Callable transport used by the OpenAI-like adapter."""

    def __call__(self, payload: JSONObject) -> JSONObject:
        """Send one provider request and return the raw provider response."""


@dataclass(frozen=True, slots=True)
class OpenAILikeModelClient(ModelClient):
    """Concrete adapter for Chat Completions-style providers."""

    model: str
    transport: OpenAILikeTransport
    request_defaults: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        object.__setattr__(
            self,
            "request_defaults",
            deep_copy_json_object(self.request_defaults),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = build_openai_like_request(
            request,
            model=self.model,
            request_defaults=self.request_defaults,
        )
        response_payload = self.transport(payload)
        return parse_openai_like_response(response_payload)


def build_openai_like_transport(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float = 60.0,
) -> OpenAILikeTransport:
    """Build a JSON-over-HTTP transport for OpenAI-like chat completions."""

    if not base_url.strip():
        raise ValueError("base_url must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
                f"openai-like request failed with HTTP {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"openai-like request failed: {exc.reason}"
            ) from exc

        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "openai-like response body was not valid JSON"
            ) from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("openai-like response payload must be a JSON object")
        return response_payload

    return transport


def build_openai_like_request(
    request: ModelRequest,
    *,
    model: str,
    request_defaults: JSONObject | None = None,
) -> JSONObject:
    """Build one OpenAI-like chat completions payload."""

    if not model:
        raise ValueError("model must not be empty")

    payload = deep_copy_json_object(request_defaults or {})
    payload["model"] = model
    payload["messages"] = [
        _build_message_payload(message)
        for message in request.messages
    ]
    if request.tools:
        payload["tools"] = [
            _build_tool_payload(tool)
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = _build_tool_choice_payload(request.tool_choice)
    return payload


def parse_openai_like_response(payload: JSONObject) -> ModelResponse:
    """Normalize one OpenAI-like response into the shared schema."""

    choices_payload = payload.get("choices")
    if not isinstance(choices_payload, list) or not choices_payload:
        raise ValueError("provider response must include at least one choice")

    choice_payload = _require_mapping(choices_payload[0], "choices[0]")
    message_payload = _require_mapping(
        choice_payload.get("message"),
        "choices[0].message",
    )

    finish_reason = choice_payload.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ValueError("choices[0].finish_reason must be a non-empty string")

    response_metadata: JSONObject = {}
    if payload.get("id") is not None:
        response_metadata["response_id"] = str(payload["id"])
    if payload.get("model") is not None:
        response_metadata["model"] = str(payload["model"])
    if choice_payload.get("index") is not None:
        response_metadata["choice_index"] = int(choice_payload["index"])

    return ModelResponse(
        message=_parse_assistant_message(message_payload, response_metadata),
        finish_reason=finish_reason,
        usage=_parse_usage(payload.get("usage")),
        metadata=response_metadata,
    )


def _build_message_payload(message: Message) -> JSONObject:
    payload: JSONObject = {"role": message.role}
    if message.name is not None:
        payload["name"] = message.name

    if message.role == "assistant":
        payload["content"] = message.content if (message.content or not message.tool_calls) else None
        if message.tool_calls:
            payload["tool_calls"] = [
                _build_tool_call_payload(tool_call)
                for tool_call in message.tool_calls
            ]
        return payload

    payload["content"] = message.content
    if message.role == "tool":
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _build_tool_payload(tool: ModelToolDefinition) -> JSONObject:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": deep_copy_json_object(tool.input_schema),
        },
    }


def _build_tool_choice_payload(tool_choice: ToolChoice) -> str | JSONObject:
    if tool_choice.mode in {"auto", "none", "required"}:
        return tool_choice.mode
    return {
        "type": "function",
        "function": {"name": tool_choice.tool_name or ""},
    }


def _build_tool_call_payload(tool_call: ToolCall) -> JSONObject:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(
                tool_call.arguments,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    }


def _parse_assistant_message(
    payload: JSONObject,
    response_metadata: JSONObject,
) -> Message:
    role = payload.get("role", "assistant")
    if role != "assistant":
        raise ValueError("provider response message role must be 'assistant'")

    message_metadata: JSONObject = {}
    if payload.get("refusal") is not None:
        message_metadata["refusal"] = str(payload["refusal"])
    if response_metadata.get("choice_index") is not None:
        message_metadata["choice_index"] = int(response_metadata["choice_index"])

    return Message(
        role="assistant",
        content=_normalize_text_content(payload.get("content")),
        tool_calls=_parse_tool_calls(payload, choice_index=message_metadata.get("choice_index")),
        metadata=message_metadata,
    )


def _parse_tool_calls(
    payload: JSONObject,
    *,
    choice_index: int | None,
) -> tuple[ToolCall, ...]:
    tool_calls_payload = payload.get("tool_calls")
    if tool_calls_payload is not None:
        if not isinstance(tool_calls_payload, list):
            raise ValueError("assistant tool_calls must be a list")
        return tuple(
            _parse_tool_call(call_payload, index=index, choice_index=choice_index)
            for index, call_payload in enumerate(tool_calls_payload)
        )

    legacy_function_call = payload.get("function_call")
    if legacy_function_call is None:
        return ()
    return (
        _parse_legacy_function_call(
            legacy_function_call,
            choice_index=choice_index,
        ),
    )


def _parse_tool_call(
    payload: object,
    *,
    index: int,
    choice_index: int | None,
) -> ToolCall:
    tool_call_payload = _require_mapping(payload, f"tool_calls[{index}]")
    function_payload = _require_mapping(
        tool_call_payload.get("function"),
        f"tool_calls[{index}].function",
    )

    tool_call_id = tool_call_payload.get("id")
    metadata: JSONObject = {}
    if tool_call_payload.get("type") is not None:
        metadata["provider_type"] = str(tool_call_payload["type"])
    if tool_call_id is None:
        tool_call_id = _synthetic_tool_call_id(choice_index, index)
        metadata["synthetic_id"] = True

    tool_name = function_payload.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(f"tool_calls[{index}].function.name must be a non-empty string")

    return ToolCall(
        id=str(tool_call_id),
        name=tool_name,
        arguments=_normalize_tool_arguments(function_payload.get("arguments")),
        metadata=metadata,
    )


def _parse_legacy_function_call(
    payload: object,
    *,
    choice_index: int | None,
) -> ToolCall:
    function_payload = _require_mapping(payload, "message.function_call")
    tool_name = function_payload.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("message.function_call.name must be a non-empty string")

    return ToolCall(
        id=_synthetic_tool_call_id(choice_index, 0),
        name=tool_name,
        arguments=_normalize_tool_arguments(function_payload.get("arguments")),
        metadata={
            "provider_type": "function",
            "legacy_function_call": True,
            "synthetic_id": True,
        },
    )


def _normalize_tool_arguments(arguments_payload: object) -> JSONObject:
    if arguments_payload is None:
        return {}
    if isinstance(arguments_payload, dict):
        return deep_copy_json_object(arguments_payload)
    if not isinstance(arguments_payload, str):
        raise ValueError("tool call arguments must be a JSON object or JSON string")
    if not arguments_payload.strip():
        return {}

    try:
        parsed_arguments = json.loads(arguments_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("tool call arguments must be valid JSON") from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError("tool call arguments must decode to a JSON object")
    return deep_copy_json_object(parsed_arguments)


def _normalize_text_content(content_payload: object) -> str:
    if content_payload is None:
        return ""
    if isinstance(content_payload, str):
        return content_payload
    if not isinstance(content_payload, list):
        raise ValueError("assistant content must be a string, list, or null")

    parts: list[str] = []
    for index, part_payload in enumerate(content_payload):
        if isinstance(part_payload, str):
            parts.append(part_payload)
            continue
        part = _require_mapping(part_payload, f"content[{index}]")
        part_type = part.get("type")
        if part_type not in {"text", "input_text", "output_text"}:
            raise ValueError(f"unsupported assistant content part type: {part_type}")
        parts.append(_extract_text_part(part, index=index))
    return "".join(parts)


def _extract_text_part(part: JSONObject, *, index: int) -> str:
    text_payload = part.get("text", part.get("value"))
    if isinstance(text_payload, str):
        return text_payload
    if isinstance(text_payload, dict):
        text_value = text_payload.get("value")
        if isinstance(text_value, str):
            return text_value
    raise ValueError(f"content[{index}] text part must include a string value")


def _parse_usage(payload: object) -> TokenUsage | None:
    if payload is None:
        return None
    usage_payload = _require_mapping(payload, "usage")
    return TokenUsage(
        input_tokens=int(usage_payload.get("prompt_tokens", usage_payload.get("input_tokens", 0))),
        output_tokens=int(
            usage_payload.get("completion_tokens", usage_payload.get("output_tokens", 0))
        ),
        total_tokens=int(usage_payload.get("total_tokens", 0)),
    )


def _require_mapping(payload: object, path: str) -> JSONObject:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping")
    return deep_copy_json_object(payload)


def _synthetic_tool_call_id(choice_index: int | None, tool_call_index: int) -> str:
    choice_suffix = 0 if choice_index is None else choice_index
    return f"call_{choice_suffix}_{tool_call_index}"
