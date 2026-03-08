"""Official OpenAI provider helpers built on the OpenAI-compatible adapter."""

from __future__ import annotations

from agentlet.llm.openai_like import (
    OpenAILikeModelClient,
    OpenAILikeTransport,
    build_openai_like_request,
    build_openai_like_transport,
    parse_openai_like_response,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

OpenAIModelClient = OpenAILikeModelClient
OpenAITransport = OpenAILikeTransport


def build_openai_transport(
    *,
    api_key: str,
    timeout_seconds: float = 60.0,
) -> OpenAITransport:
    """Build the default transport for the official OpenAI Chat Completions API."""

    return build_openai_like_transport(
        base_url=DEFAULT_OPENAI_BASE_URL,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "OpenAIModelClient",
    "OpenAITransport",
    "build_openai_like_request",
    "build_openai_transport",
    "parse_openai_like_response",
]
