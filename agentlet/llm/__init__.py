"""Model adapter package."""

from agentlet.llm.openai_like import (
    OpenAILikeModelClient,
    OpenAILikeTransport,
    build_openai_like_request,
    parse_openai_like_response,
)

__all__ = [
    "OpenAILikeModelClient",
    "OpenAILikeTransport",
    "build_openai_like_request",
    "parse_openai_like_response",
]
