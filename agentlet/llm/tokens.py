"""Token counting and estimation utilities for context management."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentlet.core.messages import Message
from agentlet.core.types import JSONObject
from agentlet.llm.schemas import ModelToolDefinition

# Approximate tokens per character for different model families
TOKENS_PER_CHAR = {
    "gpt-4": 0.25,
    "gpt-4o": 0.25,
    "gpt-3.5": 0.3,
    "claude": 0.27,
    "default": 0.3,
}

# Approximate tokens for message overhead (role, formatting)
MESSAGE_OVERHEAD_TOKENS = 4

# Approximate tokens for tool definition overhead
TOOL_DEFINITION_OVERHEAD = 20


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting implementations."""

    def count_text(self, text: str) -> int:
        """Count tokens in plain text."""

    def count_messages(self, messages: tuple[Message, ...]) -> int:
        """Count tokens in a sequence of messages."""

    def count_tool_definitions(self, tools: tuple[ModelToolDefinition, ...]) -> int:
        """Count tokens in tool definitions."""


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Token count estimate with breakdown."""

    total: int
    messages: int
    tools: int
    overhead: int

    def with_margin(self, margin_percent: float = 10.0) -> "TokenEstimate":
        """Add safety margin to account for estimation error."""
        multiplier = 1.0 + (margin_percent / 100.0)
        return TokenEstimate(
            total=int(self.total * multiplier),
            messages=self.messages,
            tools=self.tools,
            overhead=self.overhead,
        )


class ApproximateTokenizer:
    """Approximate tokenizer using character-based estimation.

    This provides fast, reasonably accurate estimates without requiring
    external dependencies like tiktoken.

    Accuracy: Typically within 10-20% of actual token counts.
    """

    def __init__(self, model_family: str = "default") -> None:
        """Initialize with model family for appropriate estimates.

        Args:
            model_family: One of "gpt-4", "gpt-4o", "gpt-3.5", "claude", or "default"
        """
        self._tokens_per_char = TOKENS_PER_CHAR.get(model_family, TOKENS_PER_CHAR["default"])

    def count_text(self, text: str) -> int:
        """Count tokens in plain text using character-based estimation."""
        return max(1, int(len(text) * self._tokens_per_char))

    def count_messages(self, messages: tuple[Message, ...]) -> int:
        """Count tokens in a sequence of messages."""
        total = 0
        for message in messages:
            # Content tokens
            content = message.content or ""
            total += self.count_text(content)

            # Tool calls
            for tool_call in message.tool_calls:
                total += self.count_text(tool_call.name)
                total += self._count_json(tool_call.arguments)

            # Message overhead (role markers, formatting)
            total += MESSAGE_OVERHEAD_TOKENS

        return total

    def count_tool_definitions(self, tools: tuple[ModelToolDefinition, ...]) -> int:
        """Count tokens in tool definitions."""
        total = 0
        for tool in tools:
            total += self.count_text(tool.name)
            total += self.count_text(tool.description)
            total += self._count_json(tool.input_schema)
            total += TOOL_DEFINITION_OVERHEAD
        return total

    def _count_json(self, value: JSONObject) -> int:
        """Approximate token count for JSON structure."""
        text = str(value)
        return self.count_text(text)

    def estimate_request(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ModelToolDefinition, ...] = (),
    ) -> TokenEstimate:
        """Get detailed token estimate for a complete request."""
        message_tokens = self.count_messages(messages)
        tool_tokens = self.count_tool_definitions(tools)

        # Additional overhead for request structure
        overhead = 3 if tools else 0

        return TokenEstimate(
            total=message_tokens + tool_tokens + overhead,
            messages=message_tokens,
            tools=tool_tokens,
            overhead=overhead,
        )


class TiktokenTokenizer:
    """Tokenizer using tiktoken for accurate OpenAI model counts.

    Falls back to approximate tokenizer if tiktoken is not available.
    """

    def __init__(self, model_name: str = "gpt-4") -> None:
        """Initialize with model name for tiktoken encoding.

        Args:
            model_name: Model name like "gpt-4", "gpt-3.5-turbo", etc.
        """
        self._model_name = model_name
        self._encoder = self._load_encoder()
        self._fallback = ApproximateTokenizer(model_family="gpt-4")

    def _load_encoder(self):
        """Try to load tiktoken encoder, return None if unavailable."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model(self._model_name)
        except (ImportError, KeyError):
            return None

    def count_text(self, text: str) -> int:
        """Count tokens using tiktoken if available, otherwise approximate."""
        if self._encoder is None:
            return self._fallback.count_text(text)
        return len(self._encoder.encode(text))

    def count_messages(self, messages: tuple[Message, ...]) -> int:
        """Count tokens in messages."""
        if self._encoder is None:
            return self._fallback.count_messages(messages)

        # Use tiktoken's message counting if available
        total = 0
        for message in messages:
            total += MESSAGE_OVERHEAD_TOKENS  # Message overhead
            total += self.count_text(message.content or "")

            # Tool calls
            for tool_call in message.tool_calls:
                total += self.count_text(tool_call.name)
                total += self.count_text(str(tool_call.arguments))

        return total

    def count_tool_definitions(self, tools: tuple[ModelToolDefinition, ...]) -> int:
        """Count tokens in tool definitions."""
        if self._encoder is None:
            return self._fallback.count_tool_definitions(tools)

        total = 0
        for tool in tools:
            total += self.count_text(tool.name)
            total += self.count_text(tool.description)
            total += self.count_text(str(tool.input_schema))
            total += TOOL_DEFINITION_OVERHEAD
        return total

    def estimate_request(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ModelToolDefinition, ...] = (),
    ) -> TokenEstimate:
        """Get detailed token estimate for a complete request."""
        message_tokens = self.count_messages(messages)
        tool_tokens = self.count_tool_definitions(tools)

        # Additional overhead for request structure
        overhead = 3 if tools else 0

        return TokenEstimate(
            total=message_tokens + tool_tokens + overhead,
            messages=message_tokens,
            tools=tool_tokens,
            overhead=overhead,
        )


def create_tokenizer(model_name: str | None = None) -> Tokenizer:
    """Factory to create appropriate tokenizer for model.

    Args:
        model_name: Model name to determine tokenizer type.
            If None, returns ApproximateTokenizer.
            If OpenAI model, tries TiktokenTokenizer.
            Otherwise, returns ApproximateTokenizer with appropriate family.

    Returns:
        Tokenizer instance appropriate for the model.
    """
    if model_name is None:
        return ApproximateTokenizer()

    model_lower = model_name.lower()

    # Try tiktoken for OpenAI models
    if any(x in model_lower for x in ["gpt-4", "gpt-3.5", "gpt-3"]):
        try:
            return TiktokenTokenizer(model_name)
        except Exception:
            pass
        return ApproximateTokenizer(model_family="gpt-4")

    if "claude" in model_lower:
        return ApproximateTokenizer(model_family="claude")

    return ApproximateTokenizer()


__all__ = [
    "ApproximateTokenizer",
    "TiktokenTokenizer",
    "TokenEstimate",
    "Tokenizer",
    "create_tokenizer",
]
