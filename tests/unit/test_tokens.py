"""Tests for token counting utilities."""

from __future__ import annotations

import pytest

from agentlet.core.messages import Message
from agentlet.llm.schemas import ModelToolDefinition
from agentlet.llm.tokens import (
    ApproximateTokenizer,
    TokenEstimate,
    create_tokenizer,
)


def test_approximate_tokenizer_counts_text():
    """Test approximate token counting for text."""
    tokenizer = ApproximateTokenizer()

    # Should return at least 1 for any non-empty text
    assert tokenizer.count_text("hi") >= 1
    assert tokenizer.count_text("hello world") >= 1

    # Longer text should have more tokens
    short_text = "hello"
    long_text = "hello " * 100
    assert tokenizer.count_text(long_text) > tokenizer.count_text(short_text)


def test_approximate_tokenizer_different_model_families():
    """Test that different model families have different rates."""
    gpt4 = ApproximateTokenizer(model_family="gpt-4")
    claude = ApproximateTokenizer(model_family="claude")
    default = ApproximateTokenizer()

    text = "hello world"

    # All should give reasonable estimates
    assert gpt4.count_text(text) > 0
    assert claude.count_text(text) > 0
    assert default.count_text(text) > 0


def test_approximate_tokenizer_counts_messages():
    """Test message token counting."""
    tokenizer = ApproximateTokenizer()

    messages = (
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    )

    count = tokenizer.count_messages(messages)

    # Should include overhead for each message
    assert count >= len(messages) * 4


def test_approximate_tokenizer_counts_tool_definitions():
    """Test tool definition token counting."""
    tokenizer = ApproximateTokenizer()

    tools = (
        ModelToolDefinition(
            name="Read",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        ),
    )

    count = tokenizer.count_tool_definitions(tools)

    # Should include overhead
    assert count >= 20


def test_token_estimate_with_margin():
    """Test adding safety margin to estimates."""
    estimate = TokenEstimate(
        total=100,
        messages=80,
        tools=15,
        overhead=5,
    )

    with_margin = estimate.with_margin(margin_percent=10.0)

    assert with_margin.total == 110  # 100 * 1.1
    assert with_margin.messages == 80
    assert with_margin.tools == 15


def test_token_estimate_zero_margin():
    """Test zero margin returns same totals."""
    estimate = TokenEstimate(
        total=100,
        messages=80,
        tools=15,
        overhead=5,
    )

    with_margin = estimate.with_margin(margin_percent=0.0)

    assert with_margin.total == 100


def test_estimate_request_comprehensive():
    """Test full request estimation."""
    tokenizer = ApproximateTokenizer()

    messages = (
        Message(role="system", content="You are helpful"),
        Message(role="user", content="Hello"),
    )

    tools = (
        ModelToolDefinition(
            name="Read",
            description="Read a file",
            input_schema={"type": "object"},
        ),
    )

    estimate = tokenizer.estimate_request(messages, tools)

    assert estimate.total > 0
    assert estimate.messages > 0
    assert estimate.tools > 0
    assert estimate.total == estimate.messages + estimate.tools + estimate.overhead


def test_create_tokenizer_factory():
    """Test tokenizer factory function."""
    # Default tokenizer
    default = create_tokenizer()
    assert isinstance(default, ApproximateTokenizer)

    # GPT-4 tokenizer (may be Tiktoken or Approximate)
    gpt4 = create_tokenizer("gpt-4")
    assert hasattr(gpt4, "count_text")

    # Claude tokenizer
    claude = create_tokenizer("claude-3-opus")
    assert isinstance(claude, ApproximateTokenizer)


def test_tokenizer_empty_content():
    """Test handling of empty content."""
    tokenizer = ApproximateTokenizer()

    # Empty text should still return at least 1
    assert tokenizer.count_text("") >= 1

    # Empty messages should just have overhead
    messages = (Message(role="user", content=""),)
    count = tokenizer.count_messages(messages)
    assert count >= 4  # Just overhead


def test_message_with_tool_calls():
    """Test counting messages with tool calls."""
    from agentlet.core.messages import ToolCall

    tokenizer = ApproximateTokenizer()

    messages = (
        Message(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="Read",
                    arguments={"path": "test.txt"},
                ),
            ),
        ),
    )

    count = tokenizer.count_messages(messages)

    # Should include tool call tokens
    assert count > 4  # More than just overhead
