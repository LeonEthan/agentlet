from __future__ import annotations

import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Import after path setup
from agentlet.agent.providers.registry import (  # noqa: E402
    LLMResponse,
    ProviderConfig,
    ProviderStreamEvent,
)
from agentlet.agent.tools.registry import ToolSpec  # noqa: E402


def build_capture_console():
    """Create a deterministic console for tests."""
    output = StringIO()
    return Console(file=output, force_terminal=False, color_system=None, width=100), output


@pytest.fixture
def capture_console():
    """Create a deterministic console for tests."""
    return build_capture_console()


class FakeProvider:
    """Test double for LLMProvider that records seen messages."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        stream_events: list[ProviderStreamEvent] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._stream_events = list(stream_events or [])
        self.seen_messages: list[list[str]] = []

    async def complete(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.seen_messages.append([message.role for message in messages])
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="fake response", finish_reason="stop")

    async def stream_complete(
        self,
        messages: list[Any],
        tools: list[Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.seen_messages.append([message.role for message in messages])
        if self._stream_events:
            for event in self._stream_events:
                yield event
            return
        last_user = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        response_text = f"echo: {last_user}"
        yield ProviderStreamEvent(kind="content_delta", text=response_text[:5])
        yield ProviderStreamEvent(kind="content_delta", text=response_text[5:])
        yield ProviderStreamEvent(
            kind="response_complete",
            response=LLMResponse(content=response_text, finish_reason="stop"),
        )


class FakeProviderRegistry:
    """Test double for ProviderRegistry that captures config and returns a fake provider."""

    def __init__(
        self,
        capture_config: list[ProviderConfig] | None = None,
        provider: FakeProvider | None = None,
    ) -> None:
        self._capture_config = capture_config
        self._provider = provider or FakeProvider()

    def create(self, config: ProviderConfig) -> FakeProvider:
        if self._capture_config is not None:
            self._capture_config.append(config)
        return self._provider


def make_cli_args(**overrides) -> SimpleNamespace:
    """Build CLI arguments namespace with sensible defaults for testing."""
    defaults = {
        "message": None,
        "print_mode": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@dataclass(frozen=True)
class EchoTool:
    """Test tool that echoes back the input text."""

    spec: ToolSpec = ToolSpec(
        name="echo",
        description="Return the same text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def execute(self, arguments: dict[str, str]) -> str:
        return arguments["text"]
