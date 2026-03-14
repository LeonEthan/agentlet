from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

from agentlet.agent.agent_loop import TurnEvent
from agentlet.agent.context import Message, ToolCall, ToolResult
from agentlet.agent.providers.registry import DEFAULT_MODEL, ProviderRegistryError
from agentlet.cli.main import _resolve_tool_policy, build_parser
from agentlet.cli.chat_app import ChatCLIError, run_chat_command, _resolve_chat_mode, _settings_from_args
from agentlet.cli.commands import CommandError, command_help_lines, parse_command, summarize_history
from agentlet.cli.presenter import ChatPresenter
from agentlet.settings import AgentletSettings
from conftest import EchoTool, build_capture_console, make_cli_args


def test_resolve_chat_mode_prefers_interactive_tty_without_message() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
    )

    message, interactive = _resolve_chat_mode(args, stdin=StringIO(""), stdin_isatty=True)

    assert message is None
    assert interactive is True


def test_resolve_chat_mode_reads_non_tty_stdin_as_one_shot() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
    )

    message, interactive = _resolve_chat_mode(
        args, stdin=StringIO("hello from stdin"), stdin_isatty=False
    )

    assert interactive is False
    assert message == "hello from stdin"


def test_resolve_chat_mode_rejects_print_mode_on_interactive_tty_without_message() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=True,
    )

    with pytest.raises(
        ChatCLIError,
        match="--print requires a message argument or redirected stdin",
    ):
        _resolve_chat_mode(args, stdin=StringIO(""), stdin_isatty=True)


def test_build_parser_rejects_removed_flags() -> None:
    parser = build_parser(AgentletSettings(provider="openai", model="gpt-5.4"))

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--continue"])

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--session", "session-123"])

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--new-session"])

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--api-key", "test-key"])

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--api-base", "http://localhost:4000/v1"])

    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--api-key", "test-key"])

    with pytest.raises(SystemExit):
        parser.parse_args(["init", "--api-base", "http://localhost:4000/v1"])


def test_parse_command_rejects_arguments() -> None:
    with pytest.raises(CommandError, match="does not take arguments"):
        parse_command("/help extra")


def test_summarize_history_uses_final_assistant_reply_after_tool_turn() -> None:
    history = [
        Message(role="user", content="look this up"),
        Message(
            role="assistant",
            content=None,
            tool_calls=(ToolCall(id="call-1", name="search", arguments_json="{}"),),
        ),
        Message(role="tool", content="result", name="search", tool_call_id="call-1"),
        Message(role="assistant", content="final answer"),
    ]

    assert summarize_history(history) == [("look this up", "final answer")]


def test_presenter_renders_tool_activity_lines() -> None:
    console, output = build_capture_console()
    presenter = ChatPresenter(console)
    tool_call = ToolCall(id="call-1", name="echo", arguments_json='{"text":"hi"}')

    presenter.handle_event(TurnEvent(kind="tool_started", tool_call=tool_call))
    presenter.handle_event(
        TurnEvent(
            kind="tool_completed",
            tool_result=ToolResult(
                tool_call_id="call-1",
                name="echo",
                content="hi",
            ),
        )
    )

    rendered = output.getvalue()
    assert "⠋" in rendered  # Braille spinner for pending
    assert "✓" in rendered  # Checkmark for success


def test_presenter_summarizes_tool_arguments_and_results() -> None:
    console, output = build_capture_console()
    presenter = ChatPresenter(console)
    tool_call = ToolCall(
        id="call-1",
        name="web_search",
        arguments_json='{"query":"agentlet phase 3","max_results":5}',
    )
    tool_result = ToolResult(
        tool_call_id="call-1",
        name="web_search",
        content='{"ok":true,"tool":"web_search","query":"agentlet phase 3","results":[{"url":"https://example.com"}]}',
    )

    presenter.handle_event(TurnEvent(kind="tool_started", tool_call=tool_call))
    presenter.handle_event(TurnEvent(kind="tool_completed", tool_result=tool_result))

    rendered = output.getvalue()
    assert "query='agentlet phase 3'" in rendered
    assert "results=1" in rendered
    assert '{"query"' not in rendered


def test_presenter_keeps_long_single_token_values_identifiable() -> None:
    console, output = build_capture_console()
    presenter = ChatPresenter(console)
    long_path = (
        "/Users/cuizhengliang/Documents/vibe-coding/agentlet/"
        "src/agentlet/cli/presenter.py"
    )
    long_url = (
        "https://example.com/really/long/path/to/a/resource/with/query"
        "?token=abcdef1234567890#section-anchor"
    )

    presenter.handle_event(
        TurnEvent(
            kind="tool_started",
            tool_call=ToolCall(
                id="call-1",
                name="read",
                arguments_json=json.dumps({"path": long_path}),
            ),
        )
    )
    presenter.handle_event(
        TurnEvent(
            kind="tool_completed",
            tool_result=ToolResult(
                tool_call_id="call-1",
                name="web_fetch",
                content=json.dumps(
                    {
                        "ok": True,
                        "tool": "web_fetch",
                        "final_url": long_url,
                        "truncated": True,
                    }
                ),
            ),
        )
    )

    rendered = output.getvalue()
    assert "path='/Users" in rendered
    assert "presenter.py'" in rendered
    assert "https://example.com" in rendered
    assert "section-anchor" in rendered
    assert "path=..." not in rendered
    assert "web_fetch ... truncated=True" not in rendered


def test_presenter_formats_help_commands_consistently() -> None:
    console, output = build_capture_console()

    ChatPresenter(console).show_help(command_help_lines())

    rendered = output.getvalue()
    assert "  /history     show recent turn summaries" in rendered
    assert "  Enter submits, Alt+Enter inserts a newline." in rendered


def test_run_chat_command_status_shows_enabled_tools(tmp_path) -> None:
    from agentlet.agent.tools.registry import ToolRegistry
    from conftest import FakeProviderRegistry

    class FakePromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)

        def prompt(self, prompt_text: str | None = None) -> str:
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

    console, output = build_capture_console()

    exit_code = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(),
        tool_registry=ToolRegistry([EchoTool()]),
        prompt_session=FakePromptSession(["/status", "/exit"]),
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "Tools:" in rendered
    assert "echo" in rendered


def _make_openai_settings(**overrides) -> AgentletSettings:
    """Build AgentletSettings with OpenAI defaults for testing."""
    defaults = {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "sk-openai-key",
        "api_base": "https://api.openai.com/v1",
    }
    defaults.update(overrides)
    return AgentletSettings(**defaults)


def test_settings_from_args_inherits_api_credentials_when_provider_matches() -> None:
    """When provider matches stored settings, api_key/api_base are inherited."""
    fallback = _make_openai_settings()
    args = make_cli_args()  # provider=None, uses fallback

    result = _settings_from_args(args, fallback=fallback)

    assert result.provider == "openai"
    assert result.api_key == "sk-openai-key"
    assert result.api_base == "https://api.openai.com/v1"


def test_settings_from_args_clears_api_credentials_when_provider_overridden() -> None:
    """When provider is overridden via CLI, api_key/api_base are cleared to None."""
    fallback = _make_openai_settings()
    args = make_cli_args(provider="anthropic", model="claude-3-sonnet")

    result = _settings_from_args(args, fallback=fallback)

    assert result.provider == "anthropic"
    assert result.model == "claude-3-sonnet"
    assert result.api_key is None  # Cleared, not inherited
    assert result.api_base is None  # Cleared, not inherited


def test_resolve_tool_policy_uses_stored_settings_without_deny_flags() -> None:
    args = make_cli_args(deny_write=False, deny_bash=False, deny_network=False)

    result = _resolve_tool_policy(
        AgentletSettings(allow_write=False, allow_bash=True, allow_network=False),
        args,
    )

    assert result.allow_write is False
    assert result.allow_bash is True
    assert result.allow_network is False


def test_resolve_tool_policy_deny_flags_override_stored_settings() -> None:
    args = make_cli_args(deny_write=True, deny_bash=True, deny_network=True)

    result = _resolve_tool_policy(
        AgentletSettings(allow_write=True, allow_bash=True, allow_network=True),
        args,
    )

    assert result.allow_write is False
    assert result.allow_bash is False
    assert result.allow_network is False
