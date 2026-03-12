from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

from agentlet.agent.agent_loop import TurnEvent
from agentlet.agent.context import Message, ToolCall, ToolResult
from agentlet.agent.providers.registry import DEFAULT_MODEL, ProviderRegistryError
from agentlet.cli.chat_app import ChatCLIError, run_chat_command, _resolve_chat_mode
from agentlet.cli.commands import CommandError, parse_command, summarize_history
from agentlet.cli.presenter import ChatPresenter
from conftest import build_capture_console, make_cli_args


def test_resolve_chat_mode_prefers_interactive_tty_without_message() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
        continue_session=False,
        session_id=None,
        new_session=False,
    )

    message, interactive = _resolve_chat_mode(args, stdin=StringIO(""), stdin_isatty=True)

    assert message is None
    assert interactive is True


def test_resolve_chat_mode_rejects_message_with_session_flags() -> None:
    args = SimpleNamespace(
        message="hello",
        print_mode=False,
        continue_session=True,
        session_id=None,
        new_session=False,
    )

    with pytest.raises(ChatCLIError, match="Session flags cannot be combined"):
        _resolve_chat_mode(args, stdin=StringIO(""), stdin_isatty=True)


def test_resolve_chat_mode_reads_non_tty_stdin_as_one_shot() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
        continue_session=False,
        session_id=None,
        new_session=False,
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
        continue_session=False,
        session_id=None,
        new_session=False,
    )

    with pytest.raises(
        ChatCLIError,
        match="--print requires a message argument or redirected stdin",
    ):
        _resolve_chat_mode(args, stdin=StringIO(""), stdin_isatty=True)


def test_resolve_chat_mode_rejects_session_flags_with_non_tty_stdin() -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
        continue_session=True,
        session_id=None,
        new_session=False,
    )

    with pytest.raises(ChatCLIError, match="Session flags require an interactive TTY"):
        _resolve_chat_mode(args, stdin=StringIO("hello from stdin"), stdin_isatty=False)


def test_run_chat_command_reports_missing_latest_session_as_cli_error(tmp_path) -> None:
    args = SimpleNamespace(
        message=None,
        print_mode=False,
        continue_session=True,
        session_id=None,
        new_session=False,
        provider="openai",
        model=DEFAULT_MODEL,
        api_key=None,
        api_base=None,
        temperature=0.0,
        max_tokens=None,
    )

    with pytest.raises(ChatCLIError, match="No latest session metadata found"):
        run_chat_command(
            args,
            stdin=StringIO(""),
            stdout=StringIO(),
            stderr=StringIO(),
            cwd=tmp_path,
            stdin_isatty=True,
        )


def test_run_chat_command_does_not_create_session_when_provider_setup_fails(tmp_path) -> None:
    class BadProviderRegistry:
        def create(self, config):
            raise ProviderRegistryError(f"Unsupported provider: {config.name}")

    with pytest.raises(ProviderRegistryError, match="Unsupported provider: bad"):
        run_chat_command(
            make_cli_args(provider="bad"),
            stdin=StringIO(""),
            stdout=StringIO(),
            stderr=StringIO(),
            provider_registry=BadProviderRegistry(),
            cwd=tmp_path,
            stdin_isatty=True,
        )

    assert not (tmp_path / ".agentlet" / "sessions").exists()


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
    assert "tool start" in rendered
    assert "tool done" in rendered
