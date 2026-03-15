from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

from agentlet.agent.agent_loop import TurnEvent
from agentlet.agent.tools.policy import DEFAULT_MAX_HTML_EXTRACT_BYTES
from agentlet.agent.context import Message, ToolCall, ToolResult
from agentlet.agent.providers.registry import (
    DEFAULT_MODEL,
    LLMResponse,
    ProviderStreamEvent,
    ProviderRegistryError,
)
from agentlet.cli.main import _build_tool_runtime, _resolve_tool_policy, build_parser
import agentlet.cli.chat_app as chat_app_module
from agentlet.cli.chat_app import (
    ChatCLIError,
    _create_agent_loop,
    _resolve_chat_mode,
    _settings_from_args,
    run_chat_command,
)
from agentlet.cli.commands import (
    CommandError,
    TurnSummary,
    command_help_lines,
    parse_command,
    summarize_history,
)
from agentlet.cli.presenter import ChatPresenter
from agentlet.settings import AgentletSettings
from conftest import (
    EchoTool,
    FakeProvider,
    FakeProviderRegistry,
    build_capture_console,
    make_cli_args,
)


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

    assert summarize_history(history) == [
        TurnSummary(1, "look this up", "final answer")
    ]


def test_summarize_history_preserves_original_turn_numbers_when_trimmed() -> None:
    history = [
        Message(role="user", content="first"),
        Message(role="assistant", content="one"),
        Message(role="user", content="second"),
        Message(role="assistant", content="two"),
        Message(role="user", content="third"),
        Message(role="assistant", content="three"),
    ]

    assert summarize_history(history, limit=2) == [
        TurnSummary(2, "second", "two"),
        TurnSummary(3, "third", "three"),
    ]


def test_presenter_show_history_uses_original_turn_numbers() -> None:
    console, output = build_capture_console()

    ChatPresenter(console).show_history(
        [
            TurnSummary(9, "ninth question", "ninth answer"),
            TurnSummary(10, "tenth question", "tenth answer"),
        ]
    )

    rendered = output.getvalue()
    assert "Turn 9" in rendered
    assert "Turn 10" in rendered


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

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            return self.prompt(prompt_text)

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


def test_run_chat_command_interactive_approval_uses_async_prompt(tmp_path) -> None:
    from agentlet.agent.tools.registry import ToolSpec, ToolRegistry

    class AsyncOnlyPromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)
            self.sync_calls = 0

        def prompt(self, prompt_text: str | None = None) -> str:
            self.sync_calls += 1
            raise AssertionError("interactive approvals should use prompt_async()")

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

    class FakeWriteTool:
        spec = ToolSpec(
            name="write",
            description="fake write",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        def __init__(self) -> None:
            self.seen_arguments: list[dict[str, str]] = []

        async def execute(self, arguments: dict[str, str]) -> str:
            self.seen_arguments.append(arguments)
            return "ok"

    class ScriptedStreamingProvider:
        def __init__(self) -> None:
            self._calls = 0

        async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
            raise AssertionError("interactive mode should stream responses")

        async def stream_complete(
            self, messages, tools=None, model=None, temperature=None, max_tokens=None
        ):
            if self._calls == 0:
                self._calls += 1
                yield ProviderStreamEvent(
                    kind="response_complete",
                    response=LLMResponse(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                id="call-1",
                                name="write",
                                arguments_json='{"path":"notes.txt"}',
                            ),
                        ),
                    ),
                )
                return

            self._calls += 1
            yield ProviderStreamEvent(kind="content_delta", text="final ")
            yield ProviderStreamEvent(kind="content_delta", text="answer")
            yield ProviderStreamEvent(
                kind="response_complete",
                response=LLMResponse(content="final answer", finish_reason="stop"),
            )

    console, output = build_capture_console()
    prompt = AsyncOnlyPromptSession(["write the file", "y", "/exit"])
    tool = FakeWriteTool()

    exit_code = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(provider=ScriptedStreamingProvider()),
        tool_registry=ToolRegistry([tool]),
        prompt_session=prompt,
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "final answer" in rendered
    assert "Turn failed" not in rendered
    assert prompt.sync_calls == 0
    assert tool.seen_arguments == [{"path": "notes.txt"}]


def test_run_chat_command_interactive_approval_eof_closes_session(tmp_path) -> None:
    from agentlet.agent.tools.registry import ToolSpec, ToolRegistry

    class FakePromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)

        def prompt(self, prompt_text: str | None = None) -> str:
            raise AssertionError("interactive approvals should use prompt_async()")

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

    class FakeWriteTool:
        spec = ToolSpec(
            name="write",
            description="fake write",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        def __init__(self) -> None:
            self.seen_arguments: list[dict[str, str]] = []

        async def execute(self, arguments: dict[str, str]) -> str:
            self.seen_arguments.append(arguments)
            return "ok"

    class ScriptedStreamingProvider:
        def __init__(self) -> None:
            self._calls = 0

        async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
            raise AssertionError("interactive mode should stream responses")

        async def stream_complete(
            self, messages, tools=None, model=None, temperature=None, max_tokens=None
        ):
            if self._calls == 0:
                self._calls += 1
                yield ProviderStreamEvent(
                    kind="response_complete",
                    response=LLMResponse(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                id="call-1",
                                name="write",
                                arguments_json='{"path":"notes.txt"}',
                            ),
                        ),
                    ),
                )
                return

            raise AssertionError("approval EOF should stop the session before another provider turn")

    console, output = build_capture_console()
    prompt = FakePromptSession(["write the file"])
    tool = FakeWriteTool()

    exit_code = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(provider=ScriptedStreamingProvider()),
        tool_registry=ToolRegistry([tool]),
        prompt_session=prompt,
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "Session closed." in rendered
    assert "Turn failed" not in rendered
    assert tool.seen_arguments == []


def test_run_chat_command_interactive_ignores_stale_tool_calls_on_final_response(
    tmp_path,
) -> None:
    from agentlet.agent.tools.registry import ToolSpec, ToolRegistry

    class RecordingPromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)
            self.seen_prompts: list[str | None] = []

        def prompt(self, prompt_text: str | None = None) -> str:
            raise AssertionError("interactive mode should use prompt_async()")

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            self.seen_prompts.append(prompt_text)
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

    class FakeWriteTool:
        spec = ToolSpec(
            name="write",
            description="fake write",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        def __init__(self) -> None:
            self.seen_arguments: list[dict[str, str]] = []

        async def execute(self, arguments: dict[str, str]) -> str:
            self.seen_arguments.append(arguments)
            return "ok"

    class StaleToolCallStreamingProvider:
        async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
            raise AssertionError("interactive mode should stream responses")

        async def stream_complete(
            self, messages, tools=None, model=None, temperature=None, max_tokens=None
        ):
            yield ProviderStreamEvent(kind="content_delta", text="final ")
            yield ProviderStreamEvent(kind="content_delta", text="answer")
            yield ProviderStreamEvent(
                kind="response_complete",
                response=LLMResponse(
                    content="final answer",
                    tool_calls=(
                        ToolCall(
                            id="call-1",
                            name="write",
                            arguments_json='{"path":"notes.txt"}',
                        ),
                    ),
                    finish_reason="stop",
                ),
            )

    console, output = build_capture_console()
    prompt = RecordingPromptSession(["show the weather", "/exit"])
    tool = FakeWriteTool()

    exit_code = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(provider=StaleToolCallStreamingProvider()),
        tool_registry=ToolRegistry([tool]),
        prompt_session=prompt,
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "final answer" in rendered
    assert tool.seen_arguments == []
    assert all(
        prompt_text is None or "Approve write notes.txt?" not in prompt_text
        for prompt_text in prompt.seen_prompts
    )


def test_run_chat_command_provider_eof_still_surfaces_as_turn_failure(tmp_path) -> None:
    class FakePromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)

        def prompt(self, prompt_text: str | None = None) -> str:
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            return self.prompt(prompt_text)

    class EOFStreamingProvider:
        async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
            raise AssertionError("interactive mode should stream responses")

        async def stream_complete(
            self, messages, tools=None, model=None, temperature=None, max_tokens=None
        ):
            raise EOFError("provider stream ended unexpectedly")
            yield

    console, output = build_capture_console()

    exit_code = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(provider=EOFStreamingProvider()),
        prompt_session=FakePromptSession(["hello", "/exit"]),
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "Turn failed" in rendered
    assert "provider stream ended unexpectedly" in rendered
    assert "Session closed." in rendered


def test_run_chat_command_closes_approval_handler_in_one_shot_and_interactive(tmp_path, monkeypatch) -> None:
    closed_handlers: list[TrackingApprovalHandler] = []

    class TrackingApprovalHandler:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.close_calls = 0
            closed_handlers.append(self)

        async def approve(self, request) -> bool:
            return True

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(chat_app_module, "InteractiveApprovalHandler", TrackingApprovalHandler)

    one_shot_exit = run_chat_command(
        make_cli_args(message="hello"),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(),
        stdin_isatty=True,
    )

    class FakePromptSession:
        def __init__(self, inputs: list[str]) -> None:
            self._inputs = list(inputs)

        def prompt(self, prompt_text: str | None = None) -> str:
            if not self._inputs:
                raise EOFError
            return self._inputs.pop(0)

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            return self.prompt(prompt_text)

    console, _ = build_capture_console()
    interactive_exit = run_chat_command(
        make_cli_args(),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=FakeProviderRegistry(),
        prompt_session=FakePromptSession(["/exit"]),
        console=console,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    assert one_shot_exit == 0
    assert interactive_exit == 0
    assert len(closed_handlers) == 2
    assert [handler.close_calls for handler in closed_handlers] == [1, 1]
    assert closed_handlers[0].kwargs["stdin"] is not None
    assert closed_handlers[0].kwargs["stdout"] is not None
    assert closed_handlers[1].kwargs["prompt_input"] is not None


def test_run_chat_command_routes_one_shot_approval_prompts_to_stderr() -> None:
    from io import StringIO

    from agentlet.agent.tools.registry import ToolSpec, ToolRegistry

    class FakeTTY(StringIO):
        def __init__(self, value: str = "", *, is_tty: bool) -> None:
            super().__init__(value)
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    class FakeWriteTool:
        spec = ToolSpec(
            name="write",
            description="fake write",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        def __init__(self) -> None:
            self.seen_arguments: list[dict[str, str]] = []

        async def execute(self, arguments: dict[str, str]) -> str:
            self.seen_arguments.append(arguments)
            return "ok"

    provider = FakeProvider(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="write",
                        arguments_json='{"path":"notes.txt"}',
                    ),
                ),
            ),
            LLMResponse(content="final answer", finish_reason="stop"),
        ]
    )
    tool = FakeWriteTool()
    stdout = StringIO()
    stderr = FakeTTY("", is_tty=True)

    exit_code = run_chat_command(
        make_cli_args(message="write the file"),
        settings=AgentletSettings(provider="openai", model="gpt-4"),
        stdin=FakeTTY("y\n", is_tty=True),
        stdout=stdout,
        stderr=stderr,
        provider_registry=FakeProviderRegistry(provider=provider),
        tool_registry=ToolRegistry([tool]),
    )

    assert exit_code == 0
    assert stdout.getvalue() == "final answer\n"
    assert "Approve write notes.txt?" in stderr.getvalue()
    assert tool.seen_arguments == [{"path": "notes.txt"}]


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


def test_settings_from_args_carries_runtime_safety_overrides() -> None:
    fallback = _make_openai_settings(
        max_iterations=8,
        max_html_extract_bytes=2_000_000,
    )
    args = make_cli_args(
        max_iterations=12,
        max_html_extract_bytes=750_000,
    )

    result = _settings_from_args(args, fallback=fallback)

    assert result.max_iterations == 12
    assert result.max_html_extract_bytes == 750_000


def test_settings_from_args_clears_api_credentials_when_provider_overridden() -> None:
    """When provider is overridden via CLI, api_key/api_base are cleared to None."""
    fallback = _make_openai_settings()
    args = make_cli_args(provider="anthropic", model="claude-3-sonnet")

    result = _settings_from_args(args, fallback=fallback)

    assert result.provider == "anthropic"
    assert result.model == "claude-3-sonnet"
    assert result.api_key is None  # Cleared, not inherited
    assert result.api_base is None  # Cleared, not inherited


def test_build_tool_runtime_uses_effective_html_extract_budget(tmp_path) -> None:
    runtime = _build_tool_runtime(
        make_cli_args(max_html_extract_bytes=750_000),
        settings=AgentletSettings(max_html_extract_bytes=2_000_000),
        cwd=tmp_path,
    )

    assert runtime.cwd == tmp_path
    assert runtime.max_html_extract_bytes == 750_000


def test_build_tool_runtime_uses_default_html_extract_budget_when_unset(tmp_path) -> None:
    runtime = _build_tool_runtime(
        make_cli_args(),
        settings=AgentletSettings(),
        cwd=tmp_path,
    )

    assert runtime.max_html_extract_bytes == DEFAULT_MAX_HTML_EXTRACT_BYTES


def test_create_agent_loop_uses_settings_max_iterations() -> None:
    loop = _create_agent_loop(
        settings=AgentletSettings(
            provider="openai",
            model="gpt-4",
            max_iterations=11,
        ),
        provider_registry=FakeProviderRegistry(),
    )

    assert loop.max_iterations == 11


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
