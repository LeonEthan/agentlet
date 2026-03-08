from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path

import pytest

from agentlet.core.approvals import ApprovalPolicy
from agentlet.core.interrupts import (
    ApprovalRequest,
    UserQuestionRequest,
    UserQuestionResponse,
)
from apps.cli import TerminalUserIO, main
from agentlet.core.loop import CompletedTurn, InterruptedTurn
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, InterruptOption, TokenUsage
from agentlet.llm.schemas import ModelRequest, ModelResponse
from agentlet.runtime.app import build_runtime_app
from agentlet.tools.interaction.ask_user_question import AskUserQuestionTool
from agentlet.tools.base import ToolDefinition, ToolResult


class FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("no fake model responses remaining")
        return self._responses.pop(0)


class FakeTool:
    def __init__(self, definition: ToolDefinition, result: ToolResult) -> None:
        self.definition = definition
        self.result = result

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return self.result


class FakeRegistry:
    def __init__(self, tools: list[FakeTool]) -> None:
        self._tools = {tool.definition.name: tool for tool in tools}

    def get(self, name: str) -> FakeTool | None:
        return self._tools.get(name)

    def definition(self, name: str) -> ToolDefinition:
        return self._tools[name].definition

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())


class FakeRuntimeApp:
    def __init__(self, outcome: CompletedTurn | InterruptedTurn) -> None:
        self.outcome = outcome
        self.tasks: list[str] = []

    def run_turn(self, *, current_task: str, resume=None):
        assert resume is None
        self.tasks.append(current_task)
        return self.outcome


def test_terminal_user_io_renders_approval_prompt_and_parses_yes() -> None:
    stdin = StringIO("yes\n")
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=stdin, stdout=stdout, stderr=StringIO(), interactive=True)

    response = user_io.request_approval(
        ApprovalRequest(
            request_id="approval_1",
            tool_name="Write",
            approval_category="mutating",
            prompt="Allow the write?",
            arguments={"path": "notes.md", "content": "hello"},
        )
    )

    assert response.decision == "approved"
    rendered = stdout.getvalue()
    assert "Approval Required" in rendered
    assert "Allow the write?" in rendered
    assert '"path": "notes.md"' in rendered


def test_terminal_user_io_renders_question_interrupt() -> None:
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=StringIO(), stdout=stdout, stderr=StringIO(), interactive=True)

    user_io.begin_question_interrupt(
        UserQuestionRequest(
            request_id="question_1",
            prompt="Which file should I edit?",
            allow_free_text=True,
            options=(),
        )
    )

    rendered = stdout.getvalue()
    assert "Agent Question" in rendered
    assert "Which file should I edit?" in rendered
    assert "Free-text answers" in rendered


def test_terminal_user_io_collects_question_response() -> None:
    stdin = StringIO("2\n")
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=stdin, stdout=stdout, stderr=StringIO(), interactive=True)
    request = UserQuestionRequest(
        request_id="question_1",
        prompt="Which file should I edit?",
        options=(
            InterruptOption(value="readme", label="README.md"),
            InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
        ),
    )

    response = user_io.resolve_question_interrupt(request)

    assert response == UserQuestionResponse(
        request_id="question_1",
        selected_option="arch",
    )
    assert "Answer" in stdout.getvalue()


def test_terminal_user_io_raises_on_eof_while_waiting_for_question_response() -> None:
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=StringIO(""), stdout=stdout, stderr=StringIO(), interactive=True)
    request = UserQuestionRequest(
        request_id="question_1",
        prompt="Which file should I edit?",
        options=(
            InterruptOption(value="readme", label="README.md"),
        ),
    )

    try:
        user_io.resolve_question_interrupt(request)
    except RuntimeError as exc:
        assert str(exc) == "Input closed while waiting for a question response."
    else:
        raise AssertionError("expected RuntimeError on EOF")


def test_terminal_user_io_rejects_unanswerable_question_request() -> None:
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=StringIO("ignored\n"), stdout=stdout, stderr=StringIO(), interactive=True)
    request = object.__new__(UserQuestionRequest)
    object.__setattr__(request, "request_id", "question_1")
    object.__setattr__(request, "prompt", "Which file should I edit?")
    object.__setattr__(request, "options", ())
    object.__setattr__(request, "allow_free_text", False)
    object.__setattr__(request, "details", {})

    try:
        user_io.resolve_question_interrupt(request)
    except RuntimeError as exc:
        assert str(exc) == "Question interrupt has no options and does not allow free text."
    else:
        raise AssertionError("expected RuntimeError for unanswerable question request")


def test_cli_main_shows_shortcuts_help() -> None:
    stdout = StringIO()
    exit_code = main(
        ["--shortcuts"],
        stdin=StringIO(),
        stdout=stdout,
        stderr=StringIO(),
    )
    assert exit_code == 0
    output = stdout.getvalue()
    assert "Keyboard Shortcuts" in output
    assert "Alt+Enter" in output
    assert "Ctrl+C" in output


def test_cli_main_renders_completed_turn_with_fake_app() -> None:
    stdout = StringIO()
    fake_app = FakeRuntimeApp(
        CompletedTurn(
            message=Message(role="assistant", content="Done."),
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )

    exit_code = main(
        ["Inspect the repo."],
        stdin=StringIO(),
        stdout=stdout,
        stderr=StringIO(),
        app_factory=lambda args, user_io: fake_app,
    )

    assert exit_code == 0
    assert fake_app.tasks == ["Inspect the repo."]
    # Output now includes rich-formatted conversation display
    output = stdout.getvalue()
    assert "You" in output  # Claude Code-style user prompt
    assert "Done." in output


def test_cli_main_prompts_for_task_when_argument_is_missing() -> None:
    stdout = StringIO()
    fake_app = FakeRuntimeApp(
        CompletedTurn(message=Message(role="assistant", content="Ready."))
    )

    exit_code = main(
        [],
        stdin=StringIO("Inspect README.md\n"),
        stdout=stdout,
        stderr=StringIO(),
        app_factory=lambda args, user_io: fake_app,
    )

    assert exit_code == 0
    assert fake_app.tasks == ["Inspect README.md"]
    # Output now includes rich-formatted content
    output = stdout.getvalue()
    assert "Ready." in output


def test_cli_main_reports_pause_for_question_interrupt() -> None:
    stdout = StringIO()
    fake_app = FakeRuntimeApp(
        InterruptedTurn(
            interrupt=InterruptMetadata(
                kind="question",
                prompt="Which file should I edit?",
                request_id="question_1",
            ),
            assistant_message=Message(
                role="assistant",
                content="Need clarification.",
                tool_calls=(
                    ToolCall(
                        id="call_question",
                        name="AskUserQuestion",
                        arguments={"prompt": "Which file should I edit?"},
                    ),
                ),
            ),
            tool_call=ToolCall(
                id="call_question",
                name="AskUserQuestion",
                arguments={"prompt": "Which file should I edit?"},
            ),
            tool_message=Message(
                role="tool",
                name="AskUserQuestion",
                content="Need clarification.",
                tool_call_id="call_question",
            ),
        )
    )

    exit_code = main(
        ["Continue."],
        stdin=StringIO(),
        stdout=stdout,
        stderr=StringIO(),
        app_factory=lambda args, user_io: fake_app,
    )

    assert exit_code == 0
    assert "Execution paused awaiting more user input." in stdout.getvalue()


def test_cli_main_runs_full_question_resume_session_with_options_and_free_text(
    tmp_path,
) -> None:
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I need clarification.",
                    tool_calls=(
                        ToolCall(
                            id="call_question",
                            name="AskUserQuestion",
                            arguments={
                                "prompt": "Which file should I edit?",
                                "request_id": "question_1",
                                "options": [
                                    {"value": "readme", "label": "README.md"},
                                    {"value": "arch", "label": "docs/ARCHITECTURE.md"},
                                ],
                                "allow_free_text": True,
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="Use notes.md."),
                finish_reason="stop",
            ),
        ]
    )

    stdout = StringIO()
    stderr = StringIO()

    # Run in interactive mode so StringIO input is read for free-text response
    from apps.cli import TerminalUserIO

    def interactive_app_factory(args, user_io):
        # Force interactive mode for the user_io so it reads from stdin StringIO
        user_io._interactive = True
        return build_runtime_app(
            model=model,
            user_io=user_io,
            workspace_root=tmp_path,
            registry=FakeRegistry([AskUserQuestionTool()]),
            approval_policy=ApprovalPolicy(),
        )

    exit_code = main(
        ["Continue."],
        stdin=StringIO("notes.md\n"),
        stdout=stdout,
        stderr=stderr,
        app_factory=interactive_app_factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Approval Required" not in stdout.getvalue()
    # Rich UI now shows "Agent Question" in the panel title
    assert "Agent Question" in stdout.getvalue()
    assert "README.md" in stdout.getvalue()
    assert "docs/ARCHITECTURE.md" in stdout.getvalue()
    assert "Free-text answers" in stdout.getvalue()
    assert "Use notes.md." in stdout.getvalue()
    assert len(model.requests) == 2
    assert model.requests[1].messages[-1].content == (
        "Interrupt resume context:\n"
        '{\n  "free_text": "notes.md",\n'
        '  "kind": "question",\n'
        '  "request_id": "question_1"\n}'
    )


class TestCliSettingsIntegration:
    """Tests for CLI integration with settings.json configuration."""

    def test_cli_main_reports_configuration_error_on_invalid_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI reports configuration error when settings file is invalid."""
        # Create invalid settings file
        agentlet_dir = tmp_path / ".agentlet"
        agentlet_dir.mkdir()
        settings_file = agentlet_dir / "settings.json"
        settings_file.write_text("not valid json")

        # Mock the settings path
        monkeypatch.setattr(
            "agentlet.config.settings.SettingsLoader.SETTINGS_PATH",
            settings_file,
        )

        exit_code = main(
            ["task"],
            stdin=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
            app_factory=lambda args, user_io: FakeRuntimeApp(
                CompletedTurn(message=Message(role="assistant", content="Done."))
            ),
        )

        assert exit_code == 2

    def test_cli_main_applies_settings_defaults_to_parser(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI uses defaults from settings file for argument parser."""
        # Create workspace in tmp_path
        workspace = tmp_path / "my_workspace"
        workspace.mkdir()
        state_dir = tmp_path / "custom_state"

        # Create settings file
        agentlet_home = tmp_path / "agentlet_home"
        agentlet_home.mkdir()
        settings_file = agentlet_home / "settings.json"
        settings_file.write_text(json.dumps({
            "defaults": {
                "provider": "anthropic",
                "workspace_root": str(workspace),
                "state_dir": str(state_dir),
            },
        }))

        # Mock the settings path
        monkeypatch.setattr(
            "agentlet.config.settings.SettingsLoader.SETTINGS_PATH",
            settings_file,
        )

        received_args = {}

        def capture_app_factory(args, user_io):
            received_args["provider"] = args.provider
            received_args["workspace_root"] = args.workspace_root
            received_args["state_dir"] = args.state_dir
            return FakeRuntimeApp(
                CompletedTurn(message=Message(role="assistant", content="Done."))
            )

        stdout = StringIO()
        exit_code = main(
            ["task"],
            stdin=StringIO(),
            stdout=stdout,
            stderr=StringIO(),
            app_factory=capture_app_factory,
        )

        assert exit_code == 0
        assert received_args["provider"] == "anthropic"
        assert received_args["workspace_root"] == str(workspace)
        assert received_args["state_dir"] == str(state_dir)

    def test_cli_main_cli_args_override_settings_defaults(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI arguments take precedence over settings defaults."""
        # Create workspaces
        settings_workspace = tmp_path / "settings_workspace"
        settings_workspace.mkdir()
        cli_workspace = tmp_path / "cli_workspace"
        cli_workspace.mkdir()

        # Create settings file
        agentlet_home = tmp_path / "agentlet_home"
        agentlet_home.mkdir()
        settings_file = agentlet_home / "settings.json"
        settings_file.write_text(json.dumps({
            "defaults": {
                "workspace_root": str(settings_workspace),
            },
        }))

        # Mock the settings path
        monkeypatch.setattr(
            "agentlet.config.settings.SettingsLoader.SETTINGS_PATH",
            settings_file,
        )

        received_args = {}

        def capture_app_factory(args, user_io):
            received_args["provider"] = args.provider
            received_args["workspace_root"] = args.workspace_root
            return FakeRuntimeApp(
                CompletedTurn(message=Message(role="assistant", content="Done."))
            )

        exit_code = main(
            ["--provider", "openai-like", "--workspace-root", str(cli_workspace), "task"],
            stdin=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
            app_factory=capture_app_factory,
        )

        assert exit_code == 0
        # CLI argument should override settings default
        assert received_args["provider"] == "openai-like"
        assert received_args["workspace_root"] == str(cli_workspace)

    def test_cli_main_applies_env_from_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI applies environment variables from settings file."""
        test_var = "AGENTLET_TEST_SETTINGS_ENV_12345"

        # Clear the variable first
        monkeypatch.delenv(test_var, raising=False)

        # Create settings file with env variable
        agentlet_home = tmp_path / "agentlet_home"
        agentlet_home.mkdir()
        settings_file = agentlet_home / "settings.json"
        settings_file.write_text(json.dumps({
            "env": {test_var: "from_settings"},
        }))

        # Mock the settings path
        monkeypatch.setattr(
            "agentlet.config.settings.SettingsLoader.SETTINGS_PATH",
            settings_file,
        )

        # Run CLI
        main(
            ["task"],
            stdin=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
            app_factory=lambda args, user_io: FakeRuntimeApp(
                CompletedTurn(message=Message(role="assistant", content="Done."))
            ),
        )

        # Verify env var was set from settings
        assert os.environ.get(test_var) == "from_settings"

    def test_cli_main_preserves_existing_env_over_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Existing environment variables take precedence over settings."""
        test_var = "AGENTLET_TEST_SETTINGS_ENV_12345"

        # Set the variable before CLI runs
        monkeypatch.setenv(test_var, "existing_value")

        # Create settings file with different value
        agentlet_home = tmp_path / "agentlet_home"
        agentlet_home.mkdir()
        settings_file = agentlet_home / "settings.json"
        settings_file.write_text(json.dumps({
            "env": {test_var: "from_settings"},
        }))

        # Mock the settings path
        monkeypatch.setattr(
            "agentlet.config.settings.SettingsLoader.SETTINGS_PATH",
            settings_file,
        )

        # Run CLI
        main(
            ["task"],
            stdin=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
            app_factory=lambda args, user_io: FakeRuntimeApp(
                CompletedTurn(message=Message(role="assistant", content="Done."))
            ),
        )

        # Existing value should be preserved
        assert os.environ.get(test_var) == "existing_value"
