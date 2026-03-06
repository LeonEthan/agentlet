from __future__ import annotations

from io import StringIO

from apps.cli import TerminalUserIO, main
from agentlet.core.loop import CompletedTurn, InterruptedTurn
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, InterruptOption, TokenUsage
from agentlet.runtime.events import (
    ApprovalRequest,
    UserQuestionRequest,
    UserQuestionResponse,
)


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
    user_io = TerminalUserIO(stdin=stdin, stdout=stdout, stderr=StringIO())

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
    assert "Approval required for Write" in rendered
    assert "Allow the write?" in rendered
    assert '"path": "notes.md"' in rendered


def test_terminal_user_io_renders_question_interrupt() -> None:
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=StringIO(), stdout=stdout, stderr=StringIO())

    user_io.begin_question_interrupt(
        UserQuestionRequest(
            request_id="question_1",
            prompt="Which file should I edit?",
            allow_free_text=True,
            options=(),
        )
    )

    rendered = stdout.getvalue()
    assert "Agent needs clarification before continuing." in rendered
    assert "Which file should I edit?" in rendered
    assert "Free-text answers are allowed." in rendered


def test_terminal_user_io_collects_question_response() -> None:
    stdin = StringIO("2\n")
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=stdin, stdout=stdout, stderr=StringIO())
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
    assert "Answer: " in stdout.getvalue()


def test_terminal_user_io_raises_on_eof_while_waiting_for_question_response() -> None:
    stdout = StringIO()
    user_io = TerminalUserIO(stdin=StringIO(""), stdout=stdout, stderr=StringIO())
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
    user_io = TerminalUserIO(stdin=StringIO("ignored\n"), stdout=stdout, stderr=StringIO())
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
    assert stdout.getvalue() == "Done.\n"


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
    assert stdout.getvalue() == "Task: Ready.\n"


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
