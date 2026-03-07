from __future__ import annotations

from agentlet.tools.interaction.ask_user_question import AskUserQuestionTool


def test_ask_user_question_tool_returns_interrupt_result() -> None:
    result = AskUserQuestionTool().execute(
        {
            "prompt": " Which file should I edit? ",
            "request_id": " question_1 ",
            "options": [
                {"value": "readme", "label": "README.md"},
                {"value": "arch", "label": "docs/ARCHITECTURE.md"},
            ],
            "details": {"origin": "model"},
        }
    )

    assert result.interrupt is True
    assert result.is_error is False
    assert result.output == "Awaiting user response."
    assert result.metadata == {
        "interrupt": {
            "kind": "question",
            "prompt": "Which file should I edit?",
            "request_id": "question_1",
            "options": [
                {"value": "readme", "label": "README.md"},
                {"value": "arch", "label": "docs/ARCHITECTURE.md"},
            ],
            "allow_free_text": False,
            "details": {
                "source_tool": "AskUserQuestion",
                "origin": "model",
            },
        }
    }


def test_ask_user_question_tool_rejects_unanswerable_question() -> None:
    result = AskUserQuestionTool().execute(
        {
            "prompt": "Which file should I edit?",
            "request_id": "question_1",
        }
    )

    assert result.is_error is True
    assert result.output == (
        "AskUserQuestion requires options or allow_free_text=True."
    )


def test_ask_user_question_tool_rejects_duplicate_option_values() -> None:
    result = AskUserQuestionTool().execute(
        {
            "prompt": "Which file should I edit?",
            "request_id": "question_1",
            "options": [
                {"value": "readme", "label": "README.md"},
                {"value": "readme", "label": "README copy"},
            ],
        }
    )

    assert result.is_error is True
    assert result.output == (
        "AskUserQuestion option 'value' entries must be unique."
    )
