"""Built-in structured question interrupt tool."""

from __future__ import annotations

from dataclasses import dataclass

from agentlet.core.types import InterruptMetadata, InterruptOption, JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult


@dataclass(frozen=True, slots=True)
class AskUserQuestionTool:
    """Return a structured interrupt that must be resumed by the runtime."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="AskUserQuestion",
            description=(
                "Ask the user a structured clarifying question and pause until "
                "the runtime resumes with an answer."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "request_id": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["value", "label"],
                            "additionalProperties": False,
                        },
                    },
                    "allow_free_text": {"type": "boolean", "default": False},
                    "details": {"type": "object"},
                },
                "required": ["prompt", "request_id"],
                "additionalProperties": False,
            },
            approval_category="external_or_interrupt",
        )

    def execute(self, arguments: JSONObject) -> ToolResult:
        prompt = arguments.get("prompt")
        request_id = arguments.get("request_id")
        allow_free_text = arguments.get("allow_free_text", False)
        details = arguments.get("details", {})
        options_payload = arguments.get("options", ())

        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.error(
                "AskUserQuestion requires a non-empty string 'prompt'."
            )
        if not isinstance(request_id, str) or not request_id.strip():
            return ToolResult.error(
                "AskUserQuestion requires a non-empty string 'request_id'."
            )
        if not isinstance(allow_free_text, bool):
            return ToolResult.error(
                "AskUserQuestion requires boolean 'allow_free_text' when provided."
            )
        if not isinstance(details, dict):
            return ToolResult.error(
                "AskUserQuestion requires object 'details' when provided."
            )
        if not isinstance(options_payload, list | tuple):
            return ToolResult.error(
                "AskUserQuestion requires array 'options' when provided."
            )

        try:
            normalized_prompt = prompt.strip()
            normalized_request_id = request_id.strip()
            options = tuple(_normalize_option(option) for option in options_payload)
            _validate_answerability(
                options,
                allow_free_text=allow_free_text,
            )
            interrupt = InterruptMetadata(
                kind="question",
                prompt=normalized_prompt,
                request_id=normalized_request_id,
                options=options,
                allow_free_text=allow_free_text,
                details={
                    "source_tool": "AskUserQuestion",
                    **details,
                },
            )
        except ValueError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.interrupt_result(
            output="Awaiting user response.",
            interrupt=interrupt,
        )


def _normalize_option(payload: object) -> InterruptOption:
    if not isinstance(payload, dict):
        raise ValueError("AskUserQuestion options must be objects.")
    value = payload.get("value")
    label = payload.get("label")
    description = payload.get("description")
    if not isinstance(value, str) or not value:
        raise ValueError("AskUserQuestion option 'value' must be a non-empty string.")
    if not isinstance(label, str) or not label:
        raise ValueError("AskUserQuestion option 'label' must be a non-empty string.")
    if description is not None and not isinstance(description, str):
        raise ValueError("AskUserQuestion option 'description' must be a string.")
    return InterruptOption(
        value=value,
        label=label,
        description=description,
    )


def _validate_answerability(
    options: tuple[InterruptOption, ...],
    *,
    allow_free_text: bool,
) -> None:
    if not options and not allow_free_text:
        raise ValueError(
            "AskUserQuestion requires options or allow_free_text=True."
        )

    seen_values: set[str] = set()
    for option in options:
        if option.value in seen_values:
            raise ValueError(
                "AskUserQuestion option 'value' entries must be unique."
            )
        seen_values.add(option.value)


__all__ = ["AskUserQuestionTool"]
