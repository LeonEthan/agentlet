"""Core agent orchestration loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Protocol
from uuid import uuid4

from agentlet.core.approvals import ApprovalPolicy
from agentlet.core.context import ContextBuilder, CurrentTaskState, PendingInterruptContext
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, JSONObject, TokenUsage, deep_copy_json_object
from agentlet.llm.base import ModelClient
from agentlet.llm.schemas import ModelRequest, ModelToolDefinition, ToolChoice
from agentlet.memory.session_store import SessionRecord
from agentlet.runtime.events import (
    ApprovalRequest,
    ApprovalResponse,
    ResumeRequest,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.tools.base import Tool, ToolDefinition, ToolResult


class SessionStoreLike(Protocol):
    """Minimal session-store contract used by the loop."""

    def load(self, *, skip_malformed: bool = False) -> list[SessionRecord]:
        """Load persisted session records."""

    def append_many(
        self,
        records: list[SessionRecord | JSONObject],
    ) -> list[SessionRecord]:
        """Append normalized session records."""


class MemoryStoreLike(Protocol):
    """Minimal durable-memory contract used by the loop."""

    def read(self) -> str:
        """Return durable memory content."""


class ToolRegistryLike(Protocol):
    """Minimal tool-registry contract used by the loop."""

    def get(self, name: str) -> Tool | None:
        """Return one tool or ``None`` when absent."""

    def definition(self, name: str) -> ToolDefinition:
        """Return the definition for one registered tool."""

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return registered tool definitions."""


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    """Terminal outcome for one completed loop run."""

    message: Message
    usage: TokenUsage | None = None
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError("completed turns require an assistant message")
        object.__setattr__(self, "metadata", deep_copy_json_object(self.metadata))


@dataclass(frozen=True, slots=True)
class ApprovalRequiredTurn:
    """Pause outcome returned when runtime approval is required."""

    request: ApprovalRequest
    assistant_message: Message
    tool_call: ToolCall
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.assistant_message.role != "assistant":
            raise ValueError("assistant_message must have role='assistant'")
        object.__setattr__(self, "metadata", deep_copy_json_object(self.metadata))


@dataclass(frozen=True, slots=True)
class InterruptedTurn:
    """Pause outcome returned when a tool requests an interrupt."""

    interrupt: InterruptMetadata
    assistant_message: Message
    tool_call: ToolCall
    tool_message: Message
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.assistant_message.role != "assistant":
            raise ValueError("assistant_message must have role='assistant'")
        if self.tool_message.role != "tool":
            raise ValueError("tool_message must have role='tool'")
        object.__setattr__(self, "metadata", deep_copy_json_object(self.metadata))


AgentLoopOutcome = CompletedTurn | ApprovalRequiredTurn | InterruptedTurn


@dataclass(slots=True)
class AgentLoop:
    """Thin orchestration loop for state loading, tool execution, and persistence."""

    model: ModelClient
    registry: ToolRegistryLike
    session_store: SessionStoreLike
    memory_store: MemoryStoreLike | None = None
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    tool_choice: ToolChoice | None = None
    max_iterations: int = 8

    def run(
        self,
        *,
        current_task: CurrentTaskState | str | None = None,
        system_instructions: str | None = None,
        resume: ResumeRequest | None = None,
    ) -> AgentLoopOutcome:
        """Run one agent turn until completion or a structured pause."""

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")

        existing_records = self.session_store.load()
        session_history = [
            record for record in existing_records if record.kind == "message"
        ]
        durable_memory = self.memory_store.read() if self.memory_store is not None else ""
        pending_question = _question_response_from_resume(resume)
        if pending_question is not None:
            request, already_consumed = _question_request_for_response(
                pending_question,
                session_history,
            )
            if request is None:
                raise ValueError(
                    "question resume request_id does not match a persisted "
                    "AskUserQuestion interrupt"
                )
            if already_consumed:
                raise ValueError(
                    "question resume request_id has already been consumed"
                )
            request.validate_response(pending_question)
        pending_interrupt = _pending_interrupt_from_resume(resume)
        pending_approval = _approval_response_from_resume(resume)

        working_messages = list(
            self.context_builder.build(
                system_instructions=system_instructions,
                session_history=session_history,
                durable_memory=durable_memory,
                pending_interrupt=pending_interrupt,
                current_task=current_task,
            )
        )
        if not working_messages:
            raise ValueError("agent loop context must not be empty")

        new_messages = _new_input_messages(
            pending_interrupt=pending_interrupt,
            current_task=current_task,
        )

        for _ in range(self.max_iterations):
            response = self.model.complete(self._build_request(working_messages))
            assistant_message = response.message
            working_messages.append(assistant_message)
            new_messages.append(assistant_message)

            if not assistant_message.tool_calls and response.finish_reason != "stop":
                raise RuntimeError(
                    "model returned a non-terminal assistant response without tool calls: "
                    f"{response.finish_reason}"
                )

            if not assistant_message.tool_calls:
                self._persist_messages(existing_records, new_messages)
                return CompletedTurn(
                    message=assistant_message,
                    usage=response.usage,
                    metadata=response.metadata,
                )

            for tool_call in assistant_message.tool_calls:
                validation_error = self._tool_validation_error(tool_call)
                if validation_error is not None:
                    tool_message = _tool_message_from_result(tool_call, validation_error)
                    working_messages.append(tool_message)
                    new_messages.append(tool_message)
                    continue

                approval_resolution = self._resolve_approval_for_tool_call(
                    tool_call=tool_call,
                    assistant_message=assistant_message,
                    pending_approval=pending_approval,
                    existing_records=existing_records,
                )
                if isinstance(approval_resolution, ApprovalRequiredTurn):
                    self._persist_messages(
                        existing_records,
                        new_messages,
                        extra_records=[
                            SessionRecord(
                                record_id=approval_resolution.request.request_id,
                                kind="approval_request",
                                payload=approval_resolution.request.as_dict(),
                            )
                        ],
                    )
                    return approval_resolution
                if isinstance(approval_resolution, ToolResult):
                    tool_message = _tool_message_from_result(
                        tool_call,
                        approval_resolution,
                    )
                    working_messages.append(tool_message)
                    new_messages.append(tool_message)
                    pending_approval = None
                    continue
                if approval_resolution == "approved":
                    pending_approval = None

                tool_message, interrupted_turn = self._execute_tool_call(
                    tool_call=tool_call,
                    assistant_message=assistant_message,
                )
                working_messages.append(tool_message)
                new_messages.append(tool_message)

                if interrupted_turn is not None:
                    self._persist_messages(existing_records, new_messages)
                    return interrupted_turn

        raise RuntimeError(
            f"agent loop exceeded max_iterations={self.max_iterations}"
        )

    def _build_request(self, working_messages: list[Message]) -> ModelRequest:
        tool_definitions = tuple(
            ModelToolDefinition.from_tool_definition(definition)
            for definition in self.registry.definitions()
        )
        return ModelRequest(
            messages=tuple(working_messages),
            tools=tool_definitions,
            tool_choice=self.tool_choice,
        )

    def _resolve_approval_for_tool_call(
        self,
        *,
        tool_call: ToolCall,
        assistant_message: Message,
        pending_approval: ApprovalResponse | None,
        existing_records: list[SessionRecord],
    ) -> ApprovalRequiredTurn | ToolResult | str | None:
        tool = self.registry.get(tool_call.name)
        if tool is None:
            return None
        definition = tool.definition
        decision = self.approval_policy.decision_for_tool(tool)
        if not decision.requires_approval:
            return None

        if pending_approval is not None:
            matched_request = _approval_request_for_response(
                pending_approval,
                existing_records,
            )
            if (
                matched_request is not None
                and _approval_request_matches_tool_call(matched_request, tool_call)
            ):
                if pending_approval.decision == "approved":
                    return "approved"
                return ToolResult.error(
                    output=(
                        f"Tool `{tool_call.name}` was not executed because approval "
                        "was rejected."
                    ),
                    metadata={
                        "tool_name": tool_call.name,
                        "error_type": "ApprovalRejected",
                        "request_id": pending_approval.request_id,
                    },
                )

        request_id = _approval_request_id()
        prompt = (
            f"Allow `{tool_call.name}` to run with the proposed arguments?"
        )
        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_call.name,
            approval_category=decision.approval_category,
            prompt=prompt,
            arguments=tool_call.arguments,
            details={
                "tool_call_id": tool_call.id,
                "reason": decision.reason,
            },
        )
        return ApprovalRequiredTurn(
            request=request,
            assistant_message=assistant_message,
            tool_call=tool_call,
        )

    def _tool_validation_error(self, tool_call: ToolCall) -> ToolResult | None:
        tool = self.registry.get(tool_call.name)
        if tool is None:
            return ToolResult.error(
                output=f"Tool `{tool_call.name}` is not registered.",
                metadata={
                    "tool_name": tool_call.name,
                    "error_type": "UnknownTool",
                },
            )

        validation_error = _validate_arguments_against_schema(
            tool_call.arguments,
            tool.definition.input_schema,
        )
        if validation_error is None:
            return None

        return ToolResult.error(
            output=(
                f"Tool `{tool_call.name}` received invalid arguments: "
                f"{validation_error}"
            ),
            metadata={
                "tool_name": tool_call.name,
                "error_type": "ToolArgumentValidationError",
            },
        )

    def _execute_tool_call(
        self,
        *,
        tool_call: ToolCall,
        assistant_message: Message,
    ) -> tuple[Message, InterruptedTurn | None]:
        tool = self.registry.get(tool_call.name)
        if tool is None:
            tool_result = ToolResult.error(
                output=f"Tool `{tool_call.name}` is not registered.",
                metadata={
                    "tool_name": tool_call.name,
                    "error_type": "UnknownTool",
                },
            )
        else:
            tool_result = _execute_tool(tool, tool_call)

        tool_message = _tool_message_from_result(tool_call, tool_result)
        if not tool_result.interrupt:
            return tool_message, None

        interrupt_payload = (
            {}
            if tool_result.metadata is None
            else tool_result.metadata.get("interrupt", {})
        )
        interrupt = InterruptMetadata.from_dict(interrupt_payload)
        return (
            tool_message,
            InterruptedTurn(
                interrupt=interrupt,
                assistant_message=assistant_message,
                tool_call=tool_call,
                tool_message=tool_message,
            ),
        )

    def _persist_messages(
        self,
        existing_records: list[SessionRecord],
        messages: list[Message],
        extra_records: list[SessionRecord] | None = None,
    ) -> None:
        if not messages and not extra_records:
            return

        starting_index = len(existing_records)
        records = [
            SessionRecord(
                record_id=f"message_{starting_index + index + 1}",
                kind="message",
                payload=asdict(message),
            )
            for index, message in enumerate(messages)
        ]
        if extra_records:
            records.extend(extra_records)
        self.session_store.append_many(records)


def _execute_tool(tool: Tool, tool_call: ToolCall) -> ToolResult:
    try:
        result = tool.execute(tool_call.arguments)
    except Exception as exc:
        return ToolResult.error(
            output=f"Tool `{tool_call.name}` failed: {exc}",
            metadata={
                "tool_name": tool_call.name,
                "error_type": type(exc).__name__,
            },
        )

    if isinstance(result, ToolResult):
        return result

    return ToolResult.error(
        output=(
            f"Tool `{tool_call.name}` returned an invalid result type: "
            f"{type(result).__name__}"
        ),
        metadata={
            "tool_name": tool_call.name,
            "error_type": "InvalidToolResult",
        },
    )


def _tool_message_from_result(tool_call: ToolCall, result: ToolResult) -> Message:
    metadata: JSONObject = {
        "tool_name": tool_call.name,
        "is_error": result.is_error,
    }
    if result.metadata:
        metadata["result"] = deep_copy_json_object(result.metadata)
    return Message(
        role="tool",
        name=tool_call.name,
        content=result.output,
        tool_call_id=tool_call.id,
        metadata=metadata,
    )


def _pending_interrupt_from_resume(
    resume: ResumeRequest | None,
) -> PendingInterruptContext | None:
    if resume is None:
        return None

    payload = resume.payload
    if resume.kind == "approval":
        return PendingInterruptContext(
            kind="approval",
            request_id=payload.request_id,
            decision=payload.decision,
            comment=payload.comment,
            details=payload.details,
        )

    return PendingInterruptContext(
        kind="question",
        request_id=payload.request_id,
        selected_option=payload.selected_option,
        free_text=payload.free_text,
        details=payload.details,
    )


def _approval_response_from_resume(
    resume: ResumeRequest | None,
) -> ApprovalResponse | None:
    if resume is None or resume.kind != "approval":
        return None
    return resume.payload


def _question_response_from_resume(
    resume: ResumeRequest | None,
) -> UserQuestionResponse | None:
    if resume is None or resume.kind != "question":
        return None
    return resume.payload


def _new_input_messages(
    *,
    pending_interrupt: PendingInterruptContext | None,
    current_task: CurrentTaskState | str | None,
) -> list[Message]:
    messages: list[Message] = []
    if pending_interrupt is not None:
        messages.append(pending_interrupt.as_message())

    task_message = _current_task_message(current_task)
    if task_message is not None:
        messages.append(task_message)
    return messages


def _current_task_message(
    current_task: CurrentTaskState | str | None,
) -> Message | None:
    if current_task is None:
        return None
    if isinstance(current_task, CurrentTaskState):
        return current_task.as_message()
    if not current_task.strip():
        return None
    return Message(role="user", content=current_task)


def _approval_request_id() -> str:
    return f"approval:{uuid4().hex}"


def _approval_request_for_response(
    response: ApprovalResponse,
    existing_records: list[SessionRecord],
) -> ApprovalRequest | None:
    for record in existing_records:
        if record.kind != "approval_request":
            continue
        payload_request_id = record.payload.get("request_id")
        if payload_request_id != response.request_id:
            continue
        return ApprovalRequest.from_dict(record.payload)
    return None


def _approval_request_matches_tool_call(
    request: ApprovalRequest,
    tool_call: ToolCall,
) -> bool:
    return (
        request.tool_name == tool_call.name
        and request.arguments == tool_call.arguments
    )


def _question_request_for_response(
    response: UserQuestionResponse,
    session_history: list[SessionRecord],
) -> tuple[UserQuestionRequest | None, bool]:
    matched_request: UserQuestionRequest | None = None
    matched_index: int | None = None

    for index in range(len(session_history) - 1, -1, -1):
        record = session_history[index]
        try:
            message = Message.from_dict(record.payload)
        except (TypeError, ValueError):
            continue
        if message.role != "tool":
            continue
        result_metadata = message.metadata.get("result")
        if not isinstance(result_metadata, dict):
            continue
        interrupt_payload = result_metadata.get("interrupt")
        if not isinstance(interrupt_payload, dict):
            continue
        if interrupt_payload.get("kind") != "question":
            continue
        if interrupt_payload.get("request_id") != response.request_id:
            continue
        interrupt = InterruptMetadata.from_dict(interrupt_payload)
        matched_request = UserQuestionRequest.from_interrupt(interrupt)
        matched_index = index
        break

    if matched_request is None or matched_index is None:
        return None, False

    return (
        matched_request,
        _question_resume_already_consumed(
            request_id=response.request_id,
            session_history=session_history[matched_index + 1 :],
        ),
    )


def _question_resume_already_consumed(
    *,
    request_id: str,
    session_history: list[SessionRecord],
) -> bool:
    for record in session_history:
        try:
            message = Message.from_dict(record.payload)
        except (TypeError, ValueError):
            continue
        if message.role != "user":
            continue
        payload = _resume_context_payload_from_message(message)
        if payload is None:
            continue
        if payload.get("kind") != "question":
            continue
        if payload.get("request_id") == request_id:
            return True
    return False


def _resume_context_payload_from_message(message: Message) -> JSONObject | None:
    prefix = "Interrupt resume context:\n"
    if not message.content.startswith(prefix):
        return None
    raw_payload = message.content[len(prefix) :]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return deep_copy_json_object(payload)


def _validate_arguments_against_schema(
    arguments: JSONObject,
    schema: JSONObject,
) -> str | None:
    schema_type = schema.get("type")
    if schema_type is None:
        return None
    if schema_type != "object":
        return f"unsupported root schema type: {schema_type}"

    properties = schema.get("properties", {})
    if properties is not None and not isinstance(properties, dict):
        return "schema properties must be a mapping"

    required = schema.get("required", [])
    if required is not None and not isinstance(required, list):
        return "schema required must be a list"

    property_schemas = properties if isinstance(properties, dict) else {}
    required_keys = [key for key in required if isinstance(key, str)]
    missing = sorted(key for key in required_keys if key not in arguments)
    if missing:
        return f"missing required arguments: {', '.join(missing)}"

    additional_properties = schema.get("additionalProperties", True)
    if additional_properties is False:
        unexpected = sorted(
            key for key in arguments if key not in property_schemas
        )
        if unexpected:
            return f"unexpected arguments: {', '.join(unexpected)}"

    for key, value in arguments.items():
        property_schema = property_schemas.get(key)
        if not isinstance(property_schema, dict):
            continue
        validation_error = _validate_schema_value(
            value,
            property_schema,
            path=key,
        )
        if validation_error is not None:
            return validation_error

    return None


def _validate_schema_value(
    value: object,
    schema: JSONObject,
    *,
    path: str,
) -> str | None:
    schema_type = schema.get("type")
    if schema_type == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        return None

    if schema_type == "boolean":
        if not isinstance(value, bool):
            return f"{path} must be a boolean"
        return None

    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} must be an integer"
        minimum = schema.get("minimum")
        if isinstance(minimum, int | float) and value < minimum:
            return f"{path} must be >= {minimum}"
        return None

    if schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"{path} must be a number"
        numeric_value = float(value)
        if not isfinite(numeric_value):
            return f"{path} must be a finite number"
        minimum = schema.get("minimum")
        if isinstance(minimum, int | float) and numeric_value < minimum:
            return f"{path} must be >= {minimum}"
        return None

    if schema_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        nested_error = _validate_arguments_against_schema(value, schema)
        if nested_error is None:
            return None
        return f"{path}.{nested_error}"

    return None
