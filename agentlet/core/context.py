"""Deterministic context assembly for model input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Literal

from agentlet.core.messages import Message
from agentlet.core.types import JSONObject, deep_copy_json_object
from agentlet.memory.session_store import SessionRecord

PendingInterruptKind = Literal["approval", "question"]
VALID_PENDING_INTERRUPT_KINDS = {"approval", "question"}
VALID_APPROVAL_DECISIONS = {"approved", "rejected"}


@dataclass(frozen=True, slots=True)
class CurrentTaskState:
    """The current unit of work that should close the assembled context."""

    task: str
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_message(self) -> Message:
        """Render the current task as the final user message."""

        content = self.task
        if self.details:
            serialized_details = json.dumps(
                self.details,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            content = f"{content}\n\nTask state:\n{serialized_details}"
        return Message(role="user", content=content)


@dataclass(frozen=True, slots=True)
class PendingInterruptContext:
    """Structured resume state that should be visible during the next model turn."""

    kind: PendingInterruptKind
    request_id: str
    selected_option: str | None = None
    free_text: str | None = None
    decision: str | None = None
    comment: str | None = None
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_PENDING_INTERRUPT_KINDS:
            raise ValueError(f"unsupported pending interrupt kind: {self.kind}")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.kind == "question":
            if self.selected_option is None and self.free_text is None:
                raise ValueError(
                    "question pending interrupts require selected_option or free_text"
                )
            if self.decision is not None:
                raise ValueError("decision is only valid for approval interrupts")
            if self.comment is not None:
                raise ValueError("comment is only valid for approval interrupts")
        if self.kind == "approval":
            if self.decision not in VALID_APPROVAL_DECISIONS:
                raise ValueError(
                    "approval pending interrupts require decision='approved' or 'rejected'"
                )
            if self.selected_option is not None:
                raise ValueError(
                    "selected_option is only valid for question interrupts"
                )
            if self.free_text is not None:
                raise ValueError("free_text is only valid for question interrupts")
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_message(self) -> Message:
        """Render pending interrupt state as explicit context metadata."""

        payload: JSONObject = {
            "kind": self.kind,
            "request_id": self.request_id,
        }
        if self.selected_option is not None:
            payload["selected_option"] = self.selected_option
        if self.free_text is not None:
            payload["free_text"] = self.free_text
        if self.decision is not None:
            payload["decision"] = self.decision
        if self.comment is not None:
            payload["comment"] = self.comment
        if self.details:
            payload["details"] = deep_copy_json_object(self.details)
        serialized_payload = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        return Message(
            role="user",
            content=f"Interrupt resume context:\n{serialized_payload}",
        )


ContextHistoryEntry = Message | SessionRecord | JSONObject


class ContextBuilder:
    """Build model-facing messages from explicit context sources."""

    def build(
        self,
        *,
        system_instructions: str | None = None,
        session_history: Iterable[ContextHistoryEntry] = (),
        durable_memory: str | None = None,
        pending_interrupt: PendingInterruptContext | None = None,
        current_task: CurrentTaskState | str | None = None,
    ) -> tuple[Message, ...]:
        """Assemble normalized messages in a stable, readable order."""

        messages: list[Message] = []
        system_message = self._build_system_instructions(system_instructions)
        if system_message is not None:
            messages.append(system_message)

        memory_message = self._build_durable_memory(durable_memory)
        if memory_message is not None:
            messages.append(memory_message)

        messages.extend(self._normalize_history(session_history))

        interrupt_message = self._build_pending_interrupt(pending_interrupt)
        if interrupt_message is not None:
            messages.append(interrupt_message)

        task_message = self._build_current_task(current_task)
        if task_message is not None:
            messages.append(task_message)

        return tuple(messages)

    def _build_system_instructions(self, content: str | None) -> Message | None:
        normalized_content = self._normalize_text(content)
        if normalized_content is None:
            return None
        return Message(role="system", content=normalized_content)

    def _build_durable_memory(self, content: str | None) -> Message | None:
        normalized_content = self._normalize_text(content)
        if normalized_content is None:
            return None
        return Message(role="system", content=f"Durable memory:\n{normalized_content}")

    def _build_pending_interrupt(
        self,
        pending_interrupt: PendingInterruptContext | None,
    ) -> Message | None:
        if pending_interrupt is None:
            return None
        return pending_interrupt.as_message()

    def _build_current_task(
        self,
        current_task: CurrentTaskState | str | None,
    ) -> Message | None:
        if current_task is None:
            return None
        if isinstance(current_task, CurrentTaskState):
            return current_task.as_message()

        normalized_content = self._normalize_text(current_task)
        if normalized_content is None:
            return None
        return Message(role="user", content=normalized_content)

    def _normalize_history(
        self,
        session_history: Iterable[ContextHistoryEntry],
    ) -> list[Message]:
        return [self._normalize_history_entry(entry) for entry in session_history]

    def _normalize_history_entry(self, entry: ContextHistoryEntry) -> Message:
        if isinstance(entry, Message):
            return entry

        if isinstance(entry, SessionRecord):
            if entry.kind != "message":
                raise ValueError(
                    "session history records must have kind='message' to build context"
                )
            return Message.from_dict(entry.payload)

        if "role" in entry:
            return Message.from_dict(entry)
        if "kind" in entry:
            return self._normalize_history_entry(SessionRecord.from_dict(entry))
        raise ValueError(
            "session history entries must be Message instances or message-like mappings"
        )

    def _normalize_text(self, content: str | None) -> str | None:
        if content is None:
            return None
        if not content.strip():
            return None
        return content
