from __future__ import annotations

"""Interactive session persistence and resume helpers."""

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlet.agent.agent_loop import TurnEvent
from agentlet.agent.context import Context, Message, ToolCall

# Directory and file constants
AGENTLET_DIR = ".agentlet"
SESSIONS_DIR = "sessions"
LATEST_FILE = "latest"
HISTORY_FILE = "history"


def get_data_dir() -> Path:
    """Return the global agentlet data directory (~/.agentlet)."""
    return Path.home() / AGENTLET_DIR


def _cwd_hash(cwd: Path) -> str:
    """Generate a short hash for the working directory path."""
    return hashlib.md5(str(cwd.resolve()).encode()).hexdigest()[:16]

# Record type constants
RECORD_TYPE_SESSION_STARTED = "session_started"
RECORD_TYPE_USER_MESSAGE = "user_message"
RECORD_TYPE_ASSISTANT_MESSAGE = "assistant_message"
RECORD_TYPE_TOOL_CALL = "tool_call"
RECORD_TYPE_TOOL_RESULT = "tool_result"
RECORD_TYPE_TURN_FINISHED = "turn_finished"

SCHEMA_VERSION = 1


class SessionError(RuntimeError):
    """Raised when session persistence or resume cannot proceed."""


class SessionNotFoundError(SessionError):
    """Raised when a requested session transcript does not exist."""


@dataclass(frozen=True)
class SessionInfo:
    """Metadata needed to keep working with one interactive session."""

    session_id: str
    transcript_path: Path
    cwd: str
    provider_name: str
    model: str
    api_base: str | None
    temperature: float
    max_tokens: int | None
    system_prompt: str


@dataclass(frozen=True)
class LoadedSession:
    """A resumed session and its reconstructed context."""

    info: SessionInfo
    context: Context


class SessionTurnRecorder:
    """Buffer one successful turn into durable transcript records."""

    def __init__(self) -> None:
        self._records: list[tuple[str, dict[str, Any]]] = []

    def observe(self, event: TurnEvent) -> None:
        if event.kind == "turn_started":
            self._records.append(
                (
                    RECORD_TYPE_USER_MESSAGE,
                    {
                        "content": event.user_input or "",
                    },
                )
            )
            return

        if event.kind == "assistant_completed":
            self._records.append(
                (
                    RECORD_TYPE_ASSISTANT_MESSAGE,
                    {
                        "content": event.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments_json": call.arguments_json,
                            }
                            for call in event.tool_calls
                        ],
                    },
                )
            )
            return

        if event.kind == "tool_requested" and event.tool_call is not None:
            self._records.append(
                (
                    RECORD_TYPE_TOOL_CALL,
                    {
                        "id": event.tool_call.id,
                        "name": event.tool_call.name,
                        "arguments_json": event.tool_call.arguments_json,
                    },
                )
            )
            return

        if event.kind == "tool_completed" and event.tool_result is not None:
            self._records.append(
                (
                    RECORD_TYPE_TOOL_RESULT,
                    {
                        "tool_call_id": event.tool_result.tool_call_id,
                        "name": event.tool_result.name,
                        "content": event.tool_result.content,
                    },
                )
            )
            return

        if event.kind == "turn_completed" and event.result is not None:
            self._records.append(
                (
                    RECORD_TYPE_TURN_FINISHED,
                    {
                        "iterations": event.result.iterations,
                        "finish_reason": event.result.finish_reason,
                    },
                )
            )
            return

        if event.kind == "turn_failed":
            # Mark as failed by clearing records - build_records will detect incomplete turn
            self._records.clear()

    def build_records(self, session_id: str) -> list[dict[str, Any]]:
        """Return JSONL-ready records for one successful interactive turn."""
        # Check if turn was marked as failed (records cleared) or is incomplete
        has_completion = any(r[0] == RECORD_TYPE_TURN_FINISHED for r in self._records)
        if not has_completion:
            raise SessionError("Cannot persist an incomplete or failed turn.")

        return [
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "timestamp": _utc_timestamp(),
                "type": record_type,
                "payload": payload,
            }
            for record_type, payload in self._records
        ]


class SessionStore:
    """Manage session transcript files scoped to one working directory."""

    def __init__(self, cwd: Path, data_dir: Path | None = None) -> None:
        """
        Args:
            cwd: The working directory this session store is scoped to.
            data_dir: Optional override for the base data directory.
                     Defaults to ~/.agentlet
        """
        self.cwd = cwd
        self.data_dir = data_dir or get_data_dir()
        # Sessions are grouped by cwd hash: ~/.agentlet/sessions/{hash}/
        self.sessions_dir = self.data_dir / SESSIONS_DIR / _cwd_hash(cwd)
        self.latest_path = self.sessions_dir / LATEST_FILE
        self.legacy_sessions_dir = self.cwd / AGENTLET_DIR / SESSIONS_DIR
        self.legacy_latest_path = self.legacy_sessions_dir / LATEST_FILE
        # History is global (not scoped to cwd)
        self.history_path = self.data_dir / HISTORY_FILE

    def start_session(
        self,
        *,
        provider_name: str,
        model: str,
        api_base: str | None,
        temperature: float,
        max_tokens: int | None,
        system_prompt: str,
    ) -> SessionInfo:
        """Create a new transcript file and write the session header."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        session_id = _generate_session_id()
        info = SessionInfo(
            session_id=session_id,
            transcript_path=self._session_path(session_id),
            cwd=str(self.cwd),
            provider_name=provider_name,
            model=model,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "timestamp": _utc_timestamp(),
            "type": RECORD_TYPE_SESSION_STARTED,
            "payload": {
                "cwd": info.cwd,
                "provider_name": provider_name,
                "model": model,
                "api_base": api_base,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
            },
        }
        with info.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        return info

    def append_records(
        self,
        session: str | SessionInfo,
        records: list[dict[str, Any]],
        *,
        update_latest: bool,
    ) -> None:
        """Append committed turn records and optionally update latest metadata."""
        if isinstance(session, SessionInfo):
            session_id = session.session_id
            path = session.transcript_path
            latest_path = path.parent / LATEST_FILE
        else:
            session_id = session
            path = self._session_path(session_id)
            latest_path = self.latest_path

        path.parent.mkdir(parents=True, exist_ok=True)

        # Batch write all records in a single file operation
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

        if update_latest:
            self._write_latest(session_id, latest_path=latest_path)

    def load_latest_session_id(self) -> str:
        """Read the latest session pointer for the current working directory."""
        try:
            content = self.latest_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            try:
                content = self.legacy_latest_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                raise SessionNotFoundError(
                    f"No latest session metadata found at {self.latest_path}."
                ) from exc
        if not content:
            if self.latest_path.exists():
                raise SessionError(f"Latest session metadata is empty: {self.latest_path}")
            raise SessionError(f"Latest session metadata is empty: {self.legacy_latest_path}")
        return content

    def load_session(self, session_id: str) -> LoadedSession:
        """Load one transcript and rebuild its in-memory context."""
        path = self._resolve_session_path(session_id)

        try:
            file_handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"Session transcript not found: {path}") from exc

        system_prompt: str | None = None
        session_cwd = str(self.cwd)
        provider_name = "unknown"
        model = "unknown"
        api_base: str | None = None
        temperature = 0.0
        max_tokens: int | None = None
        history: list[Message] = []
        known_tool_calls: dict[str, ToolCall] = {}
        turn_open = False
        open_turn_line: int | None = None

        with file_handle as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = self._parse_record(path, line_number, line)
                self._validate_record(path, line_number, record, expected_session_id=session_id)

                record_type = record["type"]
                payload = record["payload"]
                if record_type == RECORD_TYPE_SESSION_STARTED:
                    if system_prompt is not None:
                        raise SessionError(
                            f"{path}:{line_number} duplicate session_started record."
                    )
                    system_prompt = self._read_required_text(
                        path,
                        line_number,
                        payload,
                        "system_prompt",
                    )
                    session_cwd = self._read_optional_text(payload, "cwd") or session_cwd
                    provider_name = str(payload.get("provider_name") or provider_name)
                    model = str(payload.get("model") or model)
                    api_base = self._read_optional_text(payload, "api_base")
                    temperature = self._read_float(
                        path,
                        line_number,
                        payload,
                        "temperature",
                        default=0.0,
                    )
                    max_tokens = self._read_optional_int(
                        path,
                        line_number,
                        payload,
                        "max_tokens",
                    )
                    continue

                if system_prompt is None:
                    raise SessionError(
                        f"{path}:{line_number} session_started must be the first record."
                    )

                if record_type == RECORD_TYPE_USER_MESSAGE:
                    if turn_open:
                        raise SessionError(
                            f"{path}:{line_number} previous turn is missing turn_finished."
                        )
                    turn_open = True
                    open_turn_line = line_number
                    history.append(
                        Message(role="user", content=self._read_optional_text(payload, "content"))
                    )
                    continue

                if record_type == RECORD_TYPE_ASSISTANT_MESSAGE:
                    self._require_open_turn(path, line_number, record_type, turn_open)
                    open_turn_line = line_number
                    raw_tool_calls = payload.get("tool_calls", [])
                    if not isinstance(raw_tool_calls, list):
                        raise SessionError(
                            f"{path}:{line_number} tool_calls must be a list."
                        )
                    tool_calls = tuple(
                        self._tool_call_from_payload(path, line_number, item)
                        for item in raw_tool_calls
                    )
                    for tool_call in tool_calls:
                        known_tool_calls[tool_call.id] = tool_call
                    history.append(
                        Message(
                            role="assistant",
                            content=self._read_optional_text(payload, "content"),
                            tool_calls=tool_calls,
                        )
                    )
                    continue

                if record_type == RECORD_TYPE_TOOL_CALL:
                    self._require_open_turn(path, line_number, record_type, turn_open)
                    open_turn_line = line_number
                    tool_call = self._tool_call_from_payload(path, line_number, payload)
                    known = known_tool_calls.get(tool_call.id)
                    if known is None or known != tool_call:
                        raise SessionError(
                            f"{path}:{line_number} tool_call does not match the assistant transcript."
                        )
                    continue

                if record_type == RECORD_TYPE_TOOL_RESULT:
                    self._require_open_turn(path, line_number, record_type, turn_open)
                    open_turn_line = line_number
                    history.append(
                        Message(
                            role="tool",
                            content=self._read_required_text(
                                path,
                                line_number,
                                payload,
                                "content",
                                allow_empty=True,
                            ),
                            name=self._read_required_text(path, line_number, payload, "name"),
                            tool_call_id=self._read_required_text(
                                path,
                                line_number,
                                payload,
                                "tool_call_id",
                            ),
                        )
                    )
                    continue

                if record_type == RECORD_TYPE_TURN_FINISHED:
                    self._require_open_turn(path, line_number, record_type, turn_open)
                    turn_open = False
                    open_turn_line = None
                    continue

                raise SessionError(f"{path}:{line_number} unknown record type: {record_type}")

        if system_prompt is None:
            raise SessionError(f"{path} is missing a session_started record.")
        if turn_open:
            raise SessionError(
                f"{path}:{open_turn_line} incomplete turn is missing turn_finished."
            )

        info = SessionInfo(
            session_id=session_id,
            transcript_path=path,
            cwd=session_cwd,
            provider_name=provider_name,
            model=model,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return LoadedSession(
            info=info,
            context=Context(system_prompt=system_prompt, history=history),
        )

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def _legacy_session_path(self, session_id: str) -> Path:
        return self.legacy_sessions_dir / f"{session_id}.jsonl"

    def _resolve_session_path(self, session_id: str) -> Path:
        current_path = self._session_path(session_id)
        if current_path.exists():
            return current_path

        sessions_root = self.data_dir / SESSIONS_DIR
        for candidate in sorted(sessions_root.glob(f"*/{session_id}.jsonl")):
            return candidate

        migrated_path = self._migrate_legacy_session(session_id)
        if migrated_path is not None:
            return migrated_path

        return current_path

    def _migrate_legacy_session(self, session_id: str) -> Path | None:
        legacy_path = self._legacy_session_path(session_id)
        if not legacy_path.exists():
            return None

        new_path = self._session_path(session_id)
        if not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")

        if not self.latest_path.exists():
            try:
                latest_session_id = self.legacy_latest_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                latest_session_id = None
            if latest_session_id == session_id:
                self._write_latest(session_id)

        return new_path

    def _write_latest(self, session_id: str, *, latest_path: Path | None = None) -> None:
        resolved_latest_path = latest_path or self.latest_path
        resolved_latest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = resolved_latest_path.with_suffix(".tmp")
        temp_path.write_text(f"{session_id}\n", encoding="utf-8")
        temp_path.replace(resolved_latest_path)

    def _parse_record(
        self,
        path: Path,
        line_number: int,
        raw_line: str,
    ) -> dict[str, Any]:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SessionError(
                f"{path}:{line_number} malformed JSON: {exc.msg}"
            ) from exc

        if not isinstance(record, dict):
            raise SessionError(f"{path}:{line_number} session record must be an object.")
        return record

    def _validate_record(
        self,
        path: Path,
        line_number: int,
        record: dict[str, Any],
        *,
        expected_session_id: str,
    ) -> None:
        schema_version = record.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise SessionError(
                f"{path}:{line_number} unsupported schema_version: {schema_version}"
            )
        if record.get("session_id") != expected_session_id:
            raise SessionError(
                f"{path}:{line_number} session_id mismatch: {record.get('session_id')}"
            )
        if not isinstance(record.get("type"), str):
            raise SessionError(f"{path}:{line_number} record type must be a string.")
        if not isinstance(record.get("payload"), dict):
            raise SessionError(f"{path}:{line_number} record payload must be an object.")

    def _tool_call_from_payload(
        self,
        path: Path,
        line_number: int,
        payload: Any,
    ) -> ToolCall:
        if not isinstance(payload, dict):
            raise SessionError(
                f"{path}:{line_number} tool_calls entries must be objects."
            )
        return ToolCall(
            id=self._read_required_text(path, line_number, payload, "id"),
            name=self._read_required_text(path, line_number, payload, "name"),
            arguments_json=self._read_required_text(
                path,
                line_number,
                payload,
                "arguments_json",
            ),
        )

    def _read_optional_text(self, payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        return str(value)

    def _read_required_text(
        self,
        path: Path,
        line_number: int,
        payload: dict[str, Any],
        key: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        value = payload.get(key)
        if value is None:
            raise SessionError(f"{path}:{line_number} missing required field: {key}")
        if value == "" and not allow_empty:
            raise SessionError(f"{path}:{line_number} missing required field: {key}")
        return str(value)

    def _read_float(
        self,
        path: Path,
        line_number: int,
        payload: dict[str, Any],
        key: str,
        *,
        default: float,
    ) -> float:
        value = payload.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SessionError(f"{path}:{line_number} invalid float field: {key}") from exc

    def _read_optional_int(
        self,
        path: Path,
        line_number: int,
        payload: dict[str, Any],
        key: str,
    ) -> int | None:
        value = payload.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SessionError(f"{path}:{line_number} invalid int field: {key}") from exc

    def _require_open_turn(
        self,
        path: Path,
        line_number: int,
        record_type: str,
        turn_open: bool,
    ) -> None:
        if not turn_open:
            raise SessionError(
                f"{path}:{line_number} {record_type} appears outside a turn."
            )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def load_session_for_resume(
    session_store: SessionStore,
    *,
    continue_session: bool = False,
    session_id: str | None = None,
    loaded_session: LoadedSession | None = None,
    **_: Any,
) -> LoadedSession | None:
    """Resolve session loading logic for resuming an existing session.

    Args:
        session_store: The session store to use.
        continue_session: If True, load the latest session.
        session_id: Specific session ID to load.
        loaded_session: Pre-loaded session (used by tests).
    Returns:
        LoadedSession | None: The resolved session and its context, or None when
        the caller should start a fresh session after setup succeeds.
    """
    if loaded_session is not None:
        return loaded_session

    if session_id is not None:
        return session_store.load_session(session_id)

    if continue_session:
        return session_store.load_session(session_store.load_latest_session_id())

    return None
