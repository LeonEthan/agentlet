from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlet.agent.agent_loop import AgentTurnResult, TurnEvent
from agentlet.agent.context import Context, ToolCall, ToolResult
from agentlet.cli.sessions import (
    SessionError,
    SessionNotFoundError,
    SessionStore,
    SessionTurnRecorder,
    _cwd_hash,
)


def _make_store(tmp_path: Path, cwd: Path) -> SessionStore:
    """Create a SessionStore with tmp_path as the data_dir for testing."""
    return SessionStore(cwd, data_dir=tmp_path)


def _session_path(tmp_path: Path, cwd: Path, session_id: str) -> Path:
    """Return the expected session file path for testing."""
    return tmp_path / "sessions" / _cwd_hash(cwd) / f"{session_id}.jsonl"


@pytest.fixture
def project_store(tmp_path: Path) -> tuple[Path, SessionStore]:
    """Create a project directory and SessionStore for testing."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    store = _make_store(tmp_path, cwd)
    return cwd, store


def test_session_store_round_trips_completed_turns(project_store: tuple[Path, SessionStore]) -> None:
    cwd, store = project_store
    info = store.start_session(
        provider_name="openai",
        model="gpt-5.4",
        api_base="http://localhost:4000/v1",
        temperature=0.7,
        max_tokens=256,
        system_prompt="system",
    )
    recorder = SessionTurnRecorder()
    tool_call = ToolCall(
        id="call-1",
        name="echo",
        arguments_json='{"text":"hello"}',
    )

    recorder.observe(TurnEvent(kind="turn_started", user_input="say hello"))
    recorder.observe(
        TurnEvent(
            kind="assistant_completed",
            content=None,
            tool_calls=(tool_call,),
        )
    )
    recorder.observe(TurnEvent(kind="tool_requested", tool_call=tool_call))
    recorder.observe(
        TurnEvent(
            kind="tool_completed",
            tool_result=ToolResult(
                tool_call_id="call-1",
                name="echo",
                content="hello",
            ),
        )
    )
    recorder.observe(
        TurnEvent(
            kind="assistant_completed",
            content="done",
        )
    )
    recorder.observe(
        TurnEvent(
            kind="turn_completed",
            result=AgentTurnResult(
                output="done",
                context=Context(system_prompt="system"),
                iterations=2,
                finish_reason="stop",
            ),
        )
    )

    store.append_records(
        info.session_id,
        recorder.build_records(info.session_id),
        update_latest=True,
    )

    loaded = store.load_session(info.session_id)

    assert store.load_latest_session_id() == info.session_id
    assert loaded.info.api_base == "http://localhost:4000/v1"
    assert loaded.info.temperature == 0.7
    assert loaded.info.max_tokens == 256
    assert [message.role for message in loaded.context.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert loaded.context.history[-1].content == "done"


def test_session_store_updates_latest_only_after_first_completed_turn(
    project_store: tuple[Path, SessionStore],
) -> None:
    _, store = project_store
    info = store.start_session(
        provider_name="openai",
        model="gpt-5.4",
        api_base=None,
        temperature=0.0,
        max_tokens=None,
        system_prompt="system",
    )

    with pytest.raises(SessionNotFoundError, match="No latest session metadata found"):
        store.load_latest_session_id()

    recorder = SessionTurnRecorder()
    recorder.observe(TurnEvent(kind="turn_started", user_input="hello"))
    recorder.observe(TurnEvent(kind="assistant_completed", content="done"))
    recorder.observe(
        TurnEvent(
            kind="turn_completed",
            result=AgentTurnResult(
                output="done",
                context=Context(system_prompt="system"),
                iterations=1,
                finish_reason="stop",
            ),
        )
    )
    store.append_records(
        info.session_id,
        recorder.build_records(info.session_id),
        update_latest=True,
    )

    assert store.load_latest_session_id() == info.session_id


def test_session_store_reports_malformed_json_with_line_number(
    tmp_path: Path,
    project_store: tuple[Path, SessionStore],
) -> None:
    cwd, store = project_store
    session_id = "bad-session"
    transcript_path = _session_path(tmp_path, cwd, session_id)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:00+00:00",
                "type": "session_started",
                "payload": {
                    "cwd": str(cwd),
                    "provider_name": "openai",
                    "model": "gpt-5.4",
                    "system_prompt": "system",
                },
            }
        )
        + "\n"
        + "{bad json}\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionError, match=r"bad-session\.jsonl:2 malformed JSON"):
        store.load_session(session_id)


def test_session_store_reports_missing_system_prompt_in_header(
    tmp_path: Path,
    project_store: tuple[Path, SessionStore],
) -> None:
    cwd, store = project_store
    session_id = "missing-prompt"
    transcript_path = _session_path(tmp_path, cwd, session_id)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:00+00:00",
                "type": "session_started",
                "payload": {
                    "cwd": str(cwd),
                    "provider_name": "openai",
                    "model": "gpt-5.4",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionError, match=r"missing-prompt\.jsonl:1 missing required field: system_prompt"):
        store.load_session(session_id)


def test_session_store_reports_non_list_tool_calls(
    tmp_path: Path,
    project_store: tuple[Path, SessionStore],
) -> None:
    cwd, store = project_store
    session_id = "bad-tool-calls"
    transcript_path = _session_path(tmp_path, cwd, session_id)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:00+00:00",
                "type": "session_started",
                "payload": {
                    "cwd": str(cwd),
                    "provider_name": "openai",
                    "model": "gpt-5.4",
                    "system_prompt": "system",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:00.500000+00:00",
                "type": "user_message",
                "payload": {"content": "hello"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:01+00:00",
                "type": "assistant_message",
                "payload": {
                    "content": "done",
                    "tool_calls": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionError, match=r"bad-tool-calls\.jsonl:3 tool_calls must be a list"):
        store.load_session(session_id)


def test_session_store_rejects_incomplete_final_turn(
    tmp_path: Path,
    project_store: tuple[Path, SessionStore],
) -> None:
    cwd, store = project_store
    session_id = "incomplete-turn"
    transcript_path = _session_path(tmp_path, cwd, session_id)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:00+00:00",
                "type": "session_started",
                "payload": {
                    "cwd": str(cwd),
                    "provider_name": "openai",
                    "model": "gpt-5.4",
                    "system_prompt": "system",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:01+00:00",
                "type": "user_message",
                "payload": {"content": "hello"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "timestamp": "2026-03-12T00:00:02+00:00",
                "type": "assistant_message",
                "payload": {"content": "partial", "tool_calls": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SessionError,
        match=r"incomplete-turn\.jsonl:3 incomplete turn is missing turn_finished",
    ):
        store.load_session(session_id)
