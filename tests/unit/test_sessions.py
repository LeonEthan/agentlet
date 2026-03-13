from __future__ import annotations

import hashlib
import json
import logging
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


def _legacy_session_path(cwd: Path, session_id: str) -> Path:
    """Return the legacy project-local session path."""
    return cwd / ".agentlet" / "sessions" / f"{session_id}.jsonl"


def _turn_records(session_id: str, user_content: str, assistant_content: str) -> list[dict[str, object]]:
    """Build one completed turn's worth of transcript records."""
    return [
        {
            "schema_version": 1,
            "session_id": session_id,
            "timestamp": "2026-03-12T00:00:01+00:00",
            "type": "user_message",
            "payload": {"content": user_content},
        },
        {
            "schema_version": 1,
            "session_id": session_id,
            "timestamp": "2026-03-12T00:00:02+00:00",
            "type": "assistant_message",
            "payload": {"content": assistant_content, "tool_calls": []},
        },
        {
            "schema_version": 1,
            "session_id": session_id,
            "timestamp": "2026-03-12T00:00:03+00:00",
            "type": "turn_finished",
            "payload": {"iterations": 1, "finish_reason": "stop"},
        },
    ]


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


def test_session_store_migrates_legacy_project_local_sessions(tmp_path: Path) -> None:
    data_dir = tmp_path / "home-data"
    cwd = tmp_path / "project"
    cwd.mkdir()
    store = SessionStore(cwd, data_dir=data_dir)
    session_id = "legacy-session"

    legacy_path = _legacy_session_path(cwd, session_id)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
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
        + "\n".join(json.dumps(record) for record in _turn_records(session_id, "hello", "world"))
        + "\n",
        encoding="utf-8",
    )
    (legacy_path.parent / "latest").write_text(f"{session_id}\n", encoding="utf-8")

    assert store.load_latest_session_id() == session_id

    loaded = store.load_session(session_id)
    migrated_path = _session_path(data_dir, cwd, session_id)

    assert loaded.info.transcript_path == migrated_path
    assert migrated_path.exists()

    store.append_records(
        loaded.info,
        _turn_records(session_id, "again", "done"),
        update_latest=True,
    )

    reloaded = store.load_session(session_id)
    user_messages = [message.content for message in reloaded.context.history if message.role == "user"]

    assert user_messages == ["hello", "again"]
    assert store.load_latest_session_id() == session_id


def test_session_store_loads_session_id_from_any_bucket(tmp_path: Path) -> None:
    data_dir = tmp_path / "home-data"
    cwd_one = tmp_path / "project-one"
    cwd_two = tmp_path / "project-two"
    cwd_one.mkdir()
    cwd_two.mkdir()

    store_one = SessionStore(cwd_one, data_dir=data_dir)
    info = store_one.start_session(
        provider_name="openai",
        model="gpt-5.4",
        api_base=None,
        temperature=0.0,
        max_tokens=None,
        system_prompt="system",
    )
    store_one.append_records(
        info,
        _turn_records(info.session_id, "first", "reply"),
        update_latest=True,
    )

    store_two = SessionStore(cwd_two, data_dir=data_dir)
    loaded = store_two.load_session(info.session_id)

    assert loaded.info.cwd == str(cwd_one)
    assert loaded.info.transcript_path == _session_path(data_dir, cwd_one, info.session_id)

    store_two.append_records(
        loaded.info,
        _turn_records(info.session_id, "second", "follow-up"),
        update_latest=True,
    )

    reloaded = store_one.load_session(info.session_id)
    user_messages = [message.content for message in reloaded.context.history if message.role == "user"]

    assert user_messages == ["first", "second"]
    assert store_one.load_latest_session_id() == info.session_id


def test_cwd_hash_uses_sha256_of_resolved_path(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    cwd.mkdir()

    expected = hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]

    assert _cwd_hash(cwd) == expected


def test_session_store_warns_when_multiple_buckets_match_session_id(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = tmp_path / "home-data"
    sessions_root = data_dir / "sessions"
    first_match = sessions_root / "aaaa1111"
    second_match = sessions_root / "bbbb2222"
    session_id = "shared-session"

    first_match.mkdir(parents=True)
    second_match.mkdir(parents=True)
    (first_match / f"{session_id}.jsonl").write_text("first\n", encoding="utf-8")
    (second_match / f"{session_id}.jsonl").write_text("second\n", encoding="utf-8")

    store = SessionStore(tmp_path / "project", data_dir=data_dir)

    with caplog.at_level(logging.WARNING):
        resolved = store._resolve_session_path(session_id)

    assert resolved == first_match / f"{session_id}.jsonl"
    assert f"Multiple session transcripts found for {session_id}" in caplog.text


def test_session_store_legacy_migration_does_not_overwrite_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "home-data"
    cwd = tmp_path / "project"
    cwd.mkdir()
    store = SessionStore(cwd, data_dir=data_dir)
    session_id = "legacy-session"

    legacy_path = _legacy_session_path(cwd, session_id)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy session\n", encoding="utf-8")

    new_path = store._session_path(session_id)
    original_open = Path.open

    def fake_open(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == new_path and mode == "x":
            with original_open(self, "w", encoding="utf-8") as handle:
                handle.write("written by another process\n")
            raise FileExistsError
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    migrated = store._migrate_legacy_session(session_id)

    assert migrated == new_path
    assert new_path.read_text(encoding="utf-8") == "written by another process\n"
