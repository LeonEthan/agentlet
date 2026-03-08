from __future__ import annotations

import json

import pytest

from agentlet.memory import (
    MemoryStore,
    SessionRecord,
    SessionStoreConflictError,
    SessionStore,
    SessionStoreError,
    SessionStoreFormatError,
)


def test_session_store_append_and_load_round_trip(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions" / "turns.jsonl")

    appended = store.append(
        SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})
    )
    appended_many = store.append_many(
        [
            {
                "id": "tool_1",
                "kind": "tool_result",
                "payload": {"tool": "Read", "ok": True},
            },
            SessionRecord(
                record_id="meta_1",
                kind="metadata",
                payload={"turn_id": "turn_1"},
            ),
        ]
    )

    loaded = store.load()

    assert appended == SessionRecord(
        record_id="msg_1",
        kind="message",
        payload={"role": "user"},
    )
    assert appended_many == [
        SessionRecord(
            record_id="tool_1",
            kind="tool_result",
            payload={"tool": "Read", "ok": True},
        ),
        SessionRecord(
            record_id="meta_1",
            kind="metadata",
            payload={"turn_id": "turn_1"},
        ),
    ]
    assert loaded == [
        SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"}),
        SessionRecord(
            record_id="tool_1",
            kind="tool_result",
            payload={"tool": "Read", "ok": True},
        ),
        SessionRecord(
            record_id="meta_1",
            kind="metadata",
            payload={"turn_id": "turn_1"},
        ),
    ]
    assert store.path.read_text(encoding="utf-8").splitlines() == [
        '{"id": "msg_1", "kind": "message", "payload": {"role": "user"}}',
        '{"id": "tool_1", "kind": "tool_result", "payload": {"ok": true, "tool": "Read"}}',
        '{"id": "meta_1", "kind": "metadata", "payload": {"turn_id": "turn_1"}}',
    ]


def test_session_store_load_missing_file_returns_empty_list(tmp_path) -> None:
    store = SessionStore(tmp_path / "missing" / "session.jsonl")

    assert store.load() == []


def test_session_store_can_materialize_empty_jsonl_file(tmp_path) -> None:
    store = SessionStore(tmp_path / "missing" / "session.jsonl")

    store.ensure_exists()

    assert store.path.exists()
    assert store.path.read_text(encoding="utf-8") == ""
    assert store.load() == []


def test_session_store_rejects_invalid_append_input(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")

    with pytest.raises(SessionStoreError, match="id must be a non-empty string"):
        store.append({"id": "", "kind": "message", "payload": {}})

    with pytest.raises(SessionStoreError, match="must be a non-empty string"):
        store.append({"id": "msg_1", "kind": "", "payload": {}})

    with pytest.raises(SessionStoreError, match="must be a mapping"):
        store.append({"id": "msg_1", "kind": "message", "payload": "bad"})  # type: ignore[arg-type]

    with pytest.raises(SessionStoreError, match="SessionRecord or mapping"):
        store.append(["bad"])  # type: ignore[arg-type]


def test_session_store_append_is_idempotent_for_identical_records(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record = SessionRecord(
        record_id="msg_1",
        kind="message",
        payload={"role": "user", "content": "retry-safe"},
    )

    store.append(record)
    store.append(record)

    assert store.load() == [record]
    assert store.path.read_text(encoding="utf-8").count("\n") == 1


def test_session_store_rejects_conflicting_reuse_of_record_id(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.append(
        SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})
    )

    with pytest.raises(SessionStoreConflictError, match="id conflict"):
        store.append(
            SessionRecord(
                record_id="msg_1",
                kind="message",
                payload={"role": "assistant"},
            )
        )


def test_session_store_raises_for_malformed_jsonl_by_default(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.path.write_text(
        '\n'.join(
            [
                '{"id": "msg_1", "kind": "message", "payload": {"role": "user"}}',
                '{"id": "msg_2", "kind": "message", "payload": ',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SessionStoreFormatError, match="line 2"):
        store.load()


def test_session_store_can_skip_malformed_jsonl_lines(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    store.path.write_text(
        '\n'.join(
            [
                '{"id": "msg_1", "kind": "message", "payload": {"role": "user"}}',
                '["not", "a", "mapping"]',
                '{"id": "", "kind": "metadata", "payload": {}}',
                '{"id": "meta_1", "kind": "metadata", "payload": {"turn_id": "turn_1"}}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert store.load(skip_malformed=True) == [
        SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"}),
        SessionRecord(
            record_id="meta_1",
            kind="metadata",
            payload={"turn_id": "turn_1"},
        ),
    ]


def test_session_store_preserves_utf8_text_on_disk(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record = SessionRecord(
        record_id="msg_utf8",
        kind="message",
        payload={"content": "你好, world"},
    )

    store.append(record)

    on_disk = store.path.read_text(encoding="utf-8")
    assert "你好, world" in on_disk
    assert "\\u4f60\\u597d" not in on_disk
    assert store.load() == [record]


def test_memory_store_read_write_round_trip(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")

    store.write("# Durable Memory\n\n- Keep tests fast.\n")

    assert store.read() == "# Durable Memory\n\n- Keep tests fast.\n"


def test_memory_store_preserves_utf8_text(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")

    store.write("# Memory\n\n- 你好\n")

    assert store.read() == "# Memory\n\n- 你好\n"


def test_memory_store_missing_file_returns_empty_string(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")

    assert store.read() == ""


def test_memory_store_can_materialize_empty_markdown_file(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")

    store.ensure_exists()

    assert store.path.exists()
    assert store.read() == ""


def test_memory_store_rejects_non_string_content(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")

    with pytest.raises(TypeError, match="must be a string"):
        store.write(123)  # type: ignore[arg-type]


def test_session_store_get_by_id(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record1 = SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})
    record2 = SessionRecord(record_id="msg_2", kind="message", payload={"role": "assistant"})

    store.append(record1)
    store.append(record2)

    assert store.get("msg_1") == record1
    assert store.get("msg_2") == record2
    assert store.get("nonexistent") is None


def test_session_store_has_record_id(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record = SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})

    store.append(record)

    assert store.has("msg_1") is True
    assert store.has("nonexistent") is False


def test_session_store_caching_avoids_reloads(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record = SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})

    store.append(record)

    # First load populates cache
    records1 = store.load()
    # Second load should use cache
    records2 = store.load()

    assert records1 == records2
    # Records should be copies, not same objects
    assert records1 is not records2


def test_session_store_detects_external_modification(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record1 = SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})

    store.append(record1)
    store.load()  # Populate cache

    # Simulate external modification
    record2 = SessionRecord(record_id="msg_2", kind="message", payload={"role": "assistant"})
    store.path.write_text(
        json.dumps(record2.as_dict(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Should detect modification and reload
    records = store.load()
    assert len(records) == 1
    assert records[0].record_id == "msg_2"


def test_session_store_atomic_append_preserves_existing_data(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")
    record1 = SessionRecord(record_id="msg_1", kind="message", payload={"role": "user"})
    record2 = SessionRecord(record_id="msg_2", kind="message", payload={"role": "assistant"})

    store.append(record1)
    store.append(record2)

    records = store.load()
    assert len(records) == 2
    assert records[0].record_id == "msg_1"
    assert records[1].record_id == "msg_2"


def test_session_store_index_preserved_across_multiple_appends(tmp_path) -> None:
    store = SessionStore(tmp_path / "session.jsonl")

    for i in range(100):
        store.append(SessionRecord(record_id=f"msg_{i}", kind="message", payload={"index": i}))

    # All records should be indexed
    for i in range(100):
        assert store.has(f"msg_{i}") is True
        record = store.get(f"msg_{i}")
        assert record is not None
        assert record.payload["index"] == i
