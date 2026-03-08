"""Append-only JSONL-backed session history."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from agentlet.core.types import JSONObject, deep_copy_json_object


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One normalized record persisted in session history.

    ``record_id`` is the store-level idempotency key. Re-appending the same
    record is a no-op, while reusing the same id for different content is an
    error.
    """

    record_id: str
    kind: str
    payload: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("session record id must not be empty")
        if not self.kind:
            raise ValueError("session record kind must not be empty")
        object.__setattr__(self, "payload", deep_copy_json_object(self.payload))

    def as_dict(self) -> JSONObject:
        """Serialize this record to a JSON-compatible mapping."""

        return {
            "id": self.record_id,
            "kind": self.kind,
            "payload": deep_copy_json_object(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "SessionRecord":
        """Build a normalized record from a JSON-compatible mapping."""

        record_id = payload.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("session record id must be a non-empty string")

        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("session record kind must be a non-empty string")

        record_payload = payload.get("payload", {})
        if not isinstance(record_payload, dict):
            raise ValueError("session record payload must be a mapping")

        return cls(
            record_id=record_id,
            kind=kind,
            payload=deep_copy_json_object(record_payload),
        )


class SessionStoreError(ValueError):
    """Base error raised for invalid session-store input."""


class SessionStoreFormatError(SessionStoreError):
    """Raised when the JSONL file contains malformed records."""


class SessionStoreConflictError(SessionStoreError):
    """Raised when one record id is reused for different content."""


class SessionStore:
    """Persist session history as an append-only JSONL file.

    Optimized with in-memory indexing for O(1) record lookups and atomic
    file writes for crash safety.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # In-memory index for O(1) lookups - maps record_id -> record
        self._index: dict[str, SessionRecord] | None = None
        # Track if file has been modified externally
        self._last_modified: float | None = None

    def ensure_exists(self) -> None:
        """Materialize the JSONL file so the runtime layout exists on disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: SessionRecord | JSONObject) -> SessionRecord:
        """Append one normalized record to the session history file.

        If a record with the same id and identical content already exists, this
        method is a no-op. Reusing an id for different content raises
        ``SessionStoreConflictError``.
        """

        normalized_record = self._normalize_record(record)
        self._append_new_records([normalized_record])
        return normalized_record

    def append_many(
        self,
        records: Iterable[SessionRecord | JSONObject],
    ) -> list[SessionRecord]:
        """Append multiple records while preserving append-only semantics."""

        normalized_records = [self._normalize_record(record) for record in records]
        if not normalized_records:
            return []

        self._append_new_records(normalized_records)
        return normalized_records

    def load(self, *, skip_malformed: bool = False) -> list[SessionRecord]:
        """Load all normalized records from disk.

        Missing files return an empty list. If ``skip_malformed`` is false,
        malformed non-empty lines raise ``SessionStoreFormatError``.

        Uses in-memory caching for repeated loads of the same file.
        """

        if not self.path.exists():
            self._index = {}
            return []

        # Check if file has been modified since last load
        current_mtime = self.path.stat().st_mtime
        if self._index is not None and self._last_modified == current_mtime:
            return list(self._index.values())

        records: list[SessionRecord] = []
        new_index: dict[str, SessionRecord] = {}

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    if skip_malformed:
                        continue
                    self._index = None
                    raise SessionStoreFormatError(
                        f"invalid JSON in {self.path} at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(payload, dict):
                    if skip_malformed:
                        continue
                    self._index = None
                    raise SessionStoreFormatError(
                        f"invalid session record in {self.path} at line {line_number}: "
                        "record must be a mapping"
                    )

                try:
                    record = SessionRecord.from_dict(payload)
                    records.append(record)
                    new_index[record.record_id] = record
                except (TypeError, ValueError) as exc:
                    if skip_malformed:
                        continue
                    self._index = None
                    raise SessionStoreFormatError(
                        f"invalid session record in {self.path} at line {line_number}: "
                        f"{exc}"
                    ) from exc

        self._index = new_index
        self._last_modified = current_mtime
        return records

    def get(self, record_id: str) -> SessionRecord | None:
        """Get a record by id in O(1) time using the in-memory index.

        Returns None if the record does not exist.
        """
        if self._index is None:
            self.load()
        assert self._index is not None
        return self._index.get(record_id)

    def has(self, record_id: str) -> bool:
        """Check if a record exists in O(1) time."""
        return self.get(record_id) is not None

    def _normalize_record(self, record: SessionRecord | JSONObject) -> SessionRecord:
        if isinstance(record, SessionRecord):
            return record
        if not isinstance(record, dict):
            raise SessionStoreError("session record must be a SessionRecord or mapping")
        try:
            return SessionRecord.from_dict(record)
        except (TypeError, ValueError) as exc:
            raise SessionStoreError(str(exc)) from exc

    def _append_new_records(self, records: list[SessionRecord]) -> None:
        """Append records using atomic file operations for crash safety."""
        # Ensure index is loaded
        if self._index is None:
            self.load()
        assert self._index is not None

        pending_records: list[SessionRecord] = []

        for record in records:
            existing_record = self._index.get(record.record_id)
            if existing_record is None:
                self._index[record.record_id] = record
                pending_records.append(record)
                continue
            if existing_record != record:
                raise SessionStoreConflictError(
                    "session record id conflict for "
                    f"{record.record_id!r}: existing record differs from append input"
                )

        if not pending_records:
            return

        self._atomic_append_records(pending_records)
        self._last_modified = self.path.stat().st_mtime

    def _atomic_append_records(self, records: list[SessionRecord]) -> None:
        """Atomically append records to the JSONL file.

        Uses a temp file and rename for crash safety.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory for atomic rename
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=self.path.stem + ".tmp",
                suffix=".jsonl",
            )

            with open(temp_fd, "w", encoding="utf-8") as temp_handle:
                # Copy existing content if file exists
                if self.path.exists():
                    with self.path.open("r", encoding="utf-8") as existing_handle:
                        for line in existing_handle:
                            temp_handle.write(line)

                # Append new records
                for record in records:
                    temp_handle.write(
                        json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True)
                    )
                    temp_handle.write("\n")

            # Atomic rename
            Path(temp_path).replace(self.path)
            temp_path = None  # Prevent cleanup in finally

        finally:
            if temp_fd is not None:
                try:
                    import os
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass
