"""Append-only JSONL-backed session history."""

from __future__ import annotations

import json
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
    """Persist session history as an append-only JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

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
        """

        if not self.path.exists():
            return []

        records: list[SessionRecord] = []
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
                    raise SessionStoreFormatError(
                        f"invalid JSON in {self.path} at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(payload, dict):
                    if skip_malformed:
                        continue
                    raise SessionStoreFormatError(
                        f"invalid session record in {self.path} at line {line_number}: "
                        "record must be a mapping"
                    )

                try:
                    records.append(SessionRecord.from_dict(payload))
                except (TypeError, ValueError) as exc:
                    if skip_malformed:
                        continue
                    raise SessionStoreFormatError(
                        f"invalid session record in {self.path} at line {line_number}: "
                        f"{exc}"
                    ) from exc

        return records

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
        existing_records = self.load()
        existing_by_id = {record.record_id: record for record in existing_records}
        pending_records: list[SessionRecord] = []

        for record in records:
            existing_record = existing_by_id.get(record.record_id)
            if existing_record is None:
                existing_by_id[record.record_id] = record
                pending_records.append(record)
                continue
            if existing_record != record:
                raise SessionStoreConflictError(
                    "session record id conflict for "
                    f"{record.record_id!r}: existing record differs from append input"
                )

        if not pending_records:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in pending_records:
                handle.write(
                    json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True)
                )
                handle.write("\n")
