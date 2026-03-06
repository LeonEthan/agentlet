"""Shared contract types used across core, llm, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-agnostic token accounting for one model response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        computed_total = self.input_tokens + self.output_tokens
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", computed_total)
        elif self.total_tokens != computed_total:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "TokenUsage":
        return cls(
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
        )


@dataclass(frozen=True, slots=True)
class InterruptOption:
    """One structured choice surfaced when a tool requests an interrupt."""

    value: str
    label: str
    description: str | None = None

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "value": self.value,
            "label": self.label,
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "InterruptOption":
        return cls(
            value=str(payload["value"]),
            label=str(payload["label"]),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class InterruptMetadata:
    """Structured metadata a runtime can use to pause and later resume work."""

    kind: str
    prompt: str
    request_id: str | None = None
    options: tuple[InterruptOption, ...] = field(default_factory=tuple)
    allow_free_text: bool = False
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "kind": self.kind,
            "prompt": self.prompt,
            "allow_free_text": self.allow_free_text,
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        if self.options:
            payload["options"] = [option.as_dict() for option in self.options]
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "InterruptMetadata":
        details_payload = payload.get("details", {})
        if not isinstance(details_payload, dict):
            raise ValueError("interrupt details must be a mapping")

        options_payload = payload.get("options", ())
        if not isinstance(options_payload, list | tuple):
            raise ValueError("interrupt options must be a list")

        return cls(
            kind=str(payload["kind"]),
            prompt=str(payload["prompt"]),
            request_id=(
                str(payload["request_id"])
                if payload.get("request_id") is not None
                else None
            ),
            options=tuple(
                InterruptOption.from_dict(option_payload)
                for option_payload in options_payload
            ),
            allow_free_text=bool(payload.get("allow_free_text", False)),
            details=dict(details_payload),
        )
