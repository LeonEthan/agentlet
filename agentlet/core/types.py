"""Shared contract types used across core, llm, and tools."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import wraps
from time import sleep
from typing import Callable, TypeVar
from uuid import uuid4

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


class LogLevel:
    """Log level constants matching Python's logging module."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class StructuredLogger:
    """Minimal structured logger for production observability.

    Outputs JSON lines to stderr for machine parsing while keeping
    human-readable messages for development.
    """

    def __init__(self, name: str, level: int = LogLevel.INFO) -> None:
        self.name = name
        self.level = level
        self._output = __import__("sys").stderr

    def _log(
        self,
        level: int,
        level_name: str,
        message: str,
        **context: JSONValue,
    ) -> None:
        if level < self.level:
            return

        import json
        from time import time

        entry: JSONObject = {
            "timestamp": time(),
            "level": level_name,
            "logger": self.name,
            "message": message,
        }
        if context:
            entry["context"] = {k: v for k, v in context.items() if v is not None}

        self._output.write(json.dumps(entry, default=str) + "\n")
        self._output.flush()

    def debug(self, message: str, **context: JSONValue) -> None:
        self._log(LogLevel.DEBUG, "DEBUG", message, **context)

    def info(self, message: str, **context: JSONValue) -> None:
        self._log(LogLevel.INFO, "INFO", message, **context)

    def warning(self, message: str, **context: JSONValue) -> None:
        self._log(LogLevel.WARNING, "WARNING", message, **context)

    def error(self, message: str, **context: JSONValue) -> None:
        self._log(LogLevel.ERROR, "ERROR", message, **context)

    def exception(self, message: str, exc: Exception | None = None, **context: JSONValue) -> None:
        ctx = dict(context)
        if exc is not None:
            ctx["exception_type"] = type(exc).__name__
            ctx["exception_message"] = str(exc)
        self._log(LogLevel.ERROR, "ERROR", message, **ctx)


def deep_copy_json(value: JSONValue) -> JSONValue:
    """Recursively copy JSON-like values while preserving plain dict/list shapes."""

    if isinstance(value, dict):
        return {
            str(key): deep_copy_json(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [deep_copy_json(item) for item in value]
    return value


def deep_copy_json_object(payload: JSONObject) -> JSONObject:
    """Recursively copy a JSON-like mapping."""

    copied = deep_copy_json(payload)
    if not isinstance(copied, dict):
        raise ValueError("payload must be a mapping")
    return copied


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
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

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
            payload["details"] = deep_copy_json_object(self.details)
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
            details=deep_copy_json_object(details_payload),
        )


class Timer:
    """Simple timer for performance measurements.

    Usage:
        with Timer() as timer:
            do_work()
        print(f"Elapsed: {timer.elapsed_seconds:.3f}s")
    """

    def __init__(self) -> None:
        from time import monotonic

        self._start: float = monotonic()
        self._end: float | None = None

    def __enter__(self) -> "Timer":
        from time import monotonic

        self._start = monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        from time import monotonic

        self._end = monotonic()

    @property
    def elapsed_seconds(self) -> float:
        from time import monotonic

        if self._end is not None:
            return self._end - self._start
        return monotonic() - self._start


def get_logger(name: str, level: int = LogLevel.INFO) -> StructuredLogger:
    """Factory function to create a structured logger."""
    return StructuredLogger(name, level)


class RequestContext:
    """Thread-safe request context for tracing.

    Usage:
        with RequestContext.set_current("req-123"):
            # All operations within this context have access to request_id
            process_request()
    """

    _local = threading.local()

    @classmethod
    def set_current(cls, request_id: str | None) -> "RequestContext":
        """Set the current request ID for this context."""
        ctx = cls()
        ctx.request_id = request_id
        cls._local.current = ctx
        return ctx

    @classmethod
    def get_current(cls) -> "RequestContext | None":
        """Get the current request context."""
        return getattr(cls._local, "current", None)

    @classmethod
    def get_request_id(cls) -> str | None:
        """Get the current request ID."""
        ctx = cls.get_current()
        return ctx.request_id if ctx else None

    @classmethod
    def generate_id(cls) -> str:
        """Generate a new unique request ID."""
        return f"req_{uuid4().hex[:16]}"

    def __enter__(self) -> "RequestContext":
        return self

    def __exit__(self, *args: object) -> None:
        """Clear context on exit."""
        type(self)._local.current = None


T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff.

    Only retries on specified exception types. Non-retryable exceptions
    are raised immediately.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt >= max_retries:
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("retry loop exited without result or exception")

        return wrapper

    return decorator
