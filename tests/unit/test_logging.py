"""Tests for structured logging and observability."""

import json
from io import StringIO

import pytest

from agentlet.core.types import (
    LogLevel,
    StructuredLogger,
    Timer,
    get_logger,
)


@pytest.fixture
def mock_logger():
    """Create a logger that writes to a StringIO for testing."""
    logger = StructuredLogger("test", level=LogLevel.DEBUG)
    logger._output = StringIO()
    return logger


def test_logger_outputs_json_lines(mock_logger):
    """Test that logger outputs valid JSON lines."""
    mock_logger.info("test message", key="value")

    output = mock_logger._output.getvalue()
    entry = json.loads(output.strip())

    assert entry["level"] == "INFO"
    assert entry["logger"] == "test"
    assert entry["message"] == "test message"
    assert entry["context"]["key"] == "value"
    assert "timestamp" in entry


def test_logger_respects_level_filter(mock_logger):
    """Test that logger respects level filtering."""
    mock_logger.level = LogLevel.WARNING

    mock_logger.debug("debug message")
    mock_logger.info("info message")
    mock_logger.warning("warning message")

    output = mock_logger._output.getvalue()
    lines = [l for l in output.strip().split("\n") if l]

    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["message"] == "warning message"


def test_logger_exception_includes_error_info(mock_logger):
    """Test that exception method includes error details."""
    try:
        raise ValueError("test error")
    except Exception as e:
        mock_logger.exception("something failed", exc=e)

    output = mock_logger._output.getvalue()
    entry = json.loads(output.strip())

    assert entry["message"] == "something failed"
    assert entry["context"]["exception_type"] == "ValueError"
    assert entry["context"]["exception_message"] == "test error"


def test_timer_context_manager():
    """Test timer as context manager."""
    with Timer() as timer:
        pass  # Immediate exit

    assert timer.elapsed_seconds >= 0


def test_timer_manual_measurement():
    """Test timer manual measurement."""
    import time

    timer = Timer()
    time.sleep(0.01)  # Small delay

    elapsed = timer.elapsed_seconds
    assert elapsed >= 0.01

    # Timer should continue running
    assert timer.elapsed_seconds >= elapsed


def test_get_logger_factory():
    """Test get_logger factory function."""
    logger = get_logger("my_module", level=LogLevel.ERROR)

    assert logger.name == "my_module"
    assert logger.level == LogLevel.ERROR


def test_logger_skips_none_context_values(mock_logger):
    """Test that logger skips None values in context."""
    mock_logger.info("message", key="value", none_key=None)

    output = mock_logger._output.getvalue()
    entry = json.loads(output.strip())

    assert "key" in entry["context"]
    assert "none_key" not in entry["context"]
