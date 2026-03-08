"""Tests for parallel tool execution."""

from __future__ import annotations

import time

import pytest

from agentlet.core.parallel import ParallelExecutor
from agentlet.tools.base import Tool, ToolDefinition, ToolResult


class SlowTool(Tool):
    def __init__(self, delay: float):
        self.delay = delay
        self._def = ToolDefinition(
            name=f"slow_{delay}", description="A slow tool",
            input_schema={"type": "object"}, approval_category="read_only"
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._def

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        time.sleep(self.delay)
        return ToolResult(output=f"Completed after {self.delay}s")


class ErrorTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="error_tool", description="Always errors",
            input_schema={"type": "object"}, approval_category="read_only"
        )

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        raise RuntimeError("Intentional error")


def test_parallel_executor_runs_concurrently():
    """Test that tools execute in parallel (faster than sequential)."""
    executor = ParallelExecutor(max_workers=3)

    # Three 0.1s tasks should take ~0.1s in parallel, not 0.3s
    tool_calls = [
        ("call_1", SlowTool(0.1), {}),
        ("call_2", SlowTool(0.1), {}),
        ("call_3", SlowTool(0.1), {}),
    ]

    start = time.monotonic()
    results = executor.execute(tool_calls)
    elapsed = time.monotonic() - start

    assert len(results) == 3
    assert all("Completed" in r.output for r in results.values())
    assert elapsed < 0.25  # Should be much faster than 0.3s


def test_parallel_executor_handles_errors():
    """Test that errors are captured gracefully."""
    executor = ParallelExecutor()

    tool_calls = [
        ("ok", SlowTool(0.01), {}),
        ("err", ErrorTool(), {}),
    ]

    results = executor.execute(tool_calls)

    assert results["ok"].is_error is False
    assert results["err"].is_error is True
    assert "Intentional error" in results["err"].output


def test_parallel_executor_with_handler():
    """Test callback handler is called for each result."""
    executor = ParallelExecutor()
    handler_calls = []

    def handler(call_id: str, result: ToolResult) -> None:
        handler_calls.append((call_id, result.output))

    tool_calls = [
        ("a", SlowTool(0.01), {}),
        ("b", SlowTool(0.02), {}),
    ]

    executor.execute(tool_calls, handler=handler)

    assert len(handler_calls) == 2
    assert {c[0] for c in handler_calls} == {"a", "b"}


def test_parallel_executor_empty():
    """Test empty tool calls list."""
    executor = ParallelExecutor()
    results = executor.execute([])
    assert results == {}


def test_parallel_executor_single():
    """Test single tool call works."""
    executor = ParallelExecutor()
    results = executor.execute([("solo", SlowTool(0.01), {})])

    assert len(results) == 1
    assert "solo" in results
