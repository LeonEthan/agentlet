"""Parallel tool execution for improved throughput."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from agentlet.core.types import get_logger
from agentlet.tools.base import Tool, ToolResult

logger = get_logger("agentlet.parallel")


@dataclass
class ParallelExecutor:
    """Execute tools in parallel with configurable concurrency."""
    max_workers: int = 4

    def execute(
        self,
        tool_calls: list[tuple[str, Tool, dict[str, object]]],
        handler: Callable[[str, ToolResult], None] | None = None,
    ) -> dict[str, ToolResult]:
        """Execute tool calls in parallel."""
        results: dict[str, ToolResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(tool_calls)))) as executor:
            futures = {executor.submit(tc[1].execute, tc[2]): tc[0] for tc in tool_calls}
            for future in as_completed(futures):
                call_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Tool {call_id} failed", error=str(e))
                    result = ToolResult(output=f"Error: {e}", is_error=True)
                results[call_id] = result
                if handler:
                    handler(call_id, result)
        return results


__all__ = ["ParallelExecutor"]
