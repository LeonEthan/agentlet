#!/usr/bin/env python3
"""Entry point for running agentlet in Harbor benchmark container."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentlet.benchmark.headless_user_io import HeadlessUserIO
from agentlet.core.loop import CompletedTurn
from agentlet.runtime.app import build_default_runtime_app


def _write_result(output_path: Path, result: dict) -> None:
    """Write result dict to JSON file, ensuring parent directory exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run agentlet in headless benchmark mode"
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="The task instruction from benchmark",
    )
    parser.add_argument(
        "--workspace",
        default="/workspace",
        help="Working directory (Harbor standard)",
    )
    parser.add_argument(
        "--output",
        default="/workspace/.agentlet/benchmark_result.json",
        help="Path to write result JSON",
    )
    parser.add_argument(
        "--trajectory",
        default="/workspace/.agentlet/trajectory.jsonl",
        help="Path to write trajectory log",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,  # Terminal-bench allows up to 50 steps
        help="Maximum tool call iterations",
    )
    parser.add_argument(
        "--auto-approve-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-approve all operations (required for benchmark)",
    )

    args = parser.parse_args()

    # Ensure workspace exists
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)

    # Create headless UserIO with trajectory logging
    user_io = HeadlessUserIO(
        auto_approve_all=args.auto_approve_all,
        event_log_path=Path(args.trajectory),
    )

    # Build runtime with environment-backed config (AGENTLET_MODEL, etc.)
    try:
        runtime_app = build_default_runtime_app(
            user_io=user_io,
            workspace_root=workspace,
            max_iterations=args.max_iterations,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        _write_result(output_path, {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        })
        return 1

    # Run the task
    try:
        outcome = runtime_app.run_turn(current_task=args.instruction)

        # Prepare result
        result: dict = {
            "success": True,
            "outcome_type": type(outcome).__name__,
            "trajectory_summary": user_io.get_trajectory_summary(),
        }

        if isinstance(outcome, CompletedTurn) and outcome.message:
            result["assistant_message"] = outcome.message.content
            result["tool_calls"] = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in (outcome.message.tool_calls or [])
            ]

        _write_result(output_path, result)
        print(f"Result written to {args.output}")
        return 0

    except Exception as e:
        _write_result(output_path, {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        })
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        user_io.close()


if __name__ == "__main__":
    sys.exit(main())
