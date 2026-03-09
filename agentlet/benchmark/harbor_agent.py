"""Harbor BaseInstalledAgent implementation for agentlet."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harbor.context import AgentContext
    from harbor.types import ExecInput


class AgentletInstalledAgent:
    """Agentlet as a Harbor installed agent for Terminal-Bench.

    This class implements the Harbor BaseInstalledAgent protocol for
    running agentlet in Terminal-Bench 2.0 benchmark containers.

    Note: Requires harbor-framework to be installed.
    """

    @property
    def _install_agent_template_path(self) -> Path:
        """Return path to installation script template."""
        return Path(__file__).parent / "templates" / "install_agent.sh.j2"

    def create_run_agent_commands(self, instruction: str) -> list["ExecInput"]:
        """Create command to run agentlet benchmark runner.

        Args:
            instruction: The task instruction from Terminal-Bench

        Returns:
            List of ExecInput commands to execute
        """
        # Import here to avoid dependency when harbor is not installed
        from harbor.types import ExecInput

        # Use shlex.quote for safe shell escaping
        escaped_instruction = shlex.quote(instruction)

        return [
            ExecInput(
                command=(
                    f"python /opt/agentlet/agentlet/benchmark/benchmark_runner.py "
                    f"--instruction {escaped_instruction} "
                    f"--workspace /workspace "
                    f"--output /workspace/.agentlet/benchmark_result.json "
                    f"--trajectory /workspace/.agentlet/trajectory.jsonl "
                    f"--max-iterations 50 "
                    f"--auto-approve-all"
                ),
                cwd="/workspace",
                env={
                    # Pass through LLM configuration from host
                    "AGENTLET_MODEL": "{{ env('AGENTLET_MODEL') }}",
                    "AGENTLET_API_KEY": "{{ env('AGENTLET_API_KEY') }}",
                    "AGENTLET_BASE_URL": "{{ env('AGENTLET_BASE_URL', '') }}",
                    "AGENTLET_PROVIDER": "{{ env('AGENTLET_PROVIDER', 'openai_like') }}",
                },
                timeout_sec=300,  # 5 minutes per task
            )
        ]

    def populate_context_post_run(self, context: "AgentContext") -> None:
        """Parse trajectory and results into Harbor context.

        Args:
            context: Harbor AgentContext to populate with results
        """
        result_path = Path("/workspace/.agentlet/benchmark_result.json")
        trajectory_path = Path("/workspace/.agentlet/trajectory.jsonl")

        # Load structured result
        try:
            with open(result_path) as f:
                result = json.load(f)
            context.add_output(result.get("assistant_message", ""))
            context.add_trajectory("benchmark_result", result)
            for tc in result.get("tool_calls", []):
                context.add_trajectory("tool_call", tc)
        except FileNotFoundError:
            pass

        # Load raw event trajectory
        try:
            with open(trajectory_path) as f:
                events = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
            context.add_trajectory("raw_events", events)
        except FileNotFoundError:
            pass

        # Add session history if available
        session_path = Path("/workspace/.agentlet/session.jsonl")
        try:
            with open(session_path) as f:
                context.add_trajectory("session_history", f.read())
        except FileNotFoundError:
            pass


def create_installed_agent() -> AgentletInstalledAgent:
    """Factory function to create an AgentletInstalledAgent instance.

    This can be used as the entry point for Harbor:
        --agent-import-path "agentlet.benchmark.harbor_agent:create_installed_agent"
    """
    return AgentletInstalledAgent()
