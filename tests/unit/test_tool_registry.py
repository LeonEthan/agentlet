from __future__ import annotations

import asyncio

import pytest

from agentlet.agent.context import ToolCall
from agentlet.agent.tools.builtins import build_default_registry
from agentlet.agent.tools.policy import ToolPolicy, ToolRuntimeConfig
from agentlet.agent.tools.registry import (
    ToolApprovalRequest,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
    build_tool_result_content,
)
from conftest import EchoTool


def test_tool_registry_rejects_invalid_json_arguments() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ToolExecutionError, match="Invalid JSON"):
        asyncio.run(
            registry.execute(
                ToolCall(id="call-1", name="echo", arguments_json="{not-json}")
            )
        )


def test_tool_registry_rejects_non_object_arguments() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ToolExecutionError, match="must decode to an object"):
        asyncio.run(
            registry.execute(
                ToolCall(id="call-1", name="echo", arguments_json='["not","an","object"]')
            )
        )


def test_build_default_registry_hides_policy_denied_tools(tmp_path) -> None:
    registry = build_default_registry(
        ToolPolicy(allow_network=False, allow_write=False, allow_bash=False),
        ToolRuntimeConfig(cwd=tmp_path),
    )

    assert registry.get_tool_names() == ["read", "glob", "grep"]


def test_build_tool_result_content_preserves_non_ascii_text() -> None:
    content = build_tool_result_content({"content": "你好 🌍"})

    assert "你好 🌍" in content
    assert "\\u" not in content


def test_tool_registry_requests_approval_for_unsafe_tools() -> None:
    seen_requests: list[ToolApprovalRequest] = []

    class FakeBashTool:
        spec = ToolSpec(
            name="bash",
            description="fake bash",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        )

        async def execute(self, arguments: dict[str, str]) -> str:
            return arguments["command"]

    class Approver:
        async def approve(self, request: ToolApprovalRequest) -> bool:
            seen_requests.append(request)
            return True

    registry = ToolRegistry([FakeBashTool()], approval_handler=Approver())

    result = asyncio.run(
        registry.execute(
            ToolCall(id="call-1", name="bash", arguments_json='{"command":"pwd"}')
        )
    )

    assert result.content == "pwd"
    assert len(seen_requests) == 1
    assert seen_requests[0].scope == "bash"
    assert seen_requests[0].summary == "bash pwd"


def test_tool_registry_rejects_unsafe_tool_when_approval_denied() -> None:
    class FakeWriteTool:
        spec = ToolSpec(
            name="write",
            description="fake write",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        async def execute(self, arguments: dict[str, str]) -> str:
            raise AssertionError("should not execute when rejected")

    class Rejector:
        async def approve(self, request: ToolApprovalRequest) -> bool:
            return False

    registry = ToolRegistry([FakeWriteTool()], approval_handler=Rejector())

    with pytest.raises(ToolExecutionError, match="was not approved"):
        asyncio.run(
            registry.execute(
                ToolCall(id="call-1", name="write", arguments_json='{"path":"notes.txt"}')
            )
        )
