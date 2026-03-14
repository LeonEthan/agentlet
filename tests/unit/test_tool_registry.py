from __future__ import annotations

import asyncio

import pytest

from agentlet.agent.context import ToolCall
from agentlet.agent.tools.builtins import build_default_registry
from agentlet.agent.tools.policy import ToolPolicy, ToolRuntimeConfig
from agentlet.agent.tools.registry import ToolExecutionError, ToolRegistry, build_tool_result_content
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
