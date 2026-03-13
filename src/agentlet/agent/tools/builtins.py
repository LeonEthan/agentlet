from __future__ import annotations

"""Factory for building the default tool registry from policy and runtime settings."""

from agentlet.agent.tools.bash import BashTool
from agentlet.agent.tools.local_fs import EditTool, GlobTool, GrepTool, ReadTool, WriteTool
from agentlet.agent.tools.policy import ToolPolicy, ToolRuntimeConfig
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.agent.tools.web import WebFetchTool, WebSearchTool


def build_default_registry(
    policy: ToolPolicy,
    runtime: ToolRuntimeConfig,
) -> ToolRegistry:
    """Build registry with enabled tools based on policy.

    Tools enabled by default:
    - Read, Glob, Grep: always enabled (safe read-only)
    - WebSearch, WebFetch: enabled when network is allowed

    Tools gated by policy:
    - Write, Edit: enabled when allow_write is True
    - Bash: enabled when allow_bash is True
    """
    tools = []

    # Always-enabled read-only filesystem tools
    tools.append(ReadTool(runtime))
    tools.append(GlobTool(runtime))
    tools.append(GrepTool(runtime))

    # Network tools (gated by network policy)
    if policy.allow_network:
        tools.append(WebSearchTool(runtime))
        tools.append(WebFetchTool(runtime))

    # Mutation tools (gated by write policy)
    if policy.allow_write:
        tools.append(WriteTool(runtime))
        tools.append(EditTool(runtime))

    # Bash tool (gated by bash policy)
    if policy.allow_bash:
        tools.append(BashTool(runtime))

    return ToolRegistry(tools)
