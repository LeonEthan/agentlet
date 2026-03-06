from __future__ import annotations

import pytest

from agentlet.core.approvals import ApprovalPolicy
from agentlet.tools.base import ToolDefinition, ToolResult
from agentlet.tools.registry import (
    BUILT_IN_TOOL_NAMES,
    DuplicateToolError,
    ToolRegistry,
    UnknownToolError,
    builtin_tool_category,
    is_builtin_tool_name,
)


class FakeTool:
    def __init__(self, definition: ToolDefinition) -> None:
        self.definition = definition

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(output=str(arguments))


def _definition(name: str, approval_category: str = "read_only") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object"},
        approval_category=approval_category,  # type: ignore[arg-type]
    )


def test_registry_registers_and_resolves_tools_by_name() -> None:
    read_tool = FakeTool(_definition("Read", "read_only"))
    write_tool = FakeTool(_definition("Write", "mutating"))
    registry = ToolRegistry([read_tool, write_tool])

    assert "Read" in registry
    assert registry.get("Read") is read_tool
    assert registry.resolve("Write") is write_tool
    assert registry.definition("Write") == write_tool.definition
    assert registry.definitions() == (read_tool.definition, write_tool.definition)


def test_registry_rejects_duplicate_tool_registration() -> None:
    registry = ToolRegistry([FakeTool(_definition("Read", "read_only"))])

    with pytest.raises(DuplicateToolError, match="tool already registered: Read"):
        registry.register(FakeTool(_definition("Read", "read_only")))


def test_registry_reports_unknown_tools() -> None:
    registry = ToolRegistry()

    assert registry.get("Missing") is None
    with pytest.raises(UnknownToolError, match="unknown tool: Missing"):
        registry.resolve("Missing")


def test_registry_enforces_builtin_category_mapping() -> None:
    with pytest.raises(ValueError, match="built-in tool approval category mismatch"):
        ToolRegistry([FakeTool(_definition("Read", "mutating"))])


def test_builtin_tool_names_and_categories_match_architecture() -> None:
    assert BUILT_IN_TOOL_NAMES == (
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "AskUserQuestion",
    )
    assert is_builtin_tool_name("Read") is True
    assert is_builtin_tool_name("CustomTool") is False
    assert builtin_tool_category("WebFetch") == "external_or_interrupt"
    assert builtin_tool_category("CustomTool") is None


@pytest.mark.parametrize(
    ("tool_name", "approval_category", "expected_mode"),
    [
        ("Read", "read_only", "allow"),
        ("Write", "mutating", "require_approval"),
        ("Bash", "exec", "require_approval"),
        ("WebFetch", "external_or_interrupt", "require_approval"),
    ],
)
def test_approval_policy_decision_matrix(
    tool_name: str,
    approval_category: str,
    expected_mode: str,
) -> None:
    policy = ApprovalPolicy()
    definition = _definition(tool_name, approval_category)

    decision = policy.decision_for_definition(definition)

    assert decision.tool_name == tool_name
    assert decision.approval_category == approval_category
    assert decision.mode == expected_mode
    assert decision.requires_approval is (expected_mode == "require_approval")


def test_approval_policy_can_resolve_decision_from_registry() -> None:
    registry = ToolRegistry([FakeTool(_definition("Read", "read_only"))])
    policy = ApprovalPolicy()

    decision = policy.decision_for_tool_name("Read", registry)

    assert decision.mode == "allow"
    assert decision.reason == "read_only tools may run without runtime approval."


def test_approval_policy_accepts_category_overrides() -> None:
    policy = ApprovalPolicy({"external_or_interrupt": "allow"})

    decision = policy.decision_for_definition(
        _definition("AskUserQuestion", "external_or_interrupt")
    )

    assert decision.mode == "allow"
    assert decision.requires_approval is False


def test_approval_policy_rejects_invalid_override_mode() -> None:
    with pytest.raises(ValueError, match="unsupported approval mode"):
        ApprovalPolicy({"read_only": "read_only"})  # type: ignore[arg-type]
