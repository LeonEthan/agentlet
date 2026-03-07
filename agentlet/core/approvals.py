"""Centralized approval decisions for tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from agentlet.tools.base import (
    ApprovalCategory,
    Tool,
    ToolDefinition,
    VALID_APPROVAL_CATEGORIES,
)
from agentlet.tools.registry import ToolRegistry

ApprovalMode = Literal["allow", "require_approval"]
VALID_APPROVAL_MODES = {"allow", "require_approval"}
DEFAULT_CATEGORY_APPROVAL_MODES = MappingProxyType(
    {
        "read_only": "allow",
        "mutating": "require_approval",
        "exec": "require_approval",
        "external_or_interrupt": "require_approval",
    }
)


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Structured approval result returned to the runtime layer."""

    tool_name: str
    approval_category: ApprovalCategory
    mode: ApprovalMode
    reason: str

    @property
    def requires_approval(self) -> bool:
        return self.mode == "require_approval"


class ApprovalPolicy:
    """Map tool risk categories to runtime-facing approval decisions."""

    def __init__(
        self,
        category_modes: Mapping[ApprovalCategory, ApprovalMode] | None = None,
    ) -> None:
        modes = dict(DEFAULT_CATEGORY_APPROVAL_MODES)
        if category_modes is not None:
            for category, mode in category_modes.items():
                _validate_approval_category(category)
                _validate_approval_mode(mode)
                modes[category] = mode
        self._category_modes = MappingProxyType(modes)

    def decision_for_tool_name(
        self,
        tool_name: str,
        registry: ToolRegistry,
    ) -> ApprovalDecision:
        """Resolve a tool by name and return its approval decision."""

        return self.decision_for_definition(registry.definition(tool_name))

    def decision_for_tool(self, tool: Tool) -> ApprovalDecision:
        """Return the approval decision for one tool instance."""

        return self.decision_for_definition(tool.definition)

    def decision_for_definition(self, definition: ToolDefinition) -> ApprovalDecision:
        """Return the approval decision for one tool definition."""

        _validate_approval_category(definition.approval_category)
        mode = _mode_for_definition(definition, self._category_modes)
        return ApprovalDecision(
            tool_name=definition.name,
            approval_category=definition.approval_category,
            mode=mode,
            reason=_build_reason(definition.approval_category, mode),
        )

    def mode_for_category(self, category: ApprovalCategory) -> ApprovalMode:
        """Return the configured decision mode for one approval category."""

        _validate_approval_category(category)
        mode = self._category_modes[category]
        _validate_approval_mode(mode)
        return mode

    def category_modes(self) -> dict[ApprovalCategory, ApprovalMode]:
        """Return a copy of the current category-to-mode mapping."""

        return dict(self._category_modes)


def _validate_approval_category(category: str) -> None:
    if category not in VALID_APPROVAL_CATEGORIES:
        raise ValueError(f"unsupported approval category: {category}")


def _validate_approval_mode(mode: str) -> None:
    if mode not in VALID_APPROVAL_MODES:
        raise ValueError(f"unsupported approval mode: {mode}")


def _build_reason(category: ApprovalCategory, mode: ApprovalMode) -> str:
    if mode == "allow":
        return f"{category} tools may run without runtime approval."
    return f"{category} tools require runtime approval before execution."


def _mode_for_definition(
    definition: ToolDefinition,
    category_modes: Mapping[ApprovalCategory, ApprovalMode],
) -> ApprovalMode:
    if definition.name == "AskUserQuestion":
        return "allow"
    mode = category_modes[definition.approval_category]
    _validate_approval_mode(mode)
    return mode
