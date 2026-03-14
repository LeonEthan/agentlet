from __future__ import annotations

"""Interactive approval helpers for unsafe tool execution."""

from dataclasses import dataclass, field
from typing import Protocol, TextIO

from agentlet.agent.tools.registry import ToolApprovalRequest


class ApprovalPrompt(Protocol):
    """Prompt interface shared by prompt_toolkit sessions and test doubles."""

    def prompt(self, prompt_text: str | None = None) -> str: ...


@dataclass
class InteractiveApprovalHandler:
    """Session-scoped approval handler for write, bash, and network actions."""

    prompt_input: ApprovalPrompt | None = None
    stdin: TextIO | None = None
    stdout: TextIO | None = None
    auto_approve: bool = False
    approved_scopes: set[str] = field(default_factory=set)

    async def approve(self, request: ToolApprovalRequest) -> bool:
        if self.auto_approve or request.scope in self.approved_scopes:
            return True
        if not self.can_prompt():
            return False

        prompt_text = (
            f"Approve {request.summary}? "
            "[y]es/[n]o/[a]ll-for-session: "
        )
        while True:
            response = self._prompt(prompt_text).strip().lower()
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no", ""}:
                return False
            if response in {"a", "all"}:
                self.approved_scopes.add(request.scope)
                return True

    def can_prompt(self) -> bool:
        if self.prompt_input is not None:
            return True
        if self.stdin is None or self.stdout is None:
            return False
        isatty = getattr(self.stdin, "isatty", None)
        return bool(isatty and isatty())

    def _prompt(self, prompt_text: str) -> str:
        if self.prompt_input is not None:
            return self.prompt_input.prompt(prompt_text)
        assert self.stdin is not None
        assert self.stdout is not None
        self.stdout.write(prompt_text)
        self.stdout.flush()
        return self.stdin.readline()
