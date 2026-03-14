from __future__ import annotations

import asyncio
from io import StringIO

from agentlet.agent.tools.registry import ToolApprovalRequest
from agentlet.cli.approvals import InteractiveApprovalHandler


class FakePrompt:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seen_prompts: list[str] = []

    def prompt(self, prompt_text: str | None = None) -> str:
        self.seen_prompts.append(prompt_text or "")
        return self._responses.pop(0)


def test_interactive_approval_handler_approves_for_session_scope() -> None:
    handler = InteractiveApprovalHandler(prompt_input=FakePrompt(["a"]))
    request = ToolApprovalRequest(
        tool_name="web_fetch",
        scope="network",
        arguments={"url": "https://example.com"},
        summary="web_fetch https://example.com",
    )

    approved = asyncio.run(handler.approve(request))
    approved_again = asyncio.run(handler.approve(request))

    assert approved is True
    assert approved_again is True
    assert "network" in handler.approved_scopes


def test_interactive_approval_handler_rejects_without_promptable_tty() -> None:
    handler = InteractiveApprovalHandler(
        stdin=StringIO(""),
        stdout=StringIO(),
        auto_approve=False,
    )
    request = ToolApprovalRequest(
        tool_name="bash",
        scope="bash",
        arguments={"command": "pwd"},
        summary="bash pwd",
    )

    approved = asyncio.run(handler.approve(request))

    assert approved is False
