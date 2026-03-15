from __future__ import annotations

import asyncio
from io import StringIO

from agentlet.agent.tools.registry import ToolApprovalRequest
from agentlet.cli.approvals import ApprovalPromptClosed
from agentlet.cli.approvals import InteractiveApprovalHandler


class FakePrompt:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seen_prompts: list[str] = []

    def prompt(self, prompt_text: str | None = None) -> str:
        self.seen_prompts.append(prompt_text or "")
        return self._responses.pop(0)


class AsyncOnlyPrompt:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seen_prompts: list[str] = []
        self.sync_calls = 0
        self.async_calls = 0

    def prompt(self, prompt_text: str | None = None) -> str:
        self.sync_calls += 1
        raise AssertionError("prompt() should not be used when prompt_async() exists")

    async def prompt_async(self, prompt_text: str | None = None) -> str:
        self.async_calls += 1
        self.seen_prompts.append(prompt_text or "")
        return self._responses.pop(0)


class FakeTTY(StringIO):
    def __init__(self, value: str = "", *, is_tty: bool) -> None:
        super().__init__(value)
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


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


def test_interactive_approval_handler_prefers_async_prompt_when_available() -> None:
    prompt = AsyncOnlyPrompt(["y"])
    handler = InteractiveApprovalHandler(prompt_input=prompt)
    request = ToolApprovalRequest(
        tool_name="web_search",
        scope="network",
        arguments={"query": "Changsha weather forecast next week"},
        summary="web_search Changsha weather forecast next week",
    )

    approved = asyncio.run(handler.approve(request))

    assert approved is True
    assert prompt.async_calls == 1
    assert prompt.sync_calls == 0
    assert prompt.seen_prompts == [
        "Approve web_search Changsha weather forecast next week? "
        "[y]es/[n]o/[a]ll-for-session: "
    ]


def test_interactive_approval_handler_wraps_prompt_eof() -> None:
    class EOFPrompt:
        async def prompt_async(self, prompt_text: str | None = None) -> str:
            raise EOFError

    handler = InteractiveApprovalHandler(prompt_input=EOFPrompt())
    request = ToolApprovalRequest(
        tool_name="web_search",
        scope="network",
        arguments={"query": "Changsha weather forecast next week"},
        summary="web_search Changsha weather forecast next week",
    )

    try:
        asyncio.run(handler.approve(request))
    except ApprovalPromptClosed:
        pass
    else:
        raise AssertionError("approval prompt EOF should raise ApprovalPromptClosed")


def test_interactive_approval_handler_cannot_prompt_with_redirected_stdout() -> None:
    handler = InteractiveApprovalHandler(
        stdin=FakeTTY("y\n", is_tty=True),
        stdout=FakeTTY("", is_tty=False),
        auto_approve=False,
    )

    assert handler.can_prompt() is False


def test_interactive_approval_handler_restores_prompt_session_message() -> None:
    class StickyPrompt:
        def __init__(self, responses: list[str]) -> None:
            self._responses = list(responses)
            self.message = "› "
            self.seen_messages: list[str] = []

        async def prompt_async(self, prompt_text: str | None = None) -> str:
            if prompt_text is not None:
                self.message = prompt_text
            self.seen_messages.append(self.message)
            return self._responses.pop(0)

    prompt = StickyPrompt(["y"])
    handler = InteractiveApprovalHandler(prompt_input=prompt)
    request = ToolApprovalRequest(
        tool_name="web_search",
        scope="network",
        arguments={"query": "Changsha weather forecast next week"},
        summary="web_search Changsha weather forecast next week",
    )

    approved = asyncio.run(handler.approve(request))

    assert approved is True
    assert prompt.seen_messages == [
        "Approve web_search Changsha weather forecast next week? "
        "[y]es/[n]o/[a]ll-for-session: "
    ]
    assert prompt.message == "› "
