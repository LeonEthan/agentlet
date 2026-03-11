from __future__ import annotations

from agentlet.agent.providers.registry import LLMResponse
from agentlet.cli import main as cli_main


class FakeProvider:
    async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
        return LLMResponse(content="cli ok", finish_reason="stop")


class FakeProviderRegistry:
    def create(self, config):
        return FakeProvider()


def test_main_chat_prints_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main, "ProviderRegistry", FakeProviderRegistry)

    exit_code = cli_main.main(["chat", "hello from cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "cli ok"


def test_main_chat_rejects_whitespace_only_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main, "ProviderRegistry", FakeProviderRegistry)

    try:
        cli_main.main(["chat", "   "])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected the CLI to exit with a parser error.")

    captured = capsys.readouterr()
    assert "A message is required via argv or stdin." in captured.err
