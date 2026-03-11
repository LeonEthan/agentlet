from __future__ import annotations

from agentlet.agent.providers.registry import LLMResponse
from agentlet.cli import main as cli_main


class FakeProvider:
    async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
        return LLMResponse(content="cli ok", finish_reason="stop")


class FakeProviderRegistry:
    last_config = None

    def create(self, config):
        type(self).last_config = config
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


def test_main_chat_loads_defaults_from_project_dotenv(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main, "ProviderRegistry", FakeProviderRegistry)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTLET_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "OPENAI_BASE_URL=http://localhost:4000/v1\n"
        "AGENTLET_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["chat", "hello from dotenv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "cli ok"
    assert FakeProviderRegistry.last_config is not None
    assert FakeProviderRegistry.last_config.api_key == "dotenv-key"
    assert FakeProviderRegistry.last_config.api_base == "http://localhost:4000/v1"
    assert FakeProviderRegistry.last_config.model == "dotenv-model"
