from __future__ import annotations

from typing import Any

from agentlet.agent.providers.registry import LLMResponse, ProviderConfig
from agentlet.cli import main as cli_main


class FakeProvider:
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="cli ok", finish_reason="stop")


class FakeProviderRegistry:
    def __init__(self, capture_config: list[ProviderConfig] | None = None) -> None:
        self._capture_config = capture_config

    def create(self, config: ProviderConfig) -> FakeProvider:
        if self._capture_config is not None:
            self._capture_config.append(config)
        return FakeProvider()


def test_main_chat_prints_response(monkeypatch, capsys) -> None:
    captured_configs: list[ProviderConfig] = []
    monkeypatch.setattr(cli_main, "ProviderRegistry", lambda: FakeProviderRegistry(captured_configs))

    exit_code = cli_main.main(["chat", "hello from cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "cli ok"


def test_main_chat_rejects_whitespace_only_message(monkeypatch, capsys) -> None:
    captured_configs: list[ProviderConfig] = []
    monkeypatch.setattr(cli_main, "ProviderRegistry", lambda: FakeProviderRegistry(captured_configs))

    try:
        cli_main.main(["chat", "   "])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected the CLI to exit with a parser error.")

    captured = capsys.readouterr()
    assert "A message is required via argv or stdin." in captured.err


def test_main_chat_loads_defaults_from_project_dotenv(tmp_path, monkeypatch, capsys) -> None:
    captured_configs: list[ProviderConfig] = []
    monkeypatch.setattr(cli_main, "ProviderRegistry", lambda: FakeProviderRegistry(captured_configs))
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
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.api_key == "dotenv-key"
    assert config.api_base == "http://localhost:4000/v1"
    assert config.model == "dotenv-model"
