from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

from agentlet.agent.providers.registry import ProviderConfig
from agentlet.cli import main as cli_main
from agentlet.cli.chat_app import run_chat_command
from agentlet.cli.sessions import SessionStore
from agentlet.settings import AgentletSettings, default_settings_path
from conftest import FakeProvider, FakeProviderRegistry, make_cli_args


class FakePromptSession:
    """Test double for prompt session that returns predetermined inputs."""

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)

    def prompt(self, prompt_text: str) -> str:
        if not self._inputs:
            raise EOFError
        return self._inputs.pop(0)


def test_main_chat_prints_response(monkeypatch, capsys) -> None:
    captured_configs: list[ProviderConfig] = []
    monkeypatch.setattr(
        cli_main, "ProviderRegistry", lambda: FakeProviderRegistry(captured_configs)
    )

    exit_code = cli_main.main(["chat", "hello from cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "fake response"


def test_main_chat_loads_defaults_from_user_settings(tmp_path, monkeypatch, capsys) -> None:
    captured_configs: list[ProviderConfig] = []
    monkeypatch.setattr(
        cli_main, "ProviderRegistry", lambda: FakeProviderRegistry(captured_configs)
    )
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        "{\n"
        '  "provider": "openai",\n'
        '  "model": "settings-model",\n'
        '  "api_key": "settings-key",\n'
        '  "api_base": "http://localhost:4000/v1",\n'
        '  "temperature": 0.0,\n'
        '  "max_tokens": null\n'
        "}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTLET_API_KEY", raising=False)
    monkeypatch.delenv("AGENTLET_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTLET_MODEL", raising=False)
    monkeypatch.delenv("AGENTLET_PROVIDER", raising=False)

    exit_code = cli_main.main(["chat", "hello from settings"], home_dir=tmp_path)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "fake response"
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.api_key == "settings-key"
    assert config.api_base == "http://localhost:4000/v1"
    assert config.model == "settings-model"


def test_run_chat_command_interactive_new_session(tmp_path) -> None:
    """Interactive mode should resume the previous session when requested."""
    from conftest import build_capture_console

    provider = FakeProvider()
    captured_configs: list[ProviderConfig] = []
    provider_registry = FakeProviderRegistry(capture_config=captured_configs, provider=provider)
    settings = AgentletSettings(
        provider="openai",
        model="model-a",
        api_key="test-key",
        api_base="http://localhost:4000/v1",
        temperature=0.7,
        max_tokens=64,
    )
    session_store = SessionStore(tmp_path)

    console_one, output_one = build_capture_console()
    exit_code_one = run_chat_command(
        make_cli_args(),
        settings=settings,
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=provider_registry,
        prompt_session=FakePromptSession(["first turn", "/exit"]),
        console=console_one,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    latest_session_id = session_store.load_latest_session_id()

    console_two, output_two = build_capture_console()
    exit_code_two = run_chat_command(
        make_cli_args(continue_session=True),
        settings=settings,
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=provider_registry,
        prompt_session=FakePromptSession(["second turn", "/exit"]),
        console=console_two,
        cwd=tmp_path,
        stdin_isatty=True,
    )

    loaded = session_store.load_session(latest_session_id)
    user_messages = [message.content for message in loaded.context.history if message.role == "user"]

    assert exit_code_one == 0
    assert exit_code_two == 0
    assert "echo: first turn" in output_one.getvalue()
    assert "echo: second turn" in output_two.getvalue()
    assert "resumed" in output_two.getvalue()
    assert user_messages == ["first turn", "second turn"]
    assert [config.model for config in captured_configs] == ["model-a", "model-a"]
    assert [config.api_base for config in captured_configs] == [
        "http://localhost:4000/v1",
        "http://localhost:4000/v1",
    ]
    assert [config.temperature for config in captured_configs] == [0.7, 0.7]
    assert [config.max_tokens for config in captured_configs] == [64, 64]


def test_run_chat_command_one_shot_uses_cli_overrides() -> None:
    captured_configs: list[ProviderConfig] = []
    provider_registry = FakeProviderRegistry(capture_config=captured_configs)

    exit_code = run_chat_command(
        make_cli_args(
            message="hello",
            provider="anthropic",
            model="claude-3-5-sonnet",
            api_key="override-key",
            api_base="http://override.example/v1",
            temperature=0.4,
            max_tokens=512,
        ),
        settings=AgentletSettings(
            provider="openai",
            model="gpt-5.4",
            api_key="settings-key",
            api_base="http://settings.example/v1",
            temperature=0.0,
            max_tokens=128,
        ),
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
        provider_registry=provider_registry,
        stdin_isatty=True,
    )

    assert exit_code == 0
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.name == "anthropic"
    assert config.model == "claude-3-5-sonnet"
    assert config.api_key == "override-key"
    assert config.api_base == "http://override.example/v1"
    assert config.temperature == 0.4
    assert config.max_tokens == 512
