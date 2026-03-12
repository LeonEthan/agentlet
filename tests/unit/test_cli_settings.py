from __future__ import annotations

import json

import pytest

from agentlet.cli import main as cli_main
from agentlet.settings import (
    AgentletSettings,
    SettingsError,
    default_settings_path,
    load_settings,
    resolve_settings_defaults,
    write_settings,
)


def test_load_settings_reads_user_scoped_json(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "settings-model",
                "api_key": "settings-key",
                "api_base": "http://localhost:4000/v1",
                "temperature": 0.5,
                "max_tokens": 128,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_settings(settings_path)

    assert loaded == AgentletSettings(
        provider="openai",
        model="settings-model",
        api_key="settings-key",
        api_base="http://localhost:4000/v1",
        temperature=0.5,
        max_tokens=128,
    )


def test_load_settings_rejects_unknown_keys(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text('{"model":"settings-model","extra":"nope"}\n', encoding="utf-8")

    with pytest.raises(SettingsError, match="Unsupported settings keys"):
        load_settings(settings_path)


def test_resolve_settings_defaults_prefers_exported_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTLET_PROVIDER", "custom-provider")
    monkeypatch.setenv("AGENTLET_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env.example/v1")

    resolved = resolve_settings_defaults(
        AgentletSettings(
            provider="file-provider",
            model="file-model",
            api_key="file-key",
            api_base="http://file.example/v1",
            temperature=0.4,
            max_tokens=99,
        )
    )

    assert resolved == AgentletSettings(
        provider="custom-provider",
        model="env-model",
        api_key="env-key",
        api_base="http://env.example/v1",
        temperature=0.4,
        max_tokens=99,
    )


def test_write_settings_requires_force_to_overwrite(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    write_settings(AgentletSettings(model="first-model"), settings_path=settings_path)

    with pytest.raises(SettingsError, match="Use --force to overwrite"):
        write_settings(AgentletSettings(model="second-model"), settings_path=settings_path)


def test_main_init_writes_canonical_settings_file(tmp_path, capsys) -> None:
    exit_code = cli_main.main(
        [
            "init",
            "--provider",
            "openai",
            "--model",
            "init-model",
            "--api-key",
            "init-key",
            "--api-base",
            "http://localhost:4000/v1",
            "--temperature",
            "0.3",
            "--max-tokens",
            "256",
        ],
        home_dir=tmp_path,
    )

    captured = capsys.readouterr()
    settings_path = default_settings_path(tmp_path)

    assert exit_code == 0
    assert str(settings_path) in captured.out
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "init-model",
        "api_key": "init-key",
        "api_base": "http://localhost:4000/v1",
        "temperature": 0.3,
        "max_tokens": 256,
    }


def test_main_chat_rejects_invalid_settings_file(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["chat", "hello"], home_dir=tmp_path)

    assert exc_info.value.code == 2


def test_main_init_force_repairs_invalid_settings_file(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not-json}\n", encoding="utf-8")

    exit_code = cli_main.main(
        ["init", "--force", "--api-key", "fixed-key", "--model", "fixed-model"],
        home_dir=tmp_path,
    )

    assert exit_code == 0
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "fixed-model",
        "api_key": "fixed-key",
        "api_base": None,
        "temperature": 0.0,
        "max_tokens": None,
    }
