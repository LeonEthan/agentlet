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
    # Clear any existing env vars first to isolate the test
    monkeypatch.delenv("AGENTLET_API_KEY", raising=False)
    monkeypatch.delenv("AGENTLET_BASE_URL", raising=False)
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


def test_main_init_force_repairs_invalid_settings_file(tmp_path, monkeypatch) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not-json}\n", encoding="utf-8")

    # Clear environment variables to ensure test isolation
    monkeypatch.delenv("AGENTLET_API_KEY", raising=False)
    monkeypatch.delenv("AGENTLET_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

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


def test_load_settings_only_defaults_section(tmp_path) -> None:
    """Test loading settings with only defaults section (no env)."""
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "provider": "anthropic",
                    "model": "claude-3",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_settings(settings_path)

    assert loaded.provider == "anthropic"
    assert loaded.model == "claude-3"


def test_resolve_settings_defaults_agentlet_api_key_precedence(monkeypatch) -> None:
    """Test that AGENTLET_API_KEY takes precedence over OPENAI_API_KEY."""
    monkeypatch.setenv("AGENTLET_API_KEY", "agentlet-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = resolve_settings_defaults(
        AgentletSettings(api_key="file-key"),
    )

    assert resolved.api_key == "agentlet-key"


def test_resolve_settings_defaults_openai_api_key_fallback(monkeypatch) -> None:
    """Test that OPENAI_API_KEY is used when AGENTLET_API_KEY is not set."""
    monkeypatch.delenv("AGENTLET_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = resolve_settings_defaults(
        AgentletSettings(api_key="file-key"),
    )

    assert resolved.api_key == "openai-key"


def test_resolve_settings_defaults_file_api_key_fallback(monkeypatch) -> None:
    """Test that file api_key is used when no env vars are set."""
    monkeypatch.delenv("AGENTLET_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = resolve_settings_defaults(
        AgentletSettings(api_key="file-key"),
    )

    assert resolved.api_key == "file-key"


def test_resolve_settings_defaults_agentlet_base_url_precedence(monkeypatch) -> None:
    """Test that AGENTLET_BASE_URL takes precedence over OPENAI_BASE_URL."""
    monkeypatch.setenv("AGENTLET_BASE_URL", "http://agentlet.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.example/v1")

    resolved = resolve_settings_defaults(
        AgentletSettings(api_base="http://file.example/v1"),
    )

    assert resolved.api_base == "http://agentlet.example/v1"


def test_resolve_settings_defaults_openai_base_url_fallback(monkeypatch) -> None:
    """Test that OPENAI_BASE_URL is used when AGENTLET_BASE_URL is not set."""
    monkeypatch.delenv("AGENTLET_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.example/v1")

    resolved = resolve_settings_defaults(
        AgentletSettings(api_base="http://file.example/v1"),
    )

    assert resolved.api_base == "http://openai.example/v1"


def test_resolve_settings_defaults_file_base_url_fallback(monkeypatch) -> None:
    """Test that file api_base is used when no env vars are set."""
    monkeypatch.delenv("AGENTLET_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    resolved = resolve_settings_defaults(
        AgentletSettings(api_base="http://file.example/v1"),
    )

    assert resolved.api_base == "http://file.example/v1"


def test_default_settings_path_prefers_new_name(tmp_path) -> None:
    """Test that default_settings_path returns new path when neither exists."""
    path = default_settings_path(tmp_path)
    assert path.name == "settings.json"


def test_default_settings_path_uses_legacy_when_only_it_exists(tmp_path) -> None:
    """Test that legacy setting.json is used when it exists and settings.json doesn't."""
    legacy_path = tmp_path / ".agentlet" / "setting.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('{}', encoding="utf-8")

    path = default_settings_path(tmp_path)
    assert path.name == "setting.json"


def test_default_settings_path_prefers_new_when_both_exist(tmp_path) -> None:
    """Test that settings.json is preferred when both files exist."""
    new_path = tmp_path / ".agentlet" / "settings.json"
    legacy_path = tmp_path / ".agentlet" / "setting.json"
    new_path.parent.mkdir(parents=True)
    new_path.write_text('{}', encoding="utf-8")
    legacy_path.write_text('{}', encoding="utf-8")

    path = default_settings_path(tmp_path)
    assert path.name == "settings.json"


def test_load_settings_rejects_unknown_keys_in_nested_defaults(tmp_path) -> None:
    """Test that unknown keys in nested defaults section are rejected."""
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "model": "test-model",
                    "extra": "nope",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="Unsupported settings keys"):
        load_settings(settings_path)


def test_load_settings_rejects_non_object_defaults(tmp_path) -> None:
    """Test that non-object defaults section is rejected."""
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"defaults": "not-an-object"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="must be an object"):
        load_settings(settings_path)
