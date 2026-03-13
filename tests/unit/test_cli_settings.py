from __future__ import annotations

import json

import pytest

from agentlet.cli import main as cli_main
from agentlet.settings import (
    AgentletSettings,
    SettingsError,
    canonical_settings_path,
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


def test_resolve_settings_defaults_uses_stored_values_and_built_ins() -> None:
    resolved = resolve_settings_defaults(
        AgentletSettings(
            provider=None,
            model=None,
            api_key="file-key",
            api_base="http://file.example/v1",
            temperature=None,
            max_tokens=99,
        )
    )

    assert resolved == AgentletSettings(
        provider="openai",
        model="gpt-5.4",
        api_key="file-key",
        api_base="http://file.example/v1",
        temperature=0.0,
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
        "api_key": None,
        "api_base": None,
        "temperature": 0.3,
        "max_tokens": 256,
        "allow_write": None,
        "allow_bash": None,
        "allow_network": None,
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
        ["init", "--force", "--model", "fixed-model"],
        home_dir=tmp_path,
    )

    assert exit_code == 0
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "fixed-model",
        "api_key": None,
        "api_base": None,
        "temperature": 0.0,
        "max_tokens": None,
        "allow_write": None,
        "allow_bash": None,
        "allow_network": None,
    }


def test_main_init_force_migrates_legacy_settings_filename(tmp_path) -> None:
    legacy_path = tmp_path / ".agentlet" / "setting.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "legacy-model",
                "api_key": "legacy-key",
                "api_base": "http://legacy.example/v1",
                "temperature": 0.2,
                "max_tokens": 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(
        ["init", "--force", "--model", "fixed-model"],
        home_dir=tmp_path,
    )

    new_path = canonical_settings_path(tmp_path)

    assert exit_code == 0
    assert new_path.exists()
    assert default_settings_path(tmp_path) == new_path
    assert json.loads(new_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "fixed-model",
        "api_key": "legacy-key",
        "api_base": "http://legacy.example/v1",
        "temperature": 0.2,
        "max_tokens": 64,
        "allow_write": None,
        "allow_bash": None,
        "allow_network": None,
    }


def test_main_init_force_preserves_existing_sensitive_settings(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "old-model",
                "api_key": "stored-key",
                "api_base": "http://stored.example/v1",
                "temperature": 0.1,
                "max_tokens": 32,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(
        ["init", "--force", "--model", "new-model", "--temperature", "0.8"],
        home_dir=tmp_path,
    )

    assert exit_code == 0
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "new-model",
        "api_key": "stored-key",
        "api_base": "http://stored.example/v1",
        "temperature": 0.8,
        "max_tokens": 32,
        "allow_write": None,
        "allow_bash": None,
        "allow_network": None,
    }


def test_main_init_force_preserves_existing_tool_policy_settings(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "old-model",
                "api_key": "stored-key",
                "api_base": "http://stored.example/v1",
                "temperature": 0.1,
                "max_tokens": 32,
                "allow_write": False,
                "allow_bash": True,
                "allow_network": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(
        ["init", "--force", "--model", "new-model"],
        home_dir=tmp_path,
    )

    assert exit_code == 0
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "new-model",
        "api_key": "stored-key",
        "api_base": "http://stored.example/v1",
        "temperature": 0.1,
        "max_tokens": 32,
        "allow_write": False,
        "allow_bash": True,
        "allow_network": False,
    }


def test_main_init_force_rejects_provider_change_with_stored_sensitive_settings(tmp_path) -> None:
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "old-model",
                "api_key": "stored-key",
                "api_base": "http://stored.example/v1",
                "temperature": 0.1,
                "max_tokens": 32,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["init", "--force", "--provider", "anthropic", "--model", "claude-3-5-sonnet"],
            home_dir=tmp_path,
        )

    assert exc_info.value.code == 2
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "provider": "openai",
        "model": "old-model",
        "api_key": "stored-key",
        "api_base": "http://stored.example/v1",
        "temperature": 0.1,
        "max_tokens": 32,
    }


def test_load_settings_only_defaults_section(tmp_path) -> None:
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


def test_default_settings_path_prefers_new_name(tmp_path) -> None:
    path = default_settings_path(tmp_path)
    assert path.name == "settings.json"


def test_default_settings_path_uses_legacy_when_only_it_exists(tmp_path) -> None:
    legacy_path = tmp_path / ".agentlet" / "setting.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("{}", encoding="utf-8")

    path = default_settings_path(tmp_path)
    assert path.name == "setting.json"


def test_default_settings_path_prefers_new_when_both_exist(tmp_path) -> None:
    new_path = tmp_path / ".agentlet" / "settings.json"
    legacy_path = tmp_path / ".agentlet" / "setting.json"
    new_path.parent.mkdir(parents=True)
    new_path.write_text("{}", encoding="utf-8")
    legacy_path.write_text("{}", encoding="utf-8")

    path = default_settings_path(tmp_path)
    assert path.name == "settings.json"


def test_load_settings_rejects_unknown_keys_in_nested_defaults(tmp_path) -> None:
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
    settings_path = default_settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"defaults": "not-an-object"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="must be an object"):
        load_settings(settings_path)
