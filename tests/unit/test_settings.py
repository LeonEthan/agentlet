"""Unit tests for settings module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentlet.config import SettingsLoader, UserSettings, apply_env_from_settings, load_settings


class TestUserSettings:
    """Tests for UserSettings dataclass."""

    def test_default_settings_is_empty(self) -> None:
        """UserSettings defaults to empty dictionaries."""
        settings = UserSettings()
        assert settings.env == {}
        assert settings.defaults == {}

    def test_settings_with_values(self) -> None:
        """UserSettings can be created with custom values."""
        settings = UserSettings(
            env={"AGENTLET_MODEL": "gpt-4"},
            defaults={"max_iterations": 10},
        )
        assert settings.env == {"AGENTLET_MODEL": "gpt-4"}
        assert settings.defaults == {"max_iterations": 10}

    def test_settings_is_frozen(self) -> None:
        """UserSettings is immutable."""
        settings = UserSettings()
        with pytest.raises(AttributeError):
            settings.env = {"FOO": "bar"}


class TestSettingsLoaderLoad:
    """Tests for SettingsLoader.load()."""

    def test_load_returns_empty_when_file_missing(self) -> None:
        """Loading non-existent file returns empty settings."""
        settings = SettingsLoader.load(Path("/nonexistent/path.json"))
        assert settings == UserSettings()

    def test_load_valid_settings(self, tmp_path: Path) -> None:
        """Loading valid settings file returns UserSettings."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "env": {
                "AGENTLET_MODEL": "gpt-4.1-mini",
                "AGENTLET_API_KEY": "sk-test",
            },
            "defaults": {
                "max_iterations": 10,
                "workspace_root": "/home/user/projects",
            },
        }))

        settings = SettingsLoader.load(config_file)
        assert settings.env == {
            "AGENTLET_MODEL": "gpt-4.1-mini",
            "AGENTLET_API_KEY": "sk-test",
        }
        assert settings.defaults == {
            "max_iterations": 10,
            "workspace_root": "/home/user/projects",
        }

    def test_load_env_only(self, tmp_path: Path) -> None:
        """Loading settings with only env section."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "env": {"AGENTLET_MODEL": "gpt-4"},
        }))

        settings = SettingsLoader.load(config_file)
        assert settings.env == {"AGENTLET_MODEL": "gpt-4"}
        assert settings.defaults == {}

    def test_load_defaults_only(self, tmp_path: Path) -> None:
        """Loading settings with only defaults section."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"max_iterations": 5},
        }))

        settings = SettingsLoader.load(config_file)
        assert settings.env == {}
        assert settings.defaults == {"max_iterations": 5}

    def test_load_empty_object(self, tmp_path: Path) -> None:
        """Loading empty JSON object returns empty settings."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("{}")

        settings = SettingsLoader.load(config_file)
        assert settings.env == {}
        assert settings.defaults == {}

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Loading invalid JSON raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("not valid json")

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "Invalid JSON" in str(exc_info.value)
        assert str(config_file) in str(exc_info.value)

    def test_load_non_object_json(self, tmp_path: Path) -> None:
        """Loading non-object JSON raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "must contain a JSON object" in str(exc_info.value)

    def test_load_env_not_object(self, tmp_path: Path) -> None:
        """Loading settings with non-object env raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({"env": [1, 2, 3]}))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "'env'" in str(exc_info.value)
        assert "must be an object" in str(exc_info.value)

    def test_load_defaults_not_object(self, tmp_path: Path) -> None:
        """Loading settings with non-object defaults raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({"defaults": "string"}))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "'defaults'" in str(exc_info.value)
        assert "must be an object" in str(exc_info.value)

    def test_load_env_value_not_string(self, tmp_path: Path) -> None:
        """Loading settings with non-string env value raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "env": {"AGENTLET_MODEL": 123},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "AGENTLET_MODEL" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_load_invalid_top_level_key(self, tmp_path: Path) -> None:
        """Loading settings with invalid top-level key raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "invalid_key": "value",
            "env": {},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "invalid_key" in str(exc_info.value)
        assert "Invalid top-level keys" in str(exc_info.value)

    def test_load_invalid_defaults_key(self, tmp_path: Path) -> None:
        """Loading settings with invalid defaults key raises ValueError."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"invalid_option": "value"},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "invalid_option" in str(exc_info.value)
        assert "Invalid keys in 'defaults'" in str(exc_info.value)


class TestSettingsLoaderApplyEnv:
    """Tests for SettingsLoader.apply_env()."""

    def test_apply_env_sets_unset_variables(self) -> None:
        """apply_env sets variables that are not already in environment."""
        # Clear any existing test variable
        test_var = "AGENTLET_TEST_VAR_12345"
        if test_var in os.environ:
            del os.environ[test_var]

        settings = UserSettings(env={test_var: "test_value"})

        try:
            SettingsLoader.apply_env(settings)
            assert os.environ.get(test_var) == "test_value"
        finally:
            # Cleanup
            if test_var in os.environ:
                del os.environ[test_var]

    def test_apply_env_preserves_existing_variables(self) -> None:
        """apply_env does not overwrite existing environment variables."""
        test_var = "AGENTLET_TEST_VAR_12345"

        # Set the variable first
        os.environ[test_var] = "existing_value"

        settings = UserSettings(env={test_var: "new_value"})

        try:
            SettingsLoader.apply_env(settings)
            # Should preserve existing value
            assert os.environ.get(test_var) == "existing_value"
        finally:
            # Cleanup
            if test_var in os.environ:
                del os.environ[test_var]

    def test_apply_env_empty_settings(self) -> None:
        """apply_env with empty settings does nothing."""
        settings = UserSettings()
        # Should not raise
        SettingsLoader.apply_env(settings)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_settings_alias(self, tmp_path: Path) -> None:
        """load_settings is an alias for SettingsLoader.load."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({"env": {"FOO": "bar"}}))

        settings = load_settings(config_file)
        assert settings.env == {"FOO": "bar"}

    def test_apply_env_from_settings_alias(self) -> None:
        """apply_env_from_settings is an alias for SettingsLoader.apply_env."""
        test_var = "AGENTLET_TEST_ALIAS_12345"
        if test_var in os.environ:
            del os.environ[test_var]

        settings = UserSettings(env={test_var: "alias_value"})

        try:
            apply_env_from_settings(settings)
            assert os.environ.get(test_var) == "alias_value"
        finally:
            if test_var in os.environ:
                del os.environ[test_var]


class TestAllAllowedDefaultKeys:
    """Tests that all allowed default keys can be used."""

    def test_all_allowed_defaults_keys(self, tmp_path: Path) -> None:
        """All allowed default keys can be set in settings."""
        config_file = tmp_path / "settings.json"

        all_defaults = {
            "provider": "anthropic",
            "workspace_root": "/path/to/workspace",
            "state_dir": ".custom_agentlet",
            "session_path": "/path/to/session.jsonl",
            "memory_path": "/path/to/memory.md",
            "instructions_path": "/path/to/instructions.md",
            "max_iterations": 15,
            "bash_timeout_seconds": 300,
        }

        config_file.write_text(json.dumps({"defaults": all_defaults}))

        # Should not raise
        settings = SettingsLoader.load(config_file)
        assert settings.defaults == all_defaults


class TestDefaultsTypeValidation:
    """Tests for defaults value type validation."""

    def test_path_setting_rejects_non_string(self, tmp_path: Path) -> None:
        """Path settings must be strings."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"workspace_root": False},
            "env": {"AGENTLET_MODEL": "m", "AGENTLET_API_KEY": "k"},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "workspace_root" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)
        assert "bool" in str(exc_info.value)

    def test_provider_rejects_invalid_value(self, tmp_path: Path) -> None:
        """provider must be one of the supported provider names."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"provider": "bedrock"},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "provider" in str(exc_info.value)
        assert "anthropic" in str(exc_info.value)
        assert "openai-like" in str(exc_info.value)

    def test_max_iterations_rejects_non_int(self, tmp_path: Path) -> None:
        """max_iterations must be an integer."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"max_iterations": "10"},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "max_iterations" in str(exc_info.value)
        assert "must be an integer" in str(exc_info.value)

    def test_max_iterations_rejects_bool(self, tmp_path: Path) -> None:
        """max_iterations must not be a boolean."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"max_iterations": True},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "max_iterations" in str(exc_info.value)
        assert "must be an integer" in str(exc_info.value)

    def test_bash_timeout_rejects_non_number(self, tmp_path: Path) -> None:
        """bash_timeout_seconds must be a number."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"bash_timeout_seconds": "120"},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "bash_timeout_seconds" in str(exc_info.value)
        assert "must be a number" in str(exc_info.value)

    def test_bash_timeout_accepts_int(self, tmp_path: Path) -> None:
        """bash_timeout_seconds accepts integer values."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"bash_timeout_seconds": 120},
        }))

        settings = SettingsLoader.load(config_file)
        assert settings.defaults["bash_timeout_seconds"] == 120

    def test_bash_timeout_accepts_float(self, tmp_path: Path) -> None:
        """bash_timeout_seconds accepts float values."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"bash_timeout_seconds": 120.5},
        }))

        settings = SettingsLoader.load(config_file)
        assert settings.defaults["bash_timeout_seconds"] == 120.5

    def test_max_iterations_rejects_zero(self, tmp_path: Path) -> None:
        """max_iterations must be greater than 0."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"max_iterations": 0},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "max_iterations" in str(exc_info.value)
        assert "must be greater than 0" in str(exc_info.value)

    def test_max_iterations_rejects_negative(self, tmp_path: Path) -> None:
        """max_iterations must not be negative."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"max_iterations": -5},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "max_iterations" in str(exc_info.value)
        assert "must be greater than 0" in str(exc_info.value)

    def test_bash_timeout_rejects_zero(self, tmp_path: Path) -> None:
        """bash_timeout_seconds must be greater than 0."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"bash_timeout_seconds": 0},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "bash_timeout_seconds" in str(exc_info.value)
        assert "must be greater than 0" in str(exc_info.value)

    def test_bash_timeout_rejects_negative(self, tmp_path: Path) -> None:
        """bash_timeout_seconds must not be negative."""
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps({
            "defaults": {"bash_timeout_seconds": -1},
        }))

        with pytest.raises(ValueError) as exc_info:
            SettingsLoader.load(config_file)

        assert "bash_timeout_seconds" in str(exc_info.value)
        assert "must be greater than 0" in str(exc_info.value)
