"""User settings loader for ~/.agentlet/settings.json configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class UserSettings:
    """User settings loaded from ~/.agentlet/settings.json."""

    env: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)


class SettingsLoader:
    """Load and apply user settings from ~/.agentlet/settings.json."""

    SETTINGS_PATH = Path.home() / ".agentlet" / "settings.json"

    _ALLOWED_TOP_KEYS = {"env", "defaults"}
    _ALLOWED_DEFAULT_KEYS = {
        "provider",
        "workspace_root",
        "state_dir",
        "session_path",
        "memory_path",
        "instructions_path",
        "max_iterations",
        "bash_timeout_seconds",
    }

    @classmethod
    def load(cls, path: Path | None = None) -> UserSettings:
        """Load settings from file or return empty settings if file does not exist."""
        settings_path = path or cls.SETTINGS_PATH

        if not settings_path.exists():
            return UserSettings()

        try:
            content = settings_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in settings file {settings_path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Settings file {settings_path} must contain a JSON object"
            )

        cls._validate_settings(data, settings_path)

        env = data.get("env", {})
        defaults = data.get("defaults", {})

        if not isinstance(env, dict):
            raise ValueError(
                f"'env' in settings file {settings_path} must be an object"
            )
        if not isinstance(defaults, dict):
            raise ValueError(
                f"'defaults' in settings file {settings_path} must be an object"
            )

        # Validate that all env values are strings
        for key, value in env.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"Environment variable '{key}' in settings file {settings_path} "
                    f"must be a string, got {type(value).__name__}"
                )

        return UserSettings(env=env, defaults=defaults)

    @classmethod
    def apply_env(cls, settings: UserSettings) -> None:
        """Apply env vars from settings if not already set in os.environ.

        This ensures system environment variables take precedence over
        settings file values.
        """
        for key, value in settings.env.items():
            if key not in os.environ:
                os.environ[key] = value

    # Type expectations for defaults values
    _DEFAULTS_PATH_KEYS = {
        "workspace_root",
        "state_dir",
        "session_path",
        "memory_path",
        "instructions_path",
    }
    _DEFAULTS_STRING_KEYS = {"provider"}
    _DEFAULTS_INT_KEYS = {"max_iterations"}
    _DEFAULTS_NUMBER_KEYS = {"bash_timeout_seconds"}

    @classmethod
    def _validate_settings(cls, data: dict, path: Path) -> None:
        """Validate settings structure."""
        invalid_keys = set(data.keys()) - cls._ALLOWED_TOP_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid top-level keys in settings file {path}: "
                f"{', '.join(sorted(invalid_keys))}. "
                f"Allowed keys: {', '.join(sorted(cls._ALLOWED_TOP_KEYS))}"
            )

        defaults = data.get("defaults", {})
        if isinstance(defaults, dict):
            invalid_default_keys = set(defaults.keys()) - cls._ALLOWED_DEFAULT_KEYS
            if invalid_default_keys:
                raise ValueError(
                    f"Invalid keys in 'defaults' section of settings file {path}: "
                    f"{', '.join(sorted(invalid_default_keys))}. "
                    f"Allowed keys: {', '.join(sorted(cls._ALLOWED_DEFAULT_KEYS))}"
                )
            cls._validate_defaults_types(defaults, path)

    @classmethod
    def _validate_defaults_types(cls, defaults: dict, path: Path) -> None:
        """Validate that defaults values have correct types."""
        for key, value in defaults.items():
            if key in cls._DEFAULTS_PATH_KEYS or key in cls._DEFAULTS_STRING_KEYS:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Setting '{key}' in settings file {path} "
                        f"must be a string, got {type(value).__name__}"
                    )
            elif key in cls._DEFAULTS_INT_KEYS:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"Setting '{key}' in settings file {path} "
                        f"must be an integer, got {type(value).__name__}"
                    )
            elif key in cls._DEFAULTS_NUMBER_KEYS:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"Setting '{key}' in settings file {path} "
                        f"must be a number, got {type(value).__name__}"
                    )
        cls._validate_defaults_ranges(defaults, path)

    @classmethod
    def _validate_defaults_ranges(cls, defaults: dict, path: Path) -> None:
        """Validate that numeric defaults have valid ranges."""
        if "max_iterations" in defaults:
            value = defaults["max_iterations"]
            if value <= 0:
                raise ValueError(
                    f"Setting 'max_iterations' in settings file {path} "
                    f"must be greater than 0, got {value}"
                )
        if "bash_timeout_seconds" in defaults:
            value = defaults["bash_timeout_seconds"]
            if value <= 0:
                raise ValueError(
                    f"Setting 'bash_timeout_seconds' in settings file {path} "
                    f"must be greater than 0, got {value}"
                )
        if "provider" in defaults:
            value = defaults["provider"]
            if value not in {"anthropic", "openai", "openai-like", "openai_like"}:
                raise ValueError(
                    f"Setting 'provider' in settings file {path} "
                    "must be one of: anthropic, openai, openai-like, openai_like"
                )


def load_settings(path: Path | None = None) -> UserSettings:
    """Convenience function to load user settings."""
    return SettingsLoader.load(path)


def apply_env_from_settings(settings: UserSettings) -> None:
    """Convenience function to apply environment variables from settings."""
    SettingsLoader.apply_env(settings)


__all__ = [
    "UserSettings",
    "SettingsLoader",
    "load_settings",
    "apply_env_from_settings",
]
