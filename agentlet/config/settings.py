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
