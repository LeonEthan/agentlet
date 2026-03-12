from __future__ import annotations

"""User-level settings file helpers for CLI-facing configuration."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentlet.agent.providers.registry import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
)

SETTINGS_DIRNAME = ".agentlet"
SETTINGS_FILENAME = "settings.json"
SETTINGS_FILENAME_LEGACY = "setting.json"

_STRING_FIELDS = {"provider", "model", "api_key", "api_base"}

# Provider to environment variable mappings (LiteLLM standard)
_PROVIDER_API_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "anyscale": "ANYSCALE_API_KEY",
}

_PROVIDER_BASE_URL_ENVS: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "azure": "AZURE_API_BASE",
    "together_ai": "TOGETHERAI_BASE_URL",
    "fireworks": "FIREWORKS_BASE_URL",
    "anyscale": "ANYSCALE_BASE_URL",
}
_FLOAT_FIELDS = {"temperature"}
_INT_FIELDS = {"max_tokens"}
_ALLOWED_FIELDS = _STRING_FIELDS | _FLOAT_FIELDS | _INT_FIELDS


class SettingsError(ValueError):
    """Raised when the local settings file cannot be read or written safely."""


@dataclass(frozen=True)
class AgentletSettings:
    """Config values persisted in `~/.agentlet/settings.json`."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON payload order for persistence."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def default_settings_path(home_dir: Path | None = None) -> Path:
    """Return the canonical `settings.json` path for this user.

    Prefers `settings.json` (new) over `setting.json` (legacy) if both exist.
    Returns the new path by default if neither exists.
    """
    base_dir = home_dir if home_dir is not None else Path.home()
    new_path = base_dir / SETTINGS_DIRNAME / SETTINGS_FILENAME
    legacy_path = base_dir / SETTINGS_DIRNAME / SETTINGS_FILENAME_LEGACY

    # Prefer new path, but fall back to legacy if only it exists
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return new_path  # Default to new path if neither exists


def _validate_allowed_fields(data: dict[str, Any], path: Path, context: str = "") -> None:
    """Validate that data only contains allowed fields."""
    unknown_fields = sorted(set(data) - _ALLOWED_FIELDS)
    if unknown_fields:
        names = ", ".join(unknown_fields)
        context_str = f" {context}" if context else ""
        raise SettingsError(f"Unsupported settings keys in {path}{context_str}: {names}")


def _build_settings_from_dict(data: dict[str, Any], path: Path) -> AgentletSettings:
    """Build AgentletSettings from a flat dictionary after validation."""
    _validate_allowed_fields(data, path)

    return AgentletSettings(
        provider=_validate_string_field(data, "provider", path),
        model=_validate_string_field(data, "model", path),
        api_key=_validate_string_field(data, "api_key", path),
        api_base=_validate_string_field(data, "api_base", path),
        temperature=_validate_float_field(data, "temperature", path),
        max_tokens=_validate_int_field(data, "max_tokens", path),
    )


def _load_nested_settings(payload: dict[str, Any], path: Path) -> AgentletSettings:
    """Load settings from nested structure with 'defaults' section."""
    defaults_section = payload.get("defaults")
    if defaults_section is None:
        return AgentletSettings()

    if not isinstance(defaults_section, dict):
        raise SettingsError(f"Settings key `defaults` in {path} must be an object.")

    _validate_allowed_fields(defaults_section, path, "defaults")
    return _build_settings_from_dict(defaults_section, path)


def _load_flat_settings(payload: dict[str, Any], path: Path) -> AgentletSettings:
    """Load settings from flat structure (backward compatible)."""
    return _build_settings_from_dict(payload, path)


def load_settings(settings_path: Path | None = None) -> AgentletSettings:
    """Load `settings.json` when present, or return empty defaults.

    Supports both nested structure (with 'defaults' section) and flat
    structure (backward compatible).
    """
    path = settings_path or default_settings_path()
    if not path.exists():
        return AgentletSettings()
    if path.is_dir():
        raise SettingsError(f"Settings path must be a file: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid JSON in settings file {path}: {exc.msg}") from exc
    except OSError as exc:
        raise SettingsError(f"Unable to read settings file {path}: {exc.strerror}") from exc

    if not isinstance(payload, dict):
        raise SettingsError(f"Settings file must contain a JSON object: {path}")

    # Detect nested structure by presence of 'defaults' key
    if "defaults" in payload:
        return _load_nested_settings(payload, path)
    else:
        return _load_flat_settings(payload, path)


def _resolve_provider_env(
    env_values: Mapping[str, str],
    stored_value: str | None,
    provider: str | None,
    *,
    override_env: str,
    fallback_env: str,
    provider_mapping: dict[str, str],
) -> str | None:
    """Resolve a provider-specific env var with LiteLLM-standard fallback chain.

    Priority (highest to lowest):
    1. override_env (project-specific override, e.g., AGENTLET_API_KEY)
    2. {PROVIDER}_ENV (LiteLLM standard from provider_mapping)
    3. fallback_env (backward compatibility, e.g., OPENAI_API_KEY)
    4. stored_value from settings.json
    5. None
    """
    # 1. Project-specific override
    if override_env in env_values:
        return env_values[override_env]

    # 2. Provider-specific LiteLLM standard (provider is already normalized)
    if provider:
        provider_env = provider_mapping.get(provider)
        if provider_env and provider_env in env_values:
            return env_values[provider_env]

    # 3. Backward compatibility
    if fallback_env in env_values:
        return env_values[fallback_env]

    return stored_value


def resolve_settings_defaults(
    stored_settings: AgentletSettings,
    *,
    env: Mapping[str, str] | None = None,
) -> AgentletSettings:
    """Resolve the effective CLI defaults from env vars and stored settings."""
    env_values = env if env is not None else os.environ

    # Resolve provider first, as it's needed for API key/base resolution
    provider = _resolve_string_env(
        env_values,
        "AGENTLET_PROVIDER",
        stored_settings.provider,
        DEFAULT_PROVIDER,
    )
    # Normalize provider once for efficient lookup (all mapping keys are lowercase)
    normalized_provider = provider.lower() if provider else None

    return AgentletSettings(
        provider=provider,
        model=_resolve_string_env(
            env_values,
            "AGENTLET_MODEL",
            stored_settings.model,
            DEFAULT_MODEL,
        ),
        api_key=_resolve_provider_env(
            env_values,
            stored_settings.api_key,
            normalized_provider,
            override_env="AGENTLET_API_KEY",
            fallback_env="OPENAI_API_KEY",
            provider_mapping=_PROVIDER_API_KEY_ENVS,
        ),
        api_base=_resolve_provider_env(
            env_values,
            stored_settings.api_base,
            normalized_provider,
            override_env="AGENTLET_BASE_URL",
            fallback_env="OPENAI_BASE_URL",
            provider_mapping=_PROVIDER_BASE_URL_ENVS,
        ),
        temperature=stored_settings.temperature
        if stored_settings.temperature is not None
        else DEFAULT_TEMPERATURE,
        max_tokens=stored_settings.max_tokens,
    )


def write_settings(
    settings: AgentletSettings,
    *,
    settings_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Persist the canonical settings payload to disk."""
    path = settings_path or default_settings_path()
    if path.exists() and not force:
        raise SettingsError(f"Settings file already exists: {path}. Use --force to overwrite it.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
        if os.name != "nt":
            path.chmod(0o600)
    except OSError as exc:
        raise SettingsError(f"Unable to write settings file {path}: {exc.strerror}") from exc

    return path


def _validate_string_field(payload: dict[str, Any], key: str, path: Path) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsError(f"Settings key `{key}` in {path} must be a string or null.")
    return value


def _validate_float_field(payload: dict[str, Any], key: str, path: Path) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"Settings key `{key}` in {path} must be a number or null.")
    return float(value)


def _validate_int_field(payload: dict[str, Any], key: str, path: Path) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(f"Settings key `{key}` in {path} must be an integer or null.")
    return value


def _resolve_string_env(
    env_values: Mapping[str, str],
    env_name: str,
    stored_value: str | None,
    fallback: str | None,
) -> str | None:
    if env_name in env_values:
        return env_values[env_name]
    if stored_value is not None:
        return stored_value
    return fallback
