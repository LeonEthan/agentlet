from __future__ import annotations

"""Tool policy and runtime configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolRuntimeConfig:
    """Runtime configuration injected into built-in tools at construction time.

    Tool safety is runtime configuration, not hidden prompt behavior.
    Tests can supply smaller limits and fake dependencies.
    """

    cwd: Path
    bash_timeout_seconds: float = 30.0
    web_timeout_seconds: float = 10.0
    max_read_bytes: int = 64_000
    max_write_bytes: int = 128_000
    max_search_results: int = 8
    max_fetch_chars: int = 20_000
    max_fetch_bytes: int = 512_000


@dataclass(frozen=True)
class ToolPolicy:
    """Policy object for capability switches.

    Distinguishes between:
    - shipped tools: the full built-in tool set compiled into agentlet
    - enabled tools: tools allowed by the local runtime policy
    - advertised tools: tools actually exposed to the model in the current run
    """

    allow_network: bool = True
    allow_write: bool = True
    allow_bash: bool = True
