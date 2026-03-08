# agentlet

`agentlet` is a small Python agent framework for coding and research workflows.

The current repository includes:

- a thin core loop with centralized approvals,
- a fixed built-in toolset for local coding and web lookup,
- JSONL session persistence plus markdown durable memory,
- a terminal runtime with end-to-end `AskUserQuestion` pause/resume support.

## Quick Start

Install the project and test dependencies:

```bash
uv sync --dev
```

Set the model provider environment:

```bash
export AGENTLET_PROVIDER="openai"
export AGENTLET_MODEL="gpt-4.1-mini"
export AGENTLET_API_KEY="..."
# Optional for OpenAI-compatible providers:
export AGENTLET_BASE_URL="https://api.openai.com/v1"
# Required for Anthropic providers:
export AGENTLET_MAX_OUTPUT_TOKENS="1024"
```

Or use the settings file at `~/.agentlet/settings.json` (see Configuration section below).

Run the CLI against the current workspace:

```bash
uv run python -m apps.cli "Summarize docs/ARCHITECTURE.md" --workspace-root .
```

`--workspace-root` defines the root for file-system tools and for resolving
relative `Bash` working directories. It does not sandbox arbitrary shell
commands.

Run the test suite:

```bash
uv run pytest
```

## Configuration

`agentlet` supports a settings file at `~/.agentlet/settings.json` for persistent configuration:

```json
{
  "env": {
    "AGENTLET_PROVIDER": "openai",
    "AGENTLET_MODEL": "gpt-4.1-mini",
    "AGENTLET_API_KEY": "sk-...",
    "AGENTLET_BASE_URL": "https://api.openai.com/v1"
  },
  "defaults": {
    "provider": "openai",
    "max_iterations": 8,
    "bash_timeout_seconds": 120
  }
}
```

### Configuration Priority (high to low)

1. **CLI arguments** (e.g., `--workspace-root`, `--max-iterations`)
2. **Environment variables** (system env vars take precedence over settings file)
3. **Settings file** (`~/.agentlet/settings.json`)
4. **Built-in defaults**

### Settings Reference

**`env` section:**
- `AGENTLET_PROVIDER` - Provider name: `openai`, `anthropic`, or `openai_like` (`openai-like` is also accepted by the CLI)
- `AGENTLET_MODEL` - Model name to use
- `AGENTLET_API_KEY` - API key for the model provider
- `AGENTLET_BASE_URL` - Optional base URL for `openai_like`, or Anthropic API override
- `AGENTLET_MAX_OUTPUT_TOKENS` - Required for `anthropic`
- `AGENTLET_ANTHROPIC_VERSION` - Optional Anthropic API version override

**`defaults` section:**
- `provider` - Default CLI provider
- `workspace_root` - Default workspace directory
- `state_dir` - Directory for session/memory files
- `session_path` - Path to JSONL session file
- `memory_path` - Path to markdown memory file
- `instructions_path` - Path to custom instructions file
- `max_iterations` - Maximum tool call iterations per turn
- `bash_timeout_seconds` - Default timeout for Bash tool execution

## Runtime Files

By default the runtime creates:

```text
.agentlet/
├── memory.md
└── session.jsonl
```

`AGENTS.md` at the workspace root is loaded as additional system instructions
when present.
