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
export AGENTLET_MODEL="gpt-4.1-mini"
export AGENTLET_API_KEY="..."
# Optional for OpenAI-compatible providers:
export AGENTLET_BASE_URL="https://api.openai.com/v1"
```

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

## Runtime Files

By default the runtime creates:

```text
.agentlet/
├── memory.md
└── session.jsonl
```

`AGENTS.md` at the workspace root is loaded as additional system instructions
when present.
