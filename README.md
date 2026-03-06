# agentlet

`agentlet` is a small Python agent framework for coding and research workflows.

PR0 sets up the project scaffold only. It establishes package boundaries from
`docs/ARCHITECTURE.md` without implementing the core loop, tools, runtime,
provider adapters, or persistence behavior yet.

## Quick start

Create an environment and install the project in editable mode:

```bash
uv sync --dev
```

Run the import smoke test:

```bash
uv run pytest
```
