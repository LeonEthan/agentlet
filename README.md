# agentlet

<p align="center">
  <strong>A small Python agent framework for coding and research workflows.</strong>
</p>

<p align="center">
  <a href="https://github.com/yourusername/agentlet/actions"><img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests"></a>
  <a href="https://pypi.org/project/agentlet"><img src="https://img.shields.io/pypi/v/agentlet.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

**agentlet** is a minimal, production-grade agent framework that brings structure to AI-powered coding workflows without the bloat.

We built agentlet because we wanted a framework that stays out of the way—thin orchestration, explicit approvals, and durable state that you can inspect and version. No hidden side effects, no complex plugin systems, just a clean loop you can reason about.

## Features

1. **Thin Core Loop** — Centralized orchestration with clear execution flow; the loop owns state loading, tool validation, and persistence.

2. **Built-in Toolset** — Carefully selected tools for coding and research:
   - **Filesystem**: `Read`, `Write`, `Edit`, `Glob`, `Grep` with workspace sandboxing
   - **Execution**: `Bash` with configurable timeouts and working directory control
   - **Web**: `WebSearch`, `WebFetch` for real-time information
   - **Interaction**: `AskUserQuestion` for structured clarifications

3. **Approval-First Design** — Tools are categorized by risk (`read_only`, `mutating`, `exec`, `external_or_interrupt`). Default policy allows reads freely; everything else requires explicit approval.

4. **Durable Session State** — Append-only JSONL session history plus markdown memory files. Your conversations are persisted transparently and can be resumed across sessions.

5. **Structured Interrupts** — `AskUserQuestion` creates pausable, resumable workflows with replay protection. The runtime validates resumes against original requests to prevent double-execution.

6. **Multi-Provider LLM Support** — Works with OpenAI, Anthropic, and any OpenAI-compatible API (local models, custom endpoints).

7. **Zero Production Dependencies** — Core framework has no external dependencies. Only test utilities require pytest.

---

## Quick Start

*Requires Python 3.11+*

```bash
# Install with uv (or pip)
uv pip install agentlet

# Set your model credentials
export AGENTLET_MODEL="gpt-4.1-mini"
export AGENTLET_API_KEY="sk-..."

# Run a task
agentlet "Summarize the README files in this project"
```

### Using as a Library

```python
import sys
from agentlet.runtime.app import build_default_runtime_app
from apps.cli import TerminalUserIO

# Assemble the runtime
app = build_default_runtime_app(
    user_io=TerminalUserIO(
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    ),
    workspace_root=".",
)

# Run a turn
outcome = app.run_turn(current_task="Find all Python files and count lines")
print(outcome.message.content)
```

---

## Configuration

agentlet can be configured via environment variables or a settings file at `~/.agentlet/settings.json`:

```json
{
  "env": {
    "AGENTLET_PROVIDER": "openai",
    "AGENTLET_MODEL": "gpt-4.1-mini",
    "AGENTLET_API_KEY": "sk-...",
    "AGENTLET_BASE_URL": "https://api.openai.com/v1"
  },
  "defaults": {
    "max_iterations": 8,
    "bash_timeout_seconds": 120
  }
}
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `AGENTLET_PROVIDER` | `openai`, `anthropic`, or `openai_like` |
| `AGENTLET_MODEL` | Model name (e.g., `gpt-4.1-mini`, `claude-3-opus`) |
| `AGENTLET_API_KEY` | API key for the provider |
| `AGENTLET_BASE_URL` | Optional base URL for OpenAI-compatible providers |
| `AGENTLET_MAX_OUTPUT_TOKENS` | Required for Anthropic provider |

**Priority** (high to low): CLI arguments → Environment variables → Settings file → Built-in defaults

---

## Runtime Files

By default, agentlet creates the following in your workspace:

```
.agentlet/
├── session.jsonl   # Append-only conversation history
└── memory.md       # Durable memory included in system context
AGENTS.md           # Optional workspace-specific instructions
```

The session file is JSONL—human-readable, line-oriented, and perfect for version control.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Runtime App                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  AgentLoop   │  │ ToolRegistry │  │   ContextBuilder │  │
│  │              │  │              │  │                  │  │
│  │ • Orchestrate│  │ • Read/Write │  │ • Build messages │  │
│  │ • Approve    │  │ • Bash       │  │ • Load memory    │  │
│  │ • Persist    │  │ • WebSearch  │  │ • Resume state   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Model Clients                          │
│         (OpenAI / Anthropic / OpenAI-Compatible)            │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Explicit over implicit** — Control flow is direct; no hidden side effects
- **Composition over inheritance** — Dataclasses and protocols, not framework magic
- **File-backed state** — Session and memory are inspectable, portable files
- **Minimal dependencies** — Modern Python standard library first

---

## Example: Custom Tool Registration

```python
from dataclasses import dataclass
from agentlet.tools.base import Tool, ToolDefinition, ToolResult
from agentlet.tools.registry import ToolRegistry

@dataclass(frozen=True, slots=True)
class CalculatorTool:
    """A simple calculator tool."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Calculator",
            description="Perform basic arithmetic operations.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
                },
                "required": ["a", "b", "op"],
            },
            approval_category="read_only",
        )

    def execute(self, arguments: dict) -> ToolResult:
        a = arguments["a"]
        b = arguments["b"]
        op = arguments["op"]
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        return ToolResult(output=str(result))

# Build registry with custom tools
registry = ToolRegistry([CalculatorTool(), ...])
```

---

## Development

```bash
# Clone and install dependencies
uv sync --dev

# Run the test suite
uv run pytest

# Run a specific test
uv run pytest tests/unit/test_agent_loop.py -v

# Run the CLI in development mode
uv run python -m apps.cli "Your task here" --workspace-root .
```

---

## Why agentlet?

Most agent frameworks optimize for demos. agentlet optimizes for:

- **Long-running sessions** that survive restarts and can be resumed
- **Auditable execution** with full session history in plain JSONL
- **Safe automation** with approval policies that match your risk tolerance
- **Minimal footprint** — no vector databases, no DAG engines, no callback hell

If you're building agents that need to work reliably over hours or days, agentlet is designed for you.

---

## License

agentlet is released under the MIT License. See [LICENSE](LICENSE) for details.
