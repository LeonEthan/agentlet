# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`agentlet` is a Python agent framework for coding and research workflows. It features a thin core loop with centralized approvals, a fixed built-in toolset, JSONL session persistence, and a terminal runtime with end-to-end pause/resume support.

## Development Commands

This project uses `uv` for dependency management and Python 3.11+.

```bash
# Install dependencies (including dev dependencies)
uv sync --dev

# Run the test suite
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_agent_loop.py

# Run a specific test
uv run pytest tests/unit/test_agent_loop.py::test_name -v

# Run the CLI
uv run python -m apps.cli "Your task here" --workspace-root .

# Run with custom model settings
export AGENTLET_MODEL="gpt-4.1-mini"
export AGENTLET_API_KEY="..."
uv run python -m apps.cli "Summarize docs/ARCHITECTURE.md" --workspace-root .
```

## Architecture

### Module Boundaries

The codebase is organized by responsibility:

- **`agentlet/core/`** - Core execution model. Contains `loop.py` (AgentLoop orchestration), `context.py` (ContextBuilder), `interrupts.py` (pause/resume contracts), `approvals.py` (approval policy), and `messages.py` (normalized message structures). Must not contain CLI or transport logic.

- **`agentlet/llm/`** - Model provider abstraction. Contains `base.py` (ModelClient protocol), `openai_like.py` (OpenAI-compatible adapter), and `schemas.py` (request/response types). Does not own session or memory policy.

- **`agentlet/tools/`** - Capability execution. Contains `base.py` (Tool protocol and ToolResult), `registry.py` (tool registration), and subpackages by category: `fs/` (Read, Write, Edit, Glob, Grep), `exec/` (Bash), `web/` (WebSearch, WebFetch), `interaction/` (AskUserQuestion). Does not own loop control.

- **`agentlet/memory/`** - Durable state. Contains `session_store.py` (JSONL session history) and `memory_store.py` (markdown durable memory). File-backed and inspectable.

- **`agentlet/runtime/`** - App wiring and interactive flow. Contains `app.py` (runtime assembly), `events.py` (runtime events), and `user_io.py` (user interaction boundary). Handles `AskUserQuestion` resume.

- **`apps/`** - End-user entrypoints. Contains `cli.py` (terminal interface).

### Key Contracts

**ToolResult** (`agentlet/tools/base.py:52-68`) - All tools return a normalized result with `output`, `metadata`, `is_error`, and `interrupt` fields.

**Approval Categories** (`agentlet/tools/base.py:14-25`) - Tools are grouped by risk: `read_only`, `mutating`, `exec`, `external_or_interrupt`. Default policy allows `read_only` without approval; others require runtime approval.

**Interrupts** (`agentlet/core/interrupts.py`) - `AskUserQuestion` is a structured interrupt that pauses execution. The question payload is persisted in session history; resumes are validated against the original request to prevent replays.

**Session Persistence** - Stored as JSONL at `.agentlet/session.jsonl`. History is append-only and includes messages, approval requests, and interrupt metadata.

### Runtime Flow

1. Receive user input → Load session history → Load durable memory → Build context
2. Call model → If tool calls: validate → apply approval policy → execute
3. If approval required: persist partial turn + approval_request, return to runtime
4. If interrupt (AskUserQuestion): persist partial turn, return to runtime
5. On resume: inject structured context into next turn, reject replays
6. When model emits final response: persist turn, return to runtime

## Configuration

Environment variables for LLM:
- `AGENTLET_MODEL` - Model name (e.g., `gpt-4.1-mini`)
- `AGENTLET_API_KEY` - API key
- `AGENTLET_BASE_URL` - Optional, for OpenAI-compatible providers

CLI arguments:
- `--workspace-root` - Workspace directory for file-system tools (default: `.`)
- `--state-dir` - Directory for session/memory files (default: `.agentlet`)
- `--session-path`, `--memory-path`, `--instructions-path` - Override default paths

Runtime creates at workspace root:
```
.agentlet/
├── memory.md       # Durable memory included as system message
└── session.jsonl   # Append-only session history
AGENTS.md           # Optional workspace-local system instructions
```

## Testing Patterns

- Unit tests in `tests/unit/`
- E2E tests in `tests/e2e/`
- Fake implementations for testing: `FakeModelClient`, `FakeUserIO` (see `tests/e2e/test_main_loop_e2e.py` for patterns)

## Design Principles (from AGENTS.md)

- Target modern Python with minimal dependencies
- Prefer dataclasses, protocols, and simple explicit types
- Keep control flow direct; avoid hidden side effects
- Prefer composition over framework-style magic
- Default to ASCII unless a file already requires Unicode
- Keep comments rare and high-value

## Non-Goals

Avoid introducing without strong design case: multi-agent orchestration, planner subsystems, DAG execution engines, vector-database-first memory, broad plugin lifecycle systems, large callback frameworks.
