# Architecture Design

## Purpose

`agentlet` is a Python agent framework for coding and research workflows.

The framework combines two guiding ideas:

- Keep the core runtime small and easy to reason about.
- Make capabilities explicit through a constrained built-in toolset rather than a large, implicit abstraction surface.

This document defines the target architecture, major module boundaries, runtime flow, and non-goals for the first implementation.

## Design Principles

### 1. Thin Core

The core agent loop should only do four things:

1. Load state.
2. Build context.
3. Call the model.
4. Execute tools and persist the turn.

Anything else should live outside the loop.

### 2. Clear Boundaries

The framework is split by responsibility, not by technical novelty. Context building, tool execution, persistence, model access, and runtime interaction are separate modules.

### 3. Built-in Tools as First-Class Capabilities

The framework ships with a fixed high-value toolset for coding agents:

- `Read`
- `Write`
- `Edit`
- `Bash`
- `Glob`
- `Grep`
- `WebSearch`
- `WebFetch`
- `AskUserQuestion`

These tools are built in, but still conform to a common tool protocol and registry.

### 4. Human-in-the-Loop by Design

`AskUserQuestion` is treated as a runtime interrupt, not as a normal side-effectful tool call. The loop must be able to pause, surface a structured question to the user, and resume after receiving the answer.

### 5. File-Backed State First

Early versions should prefer readable local files over heavier infrastructure. Session history, memory, and agent instructions should be inspectable and editable without proprietary storage layers.

### 6. Extension Without Core Bloat

The framework should make it easy to add new providers, tools, or runtimes without expanding the core execution model.

## Architecture Overview

```text
agentlet/
├── AGENTS.md
├── docs/
│   └── ARCHITECTURE.md
├── agentlet/
│   ├── core/
│   │   ├── approvals.py
│   │   ├── context.py
│   │   ├── loop.py
│   │   ├── messages.py
│   │   └── types.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── openai_like.py
│   │   └── schemas.py
│   ├── memory/
│   │   ├── memory_store.py
│   │   └── session_store.py
│   ├── runtime/
│   │   ├── app.py
│   │   ├── events.py
│   │   └── user_io.py
│   └── tools/
│       ├── base.py
│       ├── registry.py
│       ├── exec/
│       │   └── bash.py
│       ├── fs/
│       │   ├── edit.py
│       │   ├── glob.py
│       │   ├── grep.py
│       │   ├── read.py
│       │   └── write.py
│       ├── interaction/
│       │   └── ask_user_question.py
│       └── web/
│           ├── fetch.py
│           └── search.py
└── apps/
    └── cli.py
```

## Module Responsibilities

### `agentlet.core`

Owns the core execution model.

- `loop.py`
  - Defines `AgentLoop`.
  - Orchestrates history loading, context construction, model calls, tool execution, and persistence.
  - Handles tool-call iteration and termination rules.
- `context.py`
  - Defines `ContextBuilder`.
  - Builds the effective model input from system instructions, history, memory, and current task state.
- `messages.py`
  - Defines normalized message structures used across the framework.
- `types.py`
  - Defines shared runtime dataclasses and protocols.
- `approvals.py`
  - Defines tool approval and execution policy checks.

`core` must not know about CLI rendering, HTTP transport, or channel-specific behavior.

### `agentlet.llm`

Owns model-provider abstraction.

- `base.py`
  - Defines the `ModelClient` protocol.
- `openai_like.py`
  - Implements one provider adapter for OpenAI-compatible APIs.
- `schemas.py`
  - Defines request/response schemas for model output and tool call payloads.

The LLM layer should not know about sessions, file layout, or runtime interrupts.

### `agentlet.tools`

Owns capability execution.

- `base.py`
  - Defines the `Tool` interface and `ToolResult`.
- `registry.py`
  - Registers and resolves tools.
- `fs/*`
  - File-system tools.
- `exec/*`
  - Command execution tools.
- `web/*`
  - Network-backed information tools.
- `interaction/*`
  - Human interaction tools.

The tool layer executes work. It does not own loop control, except that it may signal an interrupt through structured output.

### `agentlet.memory`

Owns durable state.

- `session_store.py`
  - Stores append-only conversation history.
  - Supports reading recent turns and writing new turns.
- `memory_store.py`
  - Stores durable user or project memory.
  - Starts as file-backed markdown.

The framework should begin with readable, local persistence such as JSONL plus markdown files.

### `agentlet.runtime`

Owns app wiring and interactive flow.

- `app.py`
  - Assembles the configured loop, registry, model client, and stores.
- `events.py`
  - Defines runtime-level events.
- `user_io.py`
  - Defines the interaction boundary for user prompts, approval requests, and question interrupts.

`runtime` is where `AskUserQuestion` is resumed.

### `apps`

Owns end-user entrypoints.

- `cli.py`
  - First runtime target.
  - Drives a local interactive terminal experience.

Later runtimes, such as HTTP or chat integrations, should be added here or as sibling application packages.

## Built-in Tool Contracts

### Read

- Reads file content from the workspace.
- Supports line ranges and output truncation.
- Must not mutate files.

### Write

- Creates new files.
- Overwrite should require an explicit flag.
- Default behavior should be safe.

### Edit

- Applies precise modifications to existing files.
- Preferred contract is exact-text replacement or patch-style editing.
- Must fail clearly if the target context does not match.

### Bash

- Runs terminal commands inside the selected working directory.
- Supports timeout and approval policy.
- Must return stdout, stderr, exit code, and execution metadata.

### Glob

- Finds files by glob pattern.
- Returns paths only.

### Grep

- Searches file contents using regex.
- Returns file path, line number, and snippet.

### WebSearch

- Searches the web for current information.
- Returns ranked search results, not full documents.

### WebFetch

- Fetches a specific URL and extracts usable text content.
- Should normalize output to plain text or markdown.

### AskUserQuestion

- Asks a structured clarifying question.
- Supports multiple choice options and optional free text.
- Returns an interrupt result that pauses the agent loop until the user responds.
- Does not require a separate approval prompt before asking; the question itself is
  the human-in-the-loop boundary.
- The interrupted question payload must be persisted in session history so a later
  resume can be validated against the original request id and option set.

## Tool Execution Model

All tools should return one normalized result type.

```python
@dataclass
class ToolResult:
    output: str
    metadata: dict[str, Any] | None = None
    is_error: bool = False
    interrupt: bool = False
```

Interpretation:

- `output`
  - Model-visible result text.
- `metadata`
  - Runtime-visible structured payload.
- `is_error`
  - Marks tool failure without throwing away the turn.
- `interrupt`
  - Signals the runtime to pause and resume later.

## Approval Model

Tools are grouped by risk so the runtime can apply consistent approval behavior.

- `read_only`
  - `Read`, `Glob`, `Grep`
- `mutating`
  - `Write`, `Edit`
- `exec`
  - `Bash`
- `external_or_interrupt`
  - `WebSearch`, `WebFetch`, `AskUserQuestion`

Approval policy should be centralized in `core.approvals`, not spread across tool implementations.

## Runtime Flow

The standard turn flow is:

1. Receive user input.
2. Load session history.
3. Load durable memory.
4. Build model context.
5. Call model.
6. If the model emits tool calls:
   1. Validate the tool request.
   2. Apply approval policy.
   3. Execute the tool.
   4. Append the tool result to the working message list.
   5. If the tool result is an interrupt, pause and wait for user input.
7. When the model emits a final assistant response, persist the turn.
8. Return the response to the runtime.

## Recommended Agent Behavior

The default system instructions should bias the agent toward this workflow:

1. Ask for clarification when the task is underspecified.
2. Use `Glob` or `Grep` before reading large code areas.
3. Use `Read` before making changes.
4. Prefer `Edit` to `Write` for existing files.
5. Use `Bash` for verification and repository inspection.
6. Use `WebSearch` and `WebFetch` only when current external information is required.

## Session and Memory Strategy

### Session History

Session history should be append-only and local.

Initial recommendation:

- One JSONL file per session.
- Each line stores a normalized message or metadata record.
- History truncation, if needed, should be logical rather than destructive.

### Durable Memory

Durable memory starts simple.

Initial recommendation:

- `MEMORY.md` for facts worth carrying across turns.
- Optional `HISTORY.md` for searchable summaries or archived context.

This keeps the system inspectable and avoids premature vector-store complexity.

## Non-Goals for v1

The following are intentionally out of scope for the first version:

- Multi-agent orchestration
- Planner or DAG workflow engines
- Complex plugin lifecycle management
- Vector-database-first memory
- Multiple UI runtimes in the initial release
- Deep event middleware stacks

## Success Criteria

The architecture is successful if:

- A single developer can understand the main loop quickly.
- Adding a new tool does not require changing the loop semantics.
- Adding a new provider does not affect session or tool code.
- The runtime can pause and resume around `AskUserQuestion`.
- Session and memory files are readable and debuggable from the filesystem.
- The framework remains useful without becoming a platform before it earns that complexity.
