# AGENTS.md

## Purpose

This repository builds `agentlet`, a Python agent framework for coding and research workflows.

The repository favors a small core, explicit module boundaries, readable state, and a fixed high-value built-in toolset.

## Primary Goals

- Keep the core loop easy to read and reason about.
- Build around a constrained built-in toolset instead of vague abstractions.
- Preserve human control through approvals and structured questions.
- Prefer file-backed state that is easy to inspect locally.
- Avoid premature framework complexity.

## Architecture Rules

Follow [`docs/ARCHITECTURE.md`](/Users/cuizhengliang/Documents/vibe-coding/agentlet/docs/ARCHITECTURE.md) as the source of truth for module boundaries.

Key rules:

- `core` orchestrates. It should not contain CLI or transport logic.
- `llm` adapts providers. It should not own session or memory policy.
- `tools` execute capabilities. They should not own loop control beyond structured results.
- `memory` persists state. It should stay simple and file-backed in early versions.
- `runtime` handles user interaction, approvals, and pause/resume behavior.
- `apps` contain concrete entrypoints such as CLI.

## Built-in Tool Set

The default built-in tools are:

- `Read`
- `Write`
- `Edit`
- `Bash`
- `Glob`
- `Grep`
- `WebSearch`
- `WebFetch`
- `AskUserQuestion`

Do not add new built-in tools casually. If a new capability is proposed, first decide whether it can be expressed as:

- a refinement of an existing tool,
- a runtime concern,
- or a later extension.

## Tool Semantics

Respect these boundaries:

- `Read` reads files only.
- `Write` creates files and should be safe by default.
- `Edit` performs precise updates to existing files.
- `Bash` executes commands from a workspace-rooted cwd and must surface exit status.
- `Glob` returns matching file paths.
- `Grep` returns regex matches with location context.
- `WebSearch` finds current external information.
- `WebFetch` retrieves and normalizes page content.
- `AskUserQuestion` is a structured clarification interrupt, not generic chat.

Prefer one normalized tool result type across all tools.

## Human-in-the-Loop Rules

`AskUserQuestion` is special:

- It should pause execution.
- It should surface structured options to the user.
- It should resume the loop only after receiving a response.

Do not implement it as a normal fire-and-forget tool.

Approval policy should be centralized and consistent across tools.

## Coding Guidance

- Target modern Python and keep dependencies minimal.
- Prefer dataclasses, protocols, and simple explicit types over heavy metaprogramming.
- Keep control flow direct. Avoid hidden side effects and deep inheritance.
- Prefer composition over framework-style magic.
- Default to ASCII unless a file already requires Unicode.
- Keep comments rare and high-value.
- Make module names reflect responsibility clearly.

## Persistence Guidance

Until there is a strong reason otherwise, prefer:

- JSONL for append-only session history
- Markdown files for durable memory and agent instructions

The on-disk state should be readable and debuggable without custom tooling.

## Non-Goals

Do not introduce these into early versions without a strong design case:

- multi-agent orchestration,
- planner subsystems,
- DAG execution engines,
- vector-database-first memory,
- broad plugin lifecycle systems,
- large callback or middleware frameworks.

## Documentation Requirements

When architecture or runtime behavior changes:

- update [`docs/ARCHITECTURE.md`](/Users/cuizhengliang/Documents/vibe-coding/agentlet/docs/ARCHITECTURE.md),
- keep tool contracts current,
- document any change that affects pause/resume, approvals, or persistence layout.

If implementation and architecture docs disagree, fix the docs or the code in the same change.

## Change Heuristics

When making changes, prefer this order of reasoning:

1. Does the change keep the core loop smaller or clearer?
2. Does it preserve module boundaries?
3. Does it avoid adding a second abstraction for the same concern?
4. Can it be done in runtime or tool code instead of core?
5. Does it keep local state inspectable?

If the answer to several of these is no, the design likely needs revision.
