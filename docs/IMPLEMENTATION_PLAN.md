# Implementation Plan

## Purpose

This document turns the target architecture in `docs/ARCHITECTURE.md` into a PR-by-PR execution plan for AI-assisted development.

The plan is optimized for:

- small, reviewable pull requests,
- stable shared contracts before broad implementation,
- parallel work on low-coupling modules,
- explicit handling of the `AskUserQuestion` interrupt flow,
- documentation staying aligned with implementation.

## Working Rules

### 1. Build the system in dependency order

The implementation sequence should be:

1. repository scaffold,
2. shared contracts,
3. leaf modules on stable interfaces,
4. loop integration,
5. runtime wiring,
6. interrupt end-to-end flow,
7. hardening and documentation sync.

### 2. Keep module boundaries strict

- `core` orchestrates only.
- `llm` adapts providers only.
- `tools` execute capabilities only.
- `memory` persists state only.
- `runtime` handles interaction and pause/resume only.
- `apps` contain entrypoints only.

No PR should introduce behavior that crosses these boundaries casually.

### 3. Prefer contract-first parallelism

Parallel work starts only after the shared contracts are merged.

Until then, avoid speculative implementations of:

- tool result payloads,
- model response schemas,
- message structures,
- interrupt metadata,
- approval decision shapes.

### 4. Keep the MVP narrow

The local coding path is the mainline MVP:

- `Read`
- `Write`
- `Edit`
- `Bash`
- `Glob`
- `Grep`
- one OpenAI-like provider
- local CLI runtime

`WebSearch`, `WebFetch`, and hardening work should not block the first usable coding agent.

## Milestones

### M0: Repository Ready

Success criteria:

- package installs,
- tests run,
- directory structure exists,
- developers can start independent PRs.

### M1: Contracts Stable

Success criteria:

- shared types and protocols are merged,
- downstream PRs do not need to invent cross-module interfaces,
- `ToolResult`, message shapes, and model schemas are stable.

### M2: Local Coding MVP

Success criteria:

- agent can read files, edit files, and run shell commands,
- main loop can orchestrate tool use,
- session persistence works,
- CLI can run a basic local coding session.

### M3: Human-in-the-Loop Flow

Success criteria:

- `AskUserQuestion` pauses execution,
- runtime surfaces a structured question,
- the loop resumes after the user answers,
- resumed execution preserves context correctly.

### M4: External Info and Hardening

Success criteria:

- web tools are implemented without changing loop semantics,
- end-to-end coverage exists for the main execution paths,
- docs and implementation agree on approvals, persistence, and interrupts.

## Dependency Map

```text
PR0  -> PR1
PR1  -> PR2, PR3, PR4, PR5, PR6, PR7, PR8
PR2  -> PR9
PR1  -> PR9
PR2, PR3, PR8, PR9 -> PR10
PR4, PR10 -> PR11
PR4, PR10, PR11 -> PR12
PR3, PR8 -> PR13
PR11, PR12, PR13 -> PR14
```

## PR Backlog

### PR0: Repository Scaffold

Branch:

- `codex/pr0-scaffold`

Goal:

- create the minimum Python project scaffold for `agentlet`.

Primary files:

- `pyproject.toml`
- `agentlet/__init__.py`
- `apps/__init__.py`
- `tests/unit/test_smoke.py`
- optional minimal `README.md`

Scope:

- package layout,
- test layout,
- minimal developer commands,
- import smoke coverage.

Out of scope:

- loop logic,
- provider logic,
- tool implementations,
- persistence logic.

Acceptance criteria:

- `uv run pytest` succeeds,
- package import works,
- dependency surface stays minimal.

Suggested tests:

- import smoke,
- package metadata smoke.

AI prompt:

```text
Implement PR0 for agentlet.

Goal:
Create the minimum Python project scaffold for a small agent framework. Do not implement architecture logic yet.

Constraints:
- Follow docs/ARCHITECTURE.md and AGENTS.md.
- Keep dependencies minimal.
- No new tools, no runtime logic, no provider logic.

Deliver:
- pyproject.toml
- package directories under agentlet/
- apps/ and tests/ skeleton
- one smoke test that imports the package
```

### PR1: Shared Contracts

Branch:

- `codex/pr1-shared-contracts`

Goal:

- lock the core shared types and protocols that all later PRs depend on.

Primary files:

- `agentlet/core/messages.py`
- `agentlet/core/types.py`
- `agentlet/tools/base.py`
- `agentlet/llm/base.py`
- `agentlet/llm/schemas.py`

Scope:

- normalized message structures,
- tool call structures,
- `ToolResult`,
- `Tool` protocol,
- `ModelClient` protocol,
- model-facing request/response schemas,
- interrupt-capable metadata shapes.

Out of scope:

- concrete tool implementations,
- concrete provider implementation,
- runtime/CLI behavior,
- persistence behavior.

Acceptance criteria:

- downstream modules can implement against these contracts without redefining them,
- contracts are provider-agnostic,
- contracts do not leak CLI, transport, or file-layout behavior.

Suggested tests:

- dataclass serialization smoke,
- schema round-trip tests,
- protocol conformance smoke.

AI prompt:

```text
Implement PR1 for agentlet: shared contracts only.

Goal:
Define the stable contracts for core, llm, and tools so downstream PRs can work in parallel.

Files:
- agentlet/core/messages.py
- agentlet/core/types.py
- agentlet/tools/base.py
- agentlet/llm/base.py
- agentlet/llm/schemas.py

Requirements:
- Define a single normalized ToolResult aligned with docs/ARCHITECTURE.md.
- Define provider-agnostic message and tool-call structures.
- Define a ModelClient protocol that does not know about session policy or filesystem state.
- Define interrupt-friendly result metadata without implementing runtime behavior.
```

### PR2: Memory Stores

Branch:

- `codex/pr2-memory-stores`

Goal:

- implement readable file-backed persistence.

Primary files:

- `agentlet/memory/session_store.py`
- `agentlet/memory/memory_store.py`

Scope:

- JSONL append-only session history,
- markdown durable memory,
- simple read/write APIs,
- missing-file behavior,
- malformed-record handling.

Out of scope:

- context assembly,
- loop behavior,
- CLI behavior.

Acceptance criteria:

- session history is append-only and readable on disk,
- durable memory is plain markdown,
- persistence layer does not depend on runtime or provider code.

Suggested tests:

- append/read round-trip,
- absent file behavior,
- malformed JSONL handling.

AI prompt:

```text
Implement PR2 for agentlet: file-backed memory stores.

Goal:
Provide simple local persistence aligned with the architecture:
- JSONL append-only session history
- Markdown durable memory

Requirements:
- Keep storage human-readable and debuggable.
- Do not introduce databases or heavy abstractions.
- Do not make policy decisions that belong to runtime or core loop.
```

### PR3: Tool Registry and Approval Policy

Branch:

- `codex/pr3-registry-approvals`

Goal:

- centralize tool lookup and risk-based approval decisions.

Primary files:

- `agentlet/tools/registry.py`
- `agentlet/core/approvals.py`

Scope:

- tool registration,
- name-based resolution,
- duplicate registration failure,
- tool risk grouping,
- approval decision objects.

Out of scope:

- tool execution,
- runtime prompting,
- CLI interactions.

Acceptance criteria:

- risk groups match the architecture doc,
- approval logic stays out of tool implementations,
- runtime interaction is not embedded in `core.approvals`.

Suggested tests:

- register/resolve,
- duplicate detection,
- unknown tool behavior,
- decision matrix coverage.

AI prompt:

```text
Implement PR3 for agentlet: tool registry and approval policy.

Goal:
Centralize tool lookup and tool-risk decisions.

Requirements:
- Registry resolves built-in tools by stable name.
- Approval policy maps tool categories to decisions.
- Keep runtime prompting out of this layer; return structured decisions only.
```

### PR4: Runtime Interaction Contracts

Branch:

- `codex/pr4-runtime-contracts`

Goal:

- define how approvals, structured questions, and resume behavior cross the runtime boundary.

Primary files:

- `agentlet/runtime/events.py`
- `agentlet/runtime/user_io.py`

Scope:

- event dataclasses,
- approval request payloads,
- structured question payloads,
- resume payloads,
- `UserIO` protocol.

Out of scope:

- CLI implementation,
- loop integration,
- concrete `AskUserQuestion` tool behavior.

Acceptance criteria:

- interrupt and resume payloads are explicit and stable,
- `AskUserQuestion` is modeled as structured runtime interaction,
- no transport-specific logic leaks into core.

Suggested tests:

- event schema smoke,
- protocol contract smoke.

AI prompt:

```text
Implement PR4 for agentlet: runtime interaction contracts.

Goal:
Define the runtime-facing contracts for approvals, structured questions, and resume behavior.

Requirements:
- AskUserQuestion must be modeled as an interrupt/resume flow, not generic chat.
- Define runtime events and payloads for approvals and user questions.
- Define a UserIO protocol or interface for the app layer to implement.
```

### PR5: Read, Glob, Grep

Branch:

- `codex/pr5-readonly-fs-tools`

Goal:

- implement the read-only filesystem tool set.

Primary files:

- `agentlet/tools/fs/read.py`
- `agentlet/tools/fs/glob.py`
- `agentlet/tools/fs/grep.py`

Scope:

- file reads,
- optional line ranges,
- output truncation,
- glob path matches,
- regex matches with location context.

Out of scope:

- writes,
- approval behavior,
- loop integration.

Acceptance criteria:

- `Read` does not mutate files,
- `Glob` returns paths only,
- `Grep` returns path, line number, and snippet,
- all tools return normalized `ToolResult`.

Suggested tests:

- file read success/failure,
- line range handling,
- truncation behavior,
- glob matches,
- grep match/no-match behavior.

AI prompt:

```text
Implement PR5 for agentlet: read-only filesystem tools.

Goal:
Implement Read, Glob, and Grep as built-in tools using the shared Tool protocol and ToolResult.

Requirements:
- Read: file content only, optional line ranges, truncation support
- Glob: matching file paths only
- Grep: regex matches with path, line number, and snippet
- Use normalized ToolResult consistently
```

### PR6: Write and Edit

Branch:

- `codex/pr6-write-edit-tools`

Goal:

- implement safe file creation and precise file modification semantics.

Primary files:

- `agentlet/tools/fs/write.py`
- `agentlet/tools/fs/edit.py`

Scope:

- create new files,
- explicit overwrite behavior,
- exact-text replace or similarly precise editing,
- clear failure on context mismatch.

Out of scope:

- approval policy,
- broad fuzzy patching,
- loop integration.

Acceptance criteria:

- `Write` does not silently overwrite,
- `Edit` fails clearly when context does not match,
- mutating tools return normalized `ToolResult`.

Suggested tests:

- create new file,
- overwrite denied without flag,
- exact match edit success,
- exact match edit failure,
- missing file behavior.

AI prompt:

```text
Implement PR6 for agentlet: mutating filesystem tools.

Goal:
Implement Write and Edit with safe-by-default semantics.

Requirements:
- Write creates files; overwrite must require an explicit flag.
- Edit performs precise modifications to existing files.
- Edit must fail clearly if the target context does not match.
```

### PR7: Bash Tool

Branch:

- `codex/pr7-bash-tool`

Goal:

- implement structured command execution.

Primary files:

- `agentlet/tools/exec/bash.py`

Scope:

- command execution,
- working directory support,
- timeout handling,
- stdout/stderr capture,
- exit code metadata.

Out of scope:

- approval prompting,
- CLI formatting,
- loop orchestration.

Acceptance criteria:

- success and failure are returned structurally,
- timeout is handled explicitly,
- metadata contains execution details needed by the runtime.

Suggested tests:

- success command,
- non-zero exit,
- timeout,
- cwd behavior.

AI prompt:

```text
Implement PR7 for agentlet: Bash tool.

Goal:
Provide a built-in Bash tool that runs commands in a working directory and returns structured results.

Requirements:
- Support cwd and timeout
- Return stdout, stderr, exit code, and execution metadata
- Do not perform approval prompting here
```

### PR8: OpenAI-like Provider Adapter

Branch:

- `codex/pr8-openai-like-provider`

Goal:

- implement one concrete provider adapter without leaking session or runtime policy into the LLM layer.

Primary files:

- `agentlet/llm/openai_like.py`

Scope:

- request building,
- response parsing,
- tool call normalization,
- assistant message normalization.

Out of scope:

- tool execution,
- session persistence,
- runtime interrupts,
- CLI behavior.

Acceptance criteria:

- provider implements `ModelClient`,
- tool calls are normalized into shared schemas,
- provider code does not know about stores or runtime interaction.

Suggested tests:

- request build,
- final message parse,
- tool call parse,
- malformed response handling.

AI prompt:

```text
Implement PR8 for agentlet: OpenAI-like model provider adapter.

Goal:
Create one concrete provider adapter that implements the shared ModelClient protocol.

Requirements:
- Adapt provider request/response shapes into agentlet shared schemas
- Support assistant final messages and tool-call outputs
- Keep provider code unaware of session storage, memory policy, and runtime interrupts
```

### PR9: Context Builder

Branch:

- `codex/pr9-context-builder`

Goal:

- implement deterministic context assembly for model input.

Primary files:

- `agentlet/core/context.py`

Scope:

- system instructions,
- session history,
- durable memory,
- current task state,
- context ordering.

Out of scope:

- model calls,
- tool execution,
- persistence,
- CLI behavior.

Acceptance criteria:

- context assembly is easy to read and test,
- output is provider-agnostic normalized messages,
- no runtime behavior is hidden here.

Suggested tests:

- empty context,
- history + memory ordering,
- current task inclusion,
- optional interrupt/resume context if required by settled contracts.

AI prompt:

```text
Implement PR9 for agentlet: ContextBuilder.

Goal:
Build the model input context from system instructions, session history, durable memory, and the current task state.

Requirements:
- Keep this module focused on context assembly only
- Produce normalized messages suitable for the ModelClient
- Make ordering explicit and easy to reason about
```

### PR10: Agent Loop

Branch:

- `codex/pr10-agent-loop`

Goal:

- implement the thin orchestration loop at the center of the architecture.

Primary files:

- `agentlet/core/loop.py`

Scope:

- load state,
- build context,
- call model,
- validate tool calls,
- apply approval policy,
- execute tools,
- append tool results,
- persist turns,
- return final response or interrupt outcome.

Out of scope:

- direct user prompting,
- CLI rendering,
- provider-specific business logic.

Acceptance criteria:

- the loop remains short and readable,
- tools do not control the loop directly,
- interrupts are returned structurally instead of handled inline via terminal I/O.

Suggested tests:

- final response without tools,
- single tool call,
- multi-tool iteration,
- tool error path,
- approval-required path,
- interrupt path,
- persistence interactions.

AI prompt:

```text
Implement PR10 for agentlet: core AgentLoop.

Goal:
Create the thin orchestration loop defined by docs/ARCHITECTURE.md.

Requirements:
- Load state
- Build context
- Call the model
- Validate and execute tool calls
- Apply centralized approval policy
- Append tool results to working messages
- Persist turns
- Return a structured interrupt outcome when a tool indicates interrupt

Critical constraints:
- No CLI logic
- No direct user prompting
- Tools do not own loop control
```

### PR11: Runtime App Wiring and CLI

Branch:

- `codex/pr11-cli-runtime`

Goal:

- wire the architecture into a runnable terminal app without collapsing boundaries.

Primary files:

- `agentlet/runtime/app.py`
- `apps/cli.py`

Scope:

- component assembly,
- runtime wiring,
- terminal interaction,
- surfacing approvals and normal responses.

Out of scope:

- new business logic in CLI,
- ad hoc tool execution paths,
- provider-specific hacks.

Acceptance criteria:

- the app can run a minimal session,
- CLI uses runtime contracts rather than bypassing them,
- assembly stays separate from orchestration semantics.

Suggested tests:

- app assembly smoke,
- CLI round-trip with fake components,
- approval rendering smoke.

AI prompt:

```text
Implement PR11 for agentlet: runtime app wiring and CLI entrypoint.

Goal:
Assemble the configured loop, registry, stores, model client, and user IO into a working terminal app.

Requirements:
- Keep runtime assembly separate from core orchestration
- CLI handles terminal interaction only
- Use runtime/user_io contracts rather than bypassing them
```

### PR12: AskUserQuestion End-to-End

Branch:

- `codex/pr12-ask-user-question`

Goal:

- implement and verify the architecture's special interrupt flow.

Primary files:

- `agentlet/tools/interaction/ask_user_question.py`
- minimal contract-aligned glue in existing runtime/core modules if needed

Scope:

- structured question payload,
- interrupt result,
- runtime pause,
- user answer capture,
- resume path,
- resumed context continuity.

Out of scope:

- generic chat UI,
- new planning abstractions,
- runtime redesign.

Acceptance criteria:

- `AskUserQuestion` pauses the loop,
- the question is surfaced as structured data,
- resume continues the same task coherently,
- the flow works end to end.

Suggested tests:

- interrupt result shape,
- runtime pause outcome,
- resume with choice,
- resume with free text,
- resumed completion.

AI prompt:

```text
Implement PR12 for agentlet: AskUserQuestion end-to-end interrupt flow.

Goal:
Implement the special AskUserQuestion tool and wire the pause/resume behavior end to end.

Requirements:
- AskUserQuestion is a structured interrupt, not a normal fire-and-forget tool
- Tool returns normalized ToolResult with interrupt=True
- Runtime can surface the structured question and resume after user input
- Core loop remains thin and does not absorb CLI logic
```

### PR13: WebSearch and WebFetch

Branch:

- `codex/pr13-web-tools`

Goal:

- add current-information tools without changing the core execution model.

Primary files:

- `agentlet/tools/web/search.py`
- `agentlet/tools/web/fetch.py`

Scope:

- ranked search results,
- URL fetch and content normalization,
- network failure handling.

Out of scope:

- loop changes,
- runtime changes,
- live-network-dependent tests.

Acceptance criteria:

- `WebSearch` returns ranked results, not full documents,
- `WebFetch` returns normalized usable text,
- web complexity does not leak into core or runtime.

Suggested tests:

- mocked search behavior,
- mocked fetch behavior,
- normalization coverage,
- failure path coverage.

AI prompt:

```text
Implement PR13 for agentlet: web tools.

Goal:
Add WebSearch and WebFetch as built-in tools without changing core loop semantics.

Requirements:
- WebSearch returns ranked search results, not full documents
- WebFetch retrieves a specific URL and normalizes usable text output
- Keep networking concerns inside the tool layer
```

### PR14: Hardening and Documentation Sync

Branch:

- `codex/pr14-hardening-docs`

Goal:

- close the MVP with end-to-end coverage and documentation that matches the real behavior.

Primary files:

- `docs/ARCHITECTURE.md`
- integration and end-to-end tests
- example files or transcript fixtures as needed

Scope:

- end-to-end transcript tests,
- unhappy-path coverage,
- persistence layout documentation,
- approval and interrupt documentation,
- architecture doc sync.

Out of scope:

- major new features,
- architecture expansion beyond the documented design.

Acceptance criteria:

- implementation and docs agree,
- a new contributor can follow the loop and runtime flow,
- mainline and failure paths have meaningful test coverage.

Suggested tests:

- end-to-end local coding flow,
- approval refusal path,
- interrupt/resume path,
- documented persistence files created as expected.

AI prompt:

```text
Implement PR14 for agentlet: hardening, end-to-end tests, and documentation sync.

Goal:
Close the MVP by ensuring implementation, tests, and architecture docs are aligned.

Requirements:
- Add focused end-to-end tests around the main loop
- Document persistence layout, approvals, and AskUserQuestion pause/resume behavior
- Update docs/ARCHITECTURE.md if implementation details changed in any justified way
```

## Parallelization Plan

### Wave 1: After PR1 merges

These PRs can proceed in parallel:

- PR2: memory stores
- PR3: registry and approvals
- PR4: runtime contracts
- PR5: read-only file tools
- PR6: write/edit tools
- PR7: bash tool
- PR8: OpenAI-like provider

### Wave 2: After PR2 and PR1 are stable

- PR9: context builder

### Wave 3: Integration sequence

- PR10: agent loop
- PR11: app wiring and CLI
- PR12: `AskUserQuestion` end-to-end

### Wave 4: Non-blocking extension and closeout

- PR13: web tools
- PR14: hardening and docs

## Suggested Team Split

### Three AI coders

- AI-A: PR2, PR9
- AI-B: PR3, PR5, PR6
- AI-C: PR4, PR7, PR8
- strongest reviewer or human lead: PR10, PR11, PR12, PR14

### Five AI coders

- AI-A: persistence
- AI-B: registry and approvals
- AI-C: filesystem tools
- AI-D: bash and provider
- AI-E: runtime contracts
- lead owner: loop, CLI, interrupt flow, hardening

## Standard PR Template

Every implementation PR should follow this structure:

```text
Title:
codex/prX-<topic>

Goal:
One sentence describing the outcome.

In scope:
- ...

Out of scope:
- ...

Files expected:
- ...

Acceptance criteria:
- ...

Tests:
- ...

Architecture constraints:
- core must not ...
- runtime must not ...
- tools must not ...

Notes for reviewer:
- follow-ups intentionally deferred
```

## Review Checklist

Use this checklist on every PR:

1. Does the change stay within its declared module boundary?
2. Did `core` avoid absorbing CLI or transport logic?
3. Did `tools` avoid taking ownership of loop control?
4. Did `llm` avoid taking ownership of session or memory policy?
5. Did `runtime` avoid becoming a second orchestration layer?
6. Is on-disk state still readable and debuggable?
7. Did the PR avoid introducing a second abstraction for the same concern?
8. If behavior changed, were docs updated in the same PR?

## Daily Execution Flow

Recommended working loop for AI-assisted development:

1. Open or update the next PR brief from this plan.
2. Assign one PR per AI coder.
3. Require each AI to stay inside the PR's declared scope.
4. Run the PR's focused tests before review.
5. Use a separate AI or human reviewer to check boundary violations and regression risk.
6. Merge only after the declared acceptance criteria are satisfied.
7. Rebase or refresh dependent PR prompts after each merge in the dependency chain.

## Known Failure Modes

Watch closely for these recurring risks:

- PR1 under-specifies contracts and forces downstream churn.
- PR10 absorbs runtime behavior and becomes too large.
- PR11 bypasses the runtime contracts and couples CLI directly to internals.
- PR12 treats `AskUserQuestion` like a normal tool instead of an interrupt.
- web tooling expands the abstraction surface too early.

If any of these show up, stop and refactor before continuing the next wave.
