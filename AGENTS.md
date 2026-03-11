# AGENTS.md

This file is the repository map, not the full manual.
Keep it short, stable, and opinionated. Put detailed design, plans, and decisions in `docs/`.

## Project Intent

`agentlet` is a Python-first agent harness built from zero.

Primary goals:
- simple, legible architecture
- elegant, minimal code
- strong boundaries between pure logic and side effects
- easy local iteration, testing, and evaluation

Non-goals for the initial phase:
- premature framework adoption
- distributed systems complexity
- hidden magic, implicit globals, or deep inheritance

## Engineering Principles

1. Prefer simple building blocks over clever abstractions.
2. Keep the core deterministic; push I/O to the edges.
3. Validate data at boundaries. Do not build on guessed shapes.
4. Favor explicit types, small modules, and narrow interfaces.
5. Add dependencies only when they remove real complexity.
6. Make the repo agent-legible: clear names, clear docs, clear ownership.
7. Treat checked-in documentation as the system of record.

## Python Stack Defaults

- Python 3.12+
- Package manager: `uv`
- Source layout: `src/`
- Tests: `pytest`
- Lint/format: prefer a minimal toolchain and one obvious command per task

Default dependency posture:
- prefer stdlib first
- prefer `pydantic` or equivalent only at true input/output boundaries
- avoid large orchestration frameworks in phase 1 unless a concrete need appears

## Preferred Repository Layout

Target layout for this project:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   └── agentlet/
│       ├── agent/
│       │   ├── agent_loop.py
│       │   ├── context.py
│       │   ├── tools/
│       │   │   └── registry.py
│       │   ├── providers/
│       │   │   ├── registry.py
│       │   │   └── litellm_provider.py
│       │   └── prompts/
│       │       └── system_prompt.py
│       └── cli/
│           └── main.py
├── tests/
│   ├── unit/
│   └── smoke/
├── scripts/                # small developer utilities
└── docs/
    ├── index.md
    ├── architecture.md
    ├── design-docs/
    ├── exec-plans/
    └── references/
```

Rules for this layout:
- `agent/context.py` must stay free of provider SDK, shell, and network coupling.
- `agent/agent_loop.py` owns orchestration, not provider-specific request shaping.
- `agent/providers/litellm_provider.py` is the only place that should import `litellm`.
- `agent/providers/registry.py` and `agent/tools/registry.py` should stay small and explicit.
- `cli/` only wires dependencies and exposes commands.
- Do not add new top-level directories without a strong reason.

## Documentation Rules

Follow the “map, then progressive disclosure” approach:
- keep `AGENTS.md` concise
- put stable architecture guidance in `docs/architecture.md`
- put phase or feature design work in `docs/design-docs/`
- put implementation plans and progress logs in `docs/exec-plans/`
- put external references or distilled vendor notes in `docs/references/`

If a task changes architecture, interfaces, or operating rules, update docs in the same change.

## Coding Rules

- Prefer composition over inheritance.
- Prefer dataclasses or simple typed objects for internal state.
- Keep functions short and intention-revealing.
- Avoid boolean-flag-heavy APIs; split responsibilities instead.
- Avoid “misc”, “utils”, and catch-all modules unless the abstraction is truly shared and coherent.
- Write comments only when intent is not obvious from the code.

## Testing Rules

- Unit test `Context`, `AgentLoop`, and provider normalization first.
- Prefer fake providers and fake tools in tests over live network calls.
- Keep at least one smoke path for the end-to-end harness loop.
- When fixing a bug, add or update the narrowest test that proves it.

## Working Agreement

For non-trivial work:
1. Confirm or create the relevant doc under `docs/`.
2. Keep architectural boundaries intact while implementing.
3. Run the smallest useful test set first, then widen coverage as confidence grows.
4. Record notable design decisions in the corresponding doc, not in ad hoc chat history.

## Near-Term Focus

The first design phase should define:
- harness scope and explicit non-scope
- core runtime loop
- context and message model
- tool execution boundary
- model provider boundary
- CLI entrypoint for local testing
- evaluation strategy for early iterations
