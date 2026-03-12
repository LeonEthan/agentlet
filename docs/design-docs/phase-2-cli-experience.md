# Phase 2 CLI Experience Design

Status: draft

## 1. Context

Phase 1 established a minimal single-turn CLI and a clean runtime split between:

- `Context` for message state
- `AgentLoop` for orchestration
- provider adapters for SDK-facing work

That foundation is good enough for API validation, but not yet good enough for long manual conversations. The next phase should improve the terminal experience so `agentlet` can be used as a realistic local harness for sustained interactive testing.

The target is not a visual clone of Claude Code. The target is the same operational feel:

- interactive multi-turn chat by default in a TTY
- obvious one-shot print mode for scripting
- resumable sessions
- clear visibility into assistant progress and tool activity
- enough ergonomics that a human can stay in the loop for a long session without fighting the terminal

## 2. Goals

Phase 2 must deliver:

1. A real interactive REPL on top of the existing runtime.
2. Streaming assistant output in the terminal.
3. Session persistence and resume within the current working directory.
4. Better terminal ergonomics:
   - multiline input
   - local history
   - slash commands for session control
   - readable status and error rendering
5. A CLI architecture that keeps `agent/` free of terminal concerns.
6. A test strategy that stays mostly fake-based and does not depend on live APIs.

## 3. Non-Goals

Phase 2 explicitly does not include:

- a full-screen terminal application as the primary interface
- background agents or detached jobs
- autonomous context compaction or long-term memory
- remote session sync
- multi-pane file explorers, diff viewers, or IDE-like workflows
- replacing the existing phase-1 one-shot command path

These may become useful later, but they are not required to make long interactive evaluation practical now.

## 4. Research Summary

### 4.1 Rich

[`rich`](https://rich.readthedocs.io/en/stable/introduction.html) provides the right primitives for the output side of phase 2:

- Markdown rendering for assistant responses
- `Panel`, `Rule`, and `Table` for status and tool output
- `Live` and [`Progress`](https://rich.readthedocs.io/en/stable/progress.html) for transient streaming or activity indicators

This is a good fit for readable, incremental terminal output without introducing an application framework.

### 4.2 prompt_toolkit

[`prompt_toolkit`](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/asking_for_input.html) covers the input side that `argparse` and `input()` do not:

- `PromptSession` with persistent history
- multiline editing
- key bindings
- auto-suggestion from history

It also supports full-screen applications, but phase 2 only needs its prompt/editing capabilities.

### 4.3 Textual

[`Textual`](https://textual.textualize.io/) is strong for full-screen TUIs and has useful features such as workers, widgets, and app-level testing. It is a valid future option if `agentlet` later needs a true terminal app with panes, inspectors, or command palettes.

It is not the best default for this phase because:

- it is heavier than the current need
- it would pull more lifecycle and state concerns into the CLI layer
- it is less incremental than a `rich` + `prompt_toolkit` upgrade

### 4.4 Claude Code interaction patterns

Claude Code documentation emphasizes an interactive terminal workflow, a print-oriented non-interactive path, and conversation continuation/resume. Those patterns map directly to what `agentlet` now needs for manual evaluation.

The useful takeaway is product shape, not feature parity:

- interactive mode for humans
- print mode for scripts
- session continuity
- terminal-visible progress

## 5. Technology Decision

Phase 2 should adopt:

- `rich` for rendering
- `prompt_toolkit` for input

Phase 2 should not adopt `textual` as the primary shell.

Reasoning:

- This keeps the architecture close to phase 1.
- It improves the user-facing CLI without turning the terminal into the core abstraction.
- It fits the repo principles: small modules, narrow interfaces, explicit data flow.

Recommended dependency additions:

- `rich`
- `prompt_toolkit`

No other UI dependency should be added in phase 2 unless it removes clear complexity.

## 6. User Experience Shape

### 6.1 Command surface

Phase 2 should preserve the current `chat` command while expanding its behavior:

```bash
agentlet chat "hello"                 # one-shot, existing behavior
agentlet chat                         # interactive when stdin is a TTY
agentlet chat --continue              # resume latest session in cwd
agentlet chat --session <session-id>  # resume a specific session
agentlet chat --new-session           # force a fresh interactive session
agentlet chat --print "hello"         # explicit one-shot scripting mode
```

Defaults:

- if a message argument is provided, run one-shot mode
- if no message argument is provided and stdin is a TTY, run interactive mode
- if no message argument is provided and stdin is not a TTY, read stdin and run one-shot mode

This keeps phase-1 scripts working while making interactive usage the natural TTY path.

### 6.2 Interactive behavior

The interactive shell should be line-oriented, not full-screen.

Recommended flow:

1. Print a short session header with:
   - model
   - provider
   - cwd
   - session id
   - quick slash-command hint
2. Use `prompt_toolkit` for message entry.
3. Once the user submits a turn, suspend input and let `rich` own output until the turn completes.
4. Stream assistant text as it arrives.
5. Render tool activity as compact event lines or small panels.
6. Return to the prompt after the turn finishes.

This avoids terminal contention between the input editor and streaming renderer.

Signal handling rules:

- one `Ctrl+C` during generation cancels the active turn and keeps the session open
- one `Ctrl+C` while idle clears the current input buffer
- two consecutive `Ctrl+C` presses within a short window while idle exit the session
- `Ctrl+D` on an empty prompt exits the session cleanly

Cancellation must not partially commit a turn into the persisted transcript. Only completed turns should be appended to session storage.

### 6.3 Slash commands

Phase 2 should support a small, explicit command set:

- `/help` show interactive commands
- `/exit` leave the session
- `/status` show session and model details
- `/new` start a fresh session
- `/history` show recent turn summaries
- `/clear` clear the visible terminal without deleting conversation state

Phase 2 should not add a large command vocabulary.

### 6.4 Rendering rules

Rendering should be structured but restrained:

- assistant responses: Markdown via `rich.markdown.Markdown`
- tool execution: one compact line on start, one compact line on finish
- warnings/errors: `Panel` with clear color contrast
- status: one concise header at session start, not a permanently pinned dashboard

The UI should optimize for long scrollback readability, not visual density.

## 7. Session Model

### 7.1 Session scope

Sessions should be scoped to the current working directory. This matches the local-harness usage model and keeps resume behavior predictable.

### 7.2 Persistence format

Phase 2 should use an append-friendly filesystem format, not a database.

Recommended layout:

```text
.agentlet/
└── sessions/
    ├── latest
    └── <session-id>.jsonl
```

Why JSONL:

- simple append semantics
- easy debugging by humans
- easy replay in tests
- no extra dependency

Session identifiers should be time-sortable and filesystem-safe.

Recommendation:

- use `uuid7` when available in the target Python runtime
- otherwise use `<utc-timestamp>-<random-suffix>` with lexicographic time ordering

The session id should be generated once at session creation time and reused for the lifetime of that transcript file.

Recommended record types:

- `session_started`
- `user_message`
- `assistant_message`
- `tool_call`
- `tool_result`
- `turn_finished`

Each record should include:

- `session_id`
- `timestamp`
- `schema_version`
- a stable `type`
- the normalized payload for that event

`schema_version` should start at `1` and be checked on resume before any transcript replay. If a future version is unsupported, the CLI should fail with a clear message that the session cannot be resumed by the current build.

### 7.3 Resume behavior

Resume should rebuild a `Context` from the stored normalized transcript.

Rules:

- `--continue` loads the session pointed to by `.agentlet/sessions/latest`
- `--session <id>` loads that specific transcript
- `--new-session` ignores any previous latest pointer
- every completed interactive turn appends to the session log and updates `latest`

Phase 2 should fail loudly on malformed session files instead of guessing.

Concurrency rules:

- each interactive process creates and writes only its own `<session-id>.jsonl` file
- no two processes should append to the same session file concurrently
- `.agentlet/sessions/latest` is best-effort metadata; last successful writer wins
- writes to `latest` should be atomic via write-to-temp then rename

Phase 2 does not need cross-process locking beyond per-session file ownership, because resume targets a specific immutable session id and concurrent sessions do not share a transcript file.

## 8. Runtime and CLI Architecture

### 8.1 Keep terminal code in `cli/`

The TUI work should stay in `src/agentlet/cli/`.

Recommended module split:

```text
src/agentlet/cli/
├── main.py
├── chat_app.py        # mode selection and high-level wiring
├── repl.py            # interactive session loop
├── presenter.py       # rich rendering for turn/session events
├── prompt.py          # prompt_toolkit session factory and key bindings
├── commands.py        # slash-command parsing and dispatch
└── sessions.py        # JSONL transcript persistence and resume
```

This keeps `main.py` small and avoids turning it into a mixed parser/UI/runtime module.

### 8.2 Reuse `Context` across turns

Phase 2 should keep the existing `Context` model and reuse one context object for an interactive session. That preserves the current architecture and makes session resume a matter of rehydrating normalized messages.

No terminal code should be added to `Context`.

### 8.3 Add normalized turn events

The CLI needs structured runtime signals, not provider-specific callbacks.

Recommended addition near the runtime boundary:

- define a small `TurnEvent` model
- allow `AgentLoop.run_turn(...)` to accept an optional event sink or observer

Recommended event kinds:

- `turn_started`
- `assistant_delta`
- `assistant_completed`
- `tool_requested`
- `tool_started`
- `tool_completed`
- `turn_completed`
- `turn_failed`

The CLI presenter should render from these events only.

This keeps the TUI independent from provider details and preserves fake-based testing.

### 8.4 Add a streaming provider path

Streaming should be a provider capability, not a CLI hack.

Recommended provider contract change:

```python
class LLMProvider(Protocol):
    async def complete(...) -> LLMResponse: ...
    async def stream_complete(...) -> AsyncIterator[ProviderStreamEvent]: ...
```

Recommended normalized provider stream events:

- `content_delta(text: str)`
- `response_complete(response: LLMResponse)`

The provider adapter should assemble backend-specific stream chunks into these normalized events before the loop sees them.

This keeps tool-call reconstruction and SDK quirks inside `litellm_provider.py`.

### 8.5 One code path for final state

`AgentLoop` should keep a single source of truth for final turn state.

Implementation rule:

- streaming mode and non-streaming mode must converge on the same final `LLMResponse` handling path

That avoids drift between interactive and one-shot behavior.

## 9. Failure Handling

Phase 2 should optimize for legibility, not silent recovery.

Rules:

- provider errors render as visible CLI errors and do not corrupt the active session
- malformed tool arguments fail the turn clearly
- failed turn output is rendered, but only successful mutations are committed to the persisted transcript
- resume errors should tell the user which session file is invalid
- interrupted turns should be discarded from persisted history unless a future version introduces explicit partial-turn records

The current `AgentLoop` copy-on-success behavior should be preserved for interactive sessions.

## 10. Testing Strategy

### 10.1 Unit tests

Add focused tests for:

- session log write and resume reconstruction
- slash-command parsing
- CLI mode selection
- presenter rendering against a captured console
- streaming event handling in `AgentLoop`
- provider stream normalization with fake LiteLLM chunks

### 10.2 Smoke tests

Add smoke coverage for:

- one interactive happy path with a fake prompt source and fake provider
- resume-latest behavior
- tool activity rendering path

These tests should avoid real TTY dependencies where possible by injecting prompt and console abstractions.

### 10.3 Manual validation

Manual validation should cover:

- long multi-turn session in a real terminal
- multiline input
- session resume after process restart
- one tool-call round trip with visible progress
- fallback one-shot scripting path

## 11. Implementation Order

Recommended order:

1. Add session persistence and transcript rehydration.
2. Refactor CLI wiring out of `main.py` into smaller `cli/` modules.
3. Add runtime turn events.
4. Add provider streaming normalization.
5. Build the interactive REPL with `prompt_toolkit`.
6. Add `rich` presenter rendering.
7. Add smoke tests for interactive and resume paths.

This order keeps each step independently testable and avoids coupling the UI build-out to unfinished runtime changes.

## 12. Exit Criteria

Phase 2 is complete when all of the following are true:

- `agentlet chat` supports interactive multi-turn use in a TTY
- assistant output streams incrementally
- sessions can be resumed from the current working directory
- one-shot usage still works for scripts and stdin pipes
- tool activity is visible in the terminal
- the runtime remains free of direct `rich` and `prompt_toolkit` imports
- tests cover interactive mode selection, session resume, and streaming behavior

## 13. References

- [Rich documentation](https://rich.readthedocs.io/en/stable/introduction.html)
- [Rich Live display](https://rich.readthedocs.io/en/stable/live.html)
- [Rich Progress](https://rich.readthedocs.io/en/stable/progress.html)
- [prompt_toolkit input guide](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/asking_for_input.html)
- [prompt_toolkit full-screen applications](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/full_screen_apps.html)
- [Textual documentation](https://textual.textualize.io/)
- [Textual workers](https://textual.textualize.io/guide/workers/)
- [Anthropic Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)
- [Anthropic Claude Code common workflows](https://code.claude.com/docs/en/docs/claude-code/common-workflows)
