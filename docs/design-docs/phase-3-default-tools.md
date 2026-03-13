# Phase 3 Default Tools Design

Status: draft

## 1. Context

Phase 1 established the basic tool-calling loop:

- providers return normalized `ToolCall` values
- `ToolRegistry` exposes tool schemas and executes tool calls
- `Context` records tool results as regular messages

Phase 2 made the harness usable for sustained local sessions through an interactive CLI, streaming output, and persisted transcripts.

What is still missing is a default tool suite that makes the harness practically useful out of the box. Today the runtime can execute tools, but it does not yet define:

- which tools ship by default
- how local filesystem and shell access are constrained
- how web search and web fetching are separated
- how tool results should be shaped so models can reason over them reliably

Phase 3 should define a small built-in tool set:

- `Read`
- `Write`
- `Edit`
- `Bash`
- `Glob`
- `Grep`
- `WebSearch`
- `WebFetch`

## 2. Goals

Phase 3 must deliver:

1. A default built-in tool set that covers the common local-agent workflow:
   - inspect files
   - search the repo
   - create or update files
   - run terminal commands
   - look up current information on the web
   - fetch and read web pages
2. A tool architecture that keeps `AgentLoop` simple and does not leak tool-specific logic into `Context` or provider adapters.
3. Explicit safety boundaries for:
   - workspace file access
   - mutating file writes
   - shell execution
   - outbound HTTP access
4. Consistent argument validation and result formatting across all built-in tools.
5. A small `Context` adaptation that keeps tool-call and tool-result messages legible and stable as the built-in tool set grows.
6. A CLI interaction model that shows tool activity in compact event lines instead of dumping raw payloads.
7. A test plan that remains mostly fake-based and deterministic.

## 3. Non-Goals

Phase 3 explicitly does not include:

- browser automation
- desktop control
- sandboxing shell commands from access outside the working directory
- patch-application or AST-edit tooling
- long-running background jobs
- a full approval UI for every individual tool call
- crawling entire sites or recursive scraping
- replacing the narrow `ToolRegistry` with a plugin framework

The target is a compact, opinionated built-in tool set, not a general agent platform.

## 4. Design Summary

Phase 3 should keep the current loop shape and add complexity at the tool edge, not in orchestration.

Recommended decisions:

- keep `ToolRegistry` as the single execution boundary
- keep `ToolResult.content` as text but formalize the shape that `Context` stores for built-in tool results
- require built-in tools to return compact JSON text envelopes so the model sees stable structure
- scope local filesystem tools to the current working directory
- separate search from fetch:
  - `WebSearch` finds candidate URLs
  - `WebFetch` retrieves and extracts readable page content
- introduce an explicit `ToolPolicy` so “built in” does not mean “unrestricted”
- reuse the existing agent-loop event stream for CLI tool activity rendering instead of creating a second tool-specific UI channel

## 5. Proposed Repository Changes

Phase 3 should extend `src/agentlet/agent/tools/` with explicit modules:

```text
src/agentlet/agent/tools/
├── registry.py
├── builtins.py
├── policy.py
├── local_fs.py
├── bash.py
└── web.py
```

Module responsibilities:

- `registry.py`: keep the small contracts and execution wiring
- `builtins.py`: build the default registry from policy and runtime settings
- `policy.py`: define workspace, network, timeout, and mutation limits
- `local_fs.py`: `Read`, `Write`, `Edit`, `Glob`, `Grep`
- `bash.py`: `Bash`
- `web.py`: `WebSearch`, `WebFetch`

This keeps names explicit and avoids a catch-all helpers module.

## 6. Tool Runtime Model

### 6.1 Tool runtime configuration

Add a small runtime config object, injected into built-in tools at construction time:

```python
@dataclass(frozen=True)
class ToolRuntimeConfig:
    cwd: Path
    bash_timeout_seconds: float = 30.0
    web_timeout_seconds: float = 10.0
    max_read_bytes: int = 64_000
    max_write_bytes: int = 128_000
    max_search_results: int = 8
    max_fetch_chars: int = 20_000
```

Design intent:

- tool safety is runtime configuration, not hidden prompt behavior
- tool constructors receive their boundaries explicitly
- tests can supply smaller limits and fake dependencies

### 6.2 Tool policy

Add a separate policy object for capability switches:

```python
@dataclass(frozen=True)
class ToolPolicy:
    allow_network: bool = True
    allow_write: bool = False
    allow_bash: bool = False
```

We should distinguish between three concepts:

- shipped tools: the full built-in tool set compiled into `agentlet`
- enabled tools: tools allowed by the local runtime policy
- advertised tools: tools actually exposed to the model in the current run

Recommended default policy:

- `Read`, `Glob`, `Grep`: enabled by default
- `WebSearch`, `WebFetch`: enabled by default when network is allowed
- `Write`, `Edit`: built in but disabled unless write access is enabled
- `Bash`: built in but disabled unless shell access is enabled

This gives a useful default while avoiding surprising mutation or shell execution.

### 6.3 Result normalization

Built-in tools should return JSON text, not free-form prose.

Example shape:

```json
{
  "ok": true,
  "tool": "read",
  "path": "src/agentlet/agent/context.py",
  "content": "...",
  "truncated": false
}
```

Why keep JSON-in-text instead of changing `ToolResult` now:

- it preserves the current `Context` and provider message model
- it gives the model predictable keys
- it avoids a premature internal protocol redesign

`ToolRegistry` should continue to raise `ToolExecutionError` for validation, policy, and execution failures. Tool errors should stay explicit rather than being returned as fake success payloads.

## 7. Context and Built-In Tool Designs

### 7.1 `Context` adaptation for tool messages

Phase 3 should keep `Context` provider-agnostic, but the growth in built-in tools makes message-shape discipline more important.

Recommended adjustments:

- keep the existing `Message(role="assistant", tool_calls=...)` representation for provider-requested tool calls
- keep the existing `Message(role="tool", name=..., tool_call_id=..., content=...)` representation for tool results
- document that built-in tool results are JSON text envelopes, not arbitrary prose
- add a small helper on `Context` or adjacent runtime code to make tool result creation explicit, rather than relying on ad hoc string conventions in each tool

Recommended helper shape:

```python
def build_tool_result_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)
```

Why this matters:

- the model sees a predictable tool-result schema across `Read`, `Bash`, `WebFetch`, and the rest
- `Context` remains ignorant of tool-specific semantics while still carrying a stable serialized form
- the CLI can parse or summarize the same JSON payload without inventing a second result model

Phase 3 should not move parsing logic into `Context`. `Context` should still only own normalized message storage and assembly.

### 7.2 `Read`

Purpose:

- read text files inside the working directory
- support narrow, line-bounded inspection instead of always reading full files

Suggested arguments:

- `path: str`
- `start_line: int | None = None`
- `end_line: int | None = None`

Rules:

- path must resolve under `cwd`
- binary files should be rejected with a clear error
- large files should be truncated by byte limit with a visible `truncated` flag
- returned line numbers should remain 1-based

Suggested result fields:

- `path`
- `content`
- `start_line`
- `end_line`
- `total_lines`
- `truncated`

### 7.3 `Write`

Purpose:

- create new files
- optionally create missing parent directories inside the workspace

Suggested arguments:

- `path: str`
- `content: str`
- `create_parents: bool = true`

Rules:

- disabled unless write access is enabled
- must fail if the target file already exists
- must fail when the path resolves outside `cwd`
- should enforce a payload size limit

Suggested result fields:

- `path`
- `bytes_written`
- `created`

### 7.4 `Edit`

Purpose:

- make precise edits to existing files without introducing a full patch parser

Suggested arguments:

- `path: str`
- `edits: list[EditOperation]`

Suggested `EditOperation` shape:

- `old_text: str`
- `new_text: str`
- `replace_all: bool = false`

Rules:

- disabled unless write access is enabled
- target file must already exist
- when `replace_all` is `false`, `old_text` must match exactly once
- when `replace_all` is `true`, zero matches should still fail
- edits are applied sequentially against the latest buffer

Suggested result fields:

- `path`
- `applied_edits`
- `total_replacements`
- `bytes_written`

This design stays simple, exact-match based, and testable. It is intentionally less ambitious than AST or unified-diff editing.

### 7.5 `Bash`

Purpose:

- run terminal commands, scripts, and common git operations starting from the workspace

Suggested arguments:

- `command: str`
- `timeout_seconds: float | None = None`

Rules:

- disabled unless shell access is enabled
- execution happens with `cwd` set to the workspace root
- commands are not sandboxed and may access resources outside `cwd` with the current user's permissions
- command timeout defaults to the runtime config limit
- stdout and stderr should be captured separately
- large outputs should be truncated with explicit byte counts
- return exit status instead of turning non-zero exits into tool exceptions

Suggested result fields:

- `command`
- `exit_code`
- `stdout`
- `stderr`
- `stdout_truncated`
- `stderr_truncated`
- `duration_ms`

Important boundary:

- command failure is not the same as tool failure
- invalid arguments, timeout, or policy denial should raise `ToolExecutionError`
- a non-zero shell exit should be a successful tool call with `exit_code != 0`

### 7.6 `Glob`

Purpose:

- find files by pattern such as `**/*.ts` or `src/**/*.py`

Suggested arguments:

- `pattern: str`
- `limit: int | None = None`

Rules:

- results are always workspace-relative
- only filesystem paths whose resolved targets remain under `cwd` are returned
- symlinks that resolve outside `cwd` are skipped
- a limit is applied to avoid overwhelming the model

Suggested result fields:

- `pattern`
- `matches`
- `truncated`

### 7.7 `Grep`

Purpose:

- search file contents with regex

Suggested arguments:

- `pattern: str`
- `glob: str | None = None`
- `limit: int | None = None`
- `case_sensitive: bool = false`

Rules:

- regex compilation errors should fail fast
- text files only
- only files whose resolved targets remain under `cwd` are searched
- symlinks that resolve outside `cwd` are skipped
- output should include enough local context to be actionable without dumping entire files

Suggested result fields:

- `pattern`
- `matches`

Suggested match item fields:

- `path`
- `line`
- `column`
- `text`

## 8. Web Tool Design

### 8.1 Why two web tools

`WebSearch` and `WebFetch` should remain separate.

Reasoning:

- search is about discovery and ranking
- fetch is about reading a specific page
- many agent tasks need one without the other
- keeping them separate makes failures and test cases clearer

### 8.2 `WebSearch`

Purpose:

- search the web for current information
- return a compact list of candidate results for follow-up fetching

Suggested arguments:

- `query: str`
- `max_results: int | None = None`
- `region: str | None = None`
- `safesearch: str | None = None`

Default implementation choice:

- use the current `ddgs` Python package
- call text search with `backend="duckduckgo"` so the default behavior matches the product requirement rather than using a multi-engine blend

Why this is the right default:

- it matches the requested DuckDuckGo-first behavior
- it avoids maintaining our own brittle scraping adapter
- it keeps search concerns isolated behind one small tool implementation

Suggested result fields:

- `query`
- `results`

Suggested result item fields:

- `title`
- `url`
- `snippet`
- `source`
- `rank`

Operational rules:

- obey a small result limit by default, for example `5`
- normalize missing fields instead of passing raw library payloads through
- treat upstream search errors as `ToolExecutionError`

### 8.3 `WebFetch`

Purpose:

- fetch a web page and extract readable content for model consumption

Suggested arguments:

- `url: str`
- `max_chars: int | None = None`

Default implementation choice:

- use `httpx` for download
- use `trafilatura` to extract readable main content and metadata from HTML

Why not raw HTML or regex extraction:

- raw HTML wastes context window and is hard for the model to use
- a main-content extractor removes navbars, footers, and boilerplate
- this is one of the few cases where an extra dependency removes real complexity

Suggested result fields:

- `url`
- `final_url`
- `status_code`
- `title`
- `content`
- `truncated`

Operational rules:

- support redirects
- reject unsupported schemes
- apply byte and character limits before returning content
- if extraction fails but fetch succeeds, return a plain-text fallback extracted from the response body when possible

## 9. CLI and Configuration Implications

Phase 3 should keep tool wiring in `cli/` and tool behavior in `agent/tools/`.

Recommended CLI-facing controls:

- `agentlet chat --allow-write`
- `agentlet chat --allow-bash`
- `agentlet chat --deny-network`

Recommended settings-file fields:

- `allow_write`
- `allow_bash`
- `allow_network`

Recommended startup behavior:

- build the default registry from CLI/settings policy
- expose only enabled tools to the model
- show the enabled tool set in `/status`

This is simpler than an interactive approval flow and fits the current CLI architecture better.

### 9.1 Interactive tool-call rendering

The interactive CLI already receives normalized loop events:

- `tool_requested`
- `tool_started`
- `tool_completed`

Phase 3 should continue using that event stream instead of special-casing built-in tools in the REPL.

Recommended rendering rules:

- show one short line when a tool starts
- show one short line when a tool completes
- never dump full JSON payloads by default
- truncate arguments and result summaries aggressively
- stop the assistant streaming block before printing tool events, then resume normal assistant rendering afterward

Example output shape:

```text
tool start  Read(path="src/agentlet/agent/context.py", start_line=1, end_line=80)
tool done   Read(path="src/agentlet/agent/context.py", lines=80, truncated=false)
tool start  WebSearch(query="latest duckdb python release", max_results=5)
tool done   WebSearch(results=5)
```

Recommended presenter behavior:

- summarize arguments by tool name and a small allowlist of important keys
- summarize results by tool-specific high-signal fields:
  - `Read`: path, line span, truncated flag
  - `Write`/`Edit`: path and replacement count
  - `Bash`: command preview and exit code
  - `Glob`/`Grep`: match count
  - `WebSearch`: result count
  - `WebFetch`: final URL, title when present, truncated flag

This keeps the terminal readable during long sessions and aligns with the current `ChatPresenter` design, which already handles tool lifecycle events separately from assistant streaming.

### 9.2 Session persistence and history

Phase 3 should keep persisting full tool messages in session transcripts, but history summaries shown to the user should remain brief.

Recommended rule:

- session storage keeps the full assistant tool-call message and full tool-result JSON text
- interactive history views continue to summarize only user and final assistant text unless a future command is added for tool inspection

This preserves replay/debug value without turning the default history view into a wall of tool output.

## 10. System Prompt Changes

The default system prompt should be expanded slightly once these tools exist. It should tell the model:

- use workspace-relative paths
- prefer `Glob` and `Grep` before broad reads
- use `WebSearch` for discovery and `WebFetch` for page content
- treat filesystem tools as workspace-bounded, but treat `Bash` as starting in `cwd` rather than sandboxed to it
- avoid mutating tools unless needed
- summarize command and tool results rather than reprinting large blobs

This should remain a small operational hint, not a long policy document.

## 11. Testing Strategy

Phase 3 should keep tests narrow and mostly fake-based.

Recommended test layers:

1. Unit tests for each local tool:
   - path normalization
   - workspace escape rejection
   - truncation behavior
   - exact-match edit semantics
2. Unit tests for `ToolRegistry`:
   - policy-denied tools are not advertised
   - invalid JSON and invalid shapes still fail at the registry boundary
3. Fake-based web tests:
   - fake search client for `WebSearch`
   - fake fetcher and fake extractor for `WebFetch`
4. Smoke tests:
   - one chat turn that reads and greps repo files
   - one chat turn that uses a fake web result and fetch path

Avoid live web tests in the default suite. If live tests are added later, they should be opt-in and clearly marked.

## 12. Dependency Changes

Recommended additions:

- [`ddgs`](https://pypi.org/project/ddgs/) for DuckDuckGo-backed search
- [`trafilatura`](https://trafilatura.readthedocs.io/en/latest/) for readable web content extraction

These additions are justified because they remove real implementation complexity that would otherwise end up as brittle scraping code in the repo.

## 13. Rollout Plan

Recommended implementation order:

1. add `ToolRuntimeConfig`, `ToolPolicy`, and default registry wiring
2. implement `Read`, `Glob`, `Grep`
3. implement `Write`, `Edit`
4. implement `Bash`
5. implement `WebSearch`
6. implement `WebFetch`
7. update prompt text, CLI flags, and tests

This sequence lands the lowest-risk tools first and keeps each step reviewable.

## 14. Open Questions

These questions can stay open until implementation starts:

- Should `Write` allow overwrite behind an explicit flag, or should overwrite remain impossible and be handled only through `Edit`?
- Should `Bash` eventually support an allowlist of commands, or is a simple on/off policy enough for phase 3?
- Should `WebFetch` return plain text only, or optionally return markdown later if that proves more model-friendly?

## 15. Recommendation

Phase 3 should proceed with one explicit built-in tool suite and a small policy layer, not an extensibility-heavy plugin architecture.

The key design choice is to preserve the current runtime core:

- `AgentLoop` stays orchestration-only
- `Context` stays provider-agnostic and side-effect-free
- built-in tools absorb filesystem, shell, and network complexity at the edge

That keeps the codebase legible while making the harness substantially more useful in real sessions.
