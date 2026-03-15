# Phase 3 Safety And Web Fetch Alignment

## Status

Implemented for current scope

Implementation update as of 2026-03-14:

- Step 1 implemented: default `max_iterations` restored to `8`; truncated HTML no longer falls back to corrupted regex stripping.
- Step 2 implemented: `max_iterations` and `max_html_extract_bytes` are configurable through settings and CLI.
- Step 3 implemented: oversized `web_fetch` results are persisted to a local artifact file and returned with `artifact_path`.
- Step 4 implemented: `ToolRegistry` now requests on-request approval for write/edit, bash, and network tools, with session-scoped `all` approval and CLI `--auto-approve`.

## Context

Recent review feedback identified two correctness regressions:

1. `AgentLoop` defaulted from a small safety budget to `max_iterations=80`.
2. `web_fetch` started truncating raw HTML before readable-text extraction.

Both changes are user-visible. The first increases the blast radius of bad tool loops,
especially now that default tools include write and bash. The second can return corrupted
content for large script-heavy pages.

This note compares `agentlet` with three mature coding-agent CLIs and proposes an
adaptation plan that fits `agentlet`'s current architecture.

Scope for this document:

- agent turn safety budget and loop control
- web fetch extraction and truncation pipeline

Out of scope:

- CLI `/history` numbering
- full approval UX implementation details
- provider-specific loop semantics

## Current `agentlet` problem shape

Current local behavior:

- `AgentLoop` now defaults to `max_iterations=80`.
- `agentlet` does not yet have a Codex/Kimi-style approval queue for write/bash/network side effects.
- `web_fetch` enforces `max_fetch_bytes` while streaming response bytes, then runs HTML extraction on the possibly truncated fragment.
- when `trafilatura.extract()` returns nothing, `web_fetch` falls back to regex-based HTML stripping.

This combination is the key mismatch:

- a high loop budget is only reasonable when there are stronger runtime guardrails around side effects
- truncating HTML before extraction is only safe if the fallback path cannot accidentally surface script/style garbage

## External references

### 1. OpenAI Codex

Observed public signals:

- Built-in approval presets pair sandbox mode with approval policy rather than relying on a tiny loop cap.
- The default preset is effectively `workspace-write + on-request approval`.
- The read-only preset is `read-only + on-request approval`.
- Tool orchestration is centralized as `approval -> sandbox selection -> execution`.
- Network access has its own approval flow, including host-scoped allow-once / allow-for-session behavior.
- Tool-output truncation happens after content is fully produced for model consumption.

Relevant sources:

- `openai/codex`: `codex-rs/utils/approval-presets/src/lib.rs`
- `openai/codex`: `codex-rs/core/src/tools/orchestrator.rs`
- `openai/codex`: `codex-rs/core/src/tools/network_approval.rs`
- `openai/codex`: `codex-rs/core/src/tools/mod.rs`
- `openai/codex`: `codex-rs/README.md`

Takeaway for `agentlet`:

- Codex does not justify a larger default loop budget by itself.
- Codex's safety story comes from approvals, sandboxing, and interruptibility.
- The transferable principle is not "allow 80 steps"; it is "centralize permissions and only relax budgets when runtime guardrails exist."

### 2. Anthropic Claude Code

Observed public signals:

- The public repo exposes an opt-in `ralph-loop` plugin, not the main product loop implementation.
- `ralph-loop` supports `--max-iterations`; if omitted it can run indefinitely, but only as an explicit user-enabled mode with loud warnings.
- Large tool outputs are no longer truncated inline; they are persisted to disk and exposed via file references.
- WebFetch binary responses are saved alongside a summary rather than dumped raw into context.

Relevant sources:

- `anthropics/claude-code`: `plugins/ralph-wiggum/hooks/stop-hook.sh`
- `anthropics/claude-code`: `plugins/ralph-wiggum/scripts/setup-ralph-loop.sh`
- `anthropics/claude-code`: `CHANGELOG.md`

Takeaway for `agentlet`:

- Long-running autonomous looping should be opt-in, not the default.
- Large fetched content should be persisted or referenced, not crudely chopped before processing.
- Claude Code's public signals support a two-tier model:
  - normal mode stays conservative
  - explicit autonomous mode can raise limits with clear operator intent

### 3. Moonshot Kimi CLI

Observed public signals:

- `LoopControl` is explicit config: `max_steps_per_turn=100`, `max_retries_per_step=3`, `max_ralph_iterations=0`.
- `default_yolo=false`; approvals are on by default unless the user enables YOLO/auto-approve.
- The main loop raises `MaxStepsReached` once the per-turn step budget is exceeded.
- `FetchURL` prefers a fetch service when configured, otherwise performs local fetch.
- Local fetch reads the full response text first, then runs `trafilatura.extract(...)`.
- If extraction fails, Kimi returns an explicit extraction failure instead of falling back to naive raw HTML stripping.

Relevant sources:

- `MoonshotAI/kimi-cli`: `src/kimi_cli/config.py`
- `MoonshotAI/kimi-cli`: `src/kimi_cli/soul/kimisoul.py`
- `MoonshotAI/kimi-cli`: `src/kimi_cli/soul/approval.py`
- `MoonshotAI/kimi-cli`: `src/kimi_cli/tools/web/fetch.py`
- `MoonshotAI/kimi-cli`: `docs/en/configuration/config-files.md`
- `MoonshotAI/kimi-cli`: `docs/en/customization/agents.md`

Takeaway for `agentlet`:

- A high step cap can be acceptable when paired with default human approval.
- For web fetch, full-document extraction is preferable to "truncate first, clean up later."
- If extraction on HTML fails, explicit failure is safer than returning misleading garbage.

## Comparative summary

| System | Default loop posture | Side-effect guardrail | Large content posture | Implication for `agentlet` |
| --- | --- | --- | --- | --- |
| Codex | not justified by a tiny hardcoded cap; relies on runtime controls | strong sandbox + on-request approval by default | truncate after full tool output exists | do not copy large budgets without approval/sandbox |
| Claude Code | autonomous looping is opt-in | product has permissions; public plugin warns loudly | persist large outputs to disk/file refs | make long loops opt-in; persist large fetch results |
| Kimi CLI | high step cap | approvals default on; YOLO opt-in | full fetch, then extract; explicit failure if extraction fails | keep extraction before truncation; budgets can be larger only with approval |

## Proposed direction for `agentlet`

## A. Agent loop safety

### A1. Immediate correction

Restore a small default `max_iterations` in `AgentLoop`.

Recommended default:

- `max_iterations = 8`

Reasoning:

- this matches `agentlet`'s earlier phase-1 design
- it is appropriate for a runtime that currently lacks approval gating
- it limits damage from malformed provider tool loops

This is the direct fix for the current regression.

### A2. Make larger budgets opt-in

If longer runs are needed, expose them explicitly through config or CLI rather than raising the default.

Recommended shape:

```python
@dataclass(frozen=True)
class AgentRuntimeSafety:
    max_iterations: int = 8
    max_retries_per_provider_call: int = 1
    autonomous_mode: bool = False
```

Possible future CLI surface:

- `--max-iterations N`
- `--autonomous`

Rules:

- normal interactive mode keeps `max_iterations=8`
- `--autonomous` may raise the default, for example to `24` or `40`
- values above the default should be visible in `/status` or startup output

### A3. Do not borrow Codex/Kimi numbers without borrowing their guardrails

This is the key design constraint.

Do not adopt `80` or `100` as the default until `agentlet` has at least one of:

- on-request approval for write/bash/network side effects
- a stricter default tool profile that excludes mutating tools
- per-turn mutating-action budgets

Recommended staged path:

1. now: restore `8`
2. next: add approval plumbing for mutating tools
3. later: consider a higher autonomous/profile-specific budget

### A4. Centralize side-effect policy before growing budgets

Borrow the architectural pattern from Codex and Kimi:

- central loop budget lives in one runtime config object
- approvals happen in one place rather than inside ad hoc tools
- "unsafe but convenient" modes are explicit

Concretely, `ToolPolicy` and runtime config should eventually distinguish:

- advertised tools
- enabled tools
- approved tool executions

Until that exists, the loop budget should remain conservative.

## B. Web fetch extraction and truncation

### B1. Principle

Readable-content extraction must happen before model-facing truncation.

Transport limits, extraction limits, and model-context limits are different concerns:

- transport limit: protects memory and latency
- extraction limit: ensures we have enough document to derive readable text
- model limit: controls how much extracted text we inline back into the conversation

The current regression mixes transport limiting with extraction.

### B2. Immediate correction

Change `web_fetch` so that HTML is not naively truncated before fallback extraction.

Recommended minimal rule set:

1. Keep `max_fetch_bytes` as a transport safeguard.
2. For HTML, try `trafilatura` on the fullest available document.
3. If the HTML body was transport-truncated and extraction fails, do not run the regex fallback on the truncated fragment.
4. Return an explicit partial/extraction failure instead.

This avoids returning raw JS/CSS while still keeping the byte cap.

Suggested failure message:

> HTML exceeded fetch byte limit before readable text could be extracted; retry with a larger fetch budget.

This is closer to Kimi's behavior than the current `agentlet` fallback.

### B3. Preferred medium-term pipeline

Adopt a two-stage fetch pipeline:

1. Fetch bytes with a hard cap.
2. For HTML responses, spool bytes to a temporary file or larger HTML buffer up to a dedicated `max_html_extract_bytes`.
3. Run extraction on the full buffered HTML.
4. Apply `max_chars` only to extracted readable text.
5. If extracted text is still large, persist the full extracted result and inline only a summary or excerpt.

Recommended config split:

```python
@dataclass(frozen=True)
class ToolRuntimeConfig:
    ...
    max_fetch_chars: int = 20_000
    max_fetch_bytes: int = 512_000
    max_html_extract_bytes: int = 2_000_000
    persist_large_fetch_results: bool = True
```

This mirrors the separation seen across mature tools:

- Kimi: full response text before extraction
- Claude Code: persist large tool output instead of truncating inline
- Codex: truncate only at model-formatting time

### B4. Fallback behavior should depend on document completeness

Recommended decision table:

| Condition | Behavior |
| --- | --- |
| plain text or markdown | decode full available body, then apply output truncation |
| HTML, extraction succeeds | return extracted content, then apply output truncation |
| HTML, extraction fails, body complete | use fallback HTML-to-text stripper |
| HTML, extraction fails, body truncated | return explicit extraction failure or partial-result notice, not fallback raw text |

This is the most important correction for correctness.

### B5. Prefer persistence over aggressive inline truncation

Borrow the Claude Code pattern for large outputs:

- store large fetch artifacts under a runtime-managed temp directory
- return a compact inline preview plus metadata
- include a local artifact path when available

Example result shape:

```json
{
  "ok": true,
  "tool": "web_fetch",
  "url": "...",
  "final_url": "...",
  "title": "...",
  "content": "short inline excerpt",
  "artifact_path": "/tmp/agentlet-fetch/abc123.txt",
  "truncated": true,
  "extraction_status": "complete"
}
```

This improves both correctness and context efficiency.

## Recommended implementation order

### Step 1

Fix the regressions only:

- restore `AgentLoop(max_iterations=8)`
- change HTML fetch fallback so truncated HTML does not fall through to regex stripping

### Step 2

Add explicit runtime knobs:

- CLI/config override for `max_iterations`
- separate `max_html_extract_bytes`

### Step 3

Add artifact persistence for large fetch results.

### Step 4

Design an approval layer for write/bash/network, then reconsider higher loop budgets.

## Decision summary

Recommended decisions:

- revert the default loop budget to `8`
- make any larger loop budget opt-in
- do not justify a larger default by referencing Codex/Kimi unless `agentlet` also adopts approval-style guardrails
- move web fetch truncation to after readable-text extraction
- if HTML extraction fails on a truncated document, fail explicitly instead of returning corrupted fallback text
- treat large fetch content as an artifact problem, not just a truncation problem

## Why this is the right borrowing strategy

The mature tools are converging on the same underlying pattern:

- autonomy is bounded by permissions, not just counters
- explicit long-running modes are opt-in
- content truncation is a presentation concern, not an extraction concern
- large outputs should degrade into references or artifacts, not corruption

`agentlet` should borrow those principles directly, but keep the implementation
small and phase-appropriate.
