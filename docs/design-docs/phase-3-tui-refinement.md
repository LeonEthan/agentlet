# Phase 3 TUI Refinement Design

Status: draft

## 1. Context

Phase 2 delivered a functional interactive REPL with streaming output, session persistence, and slash commands. The current TUI uses `rich` effectively but suffers from visual clutter:

- Heavy Panel borders around every information block
- Table-based layouts that feel like configuration files
- Verbose tool execution labels (`[cyan]tool start[/]`, `[green]tool done[/]`)
- Section headers (Rule) that interrupt content flow

The goal of Phase 3 is to evolve the terminal experience from "functional" to "elegant" by adopting the visual restraint and content-first philosophy of Claude Code.

## 2. Goals

Phase 3 must deliver:

1. A cleaner, more minimalist visual design
2. Reduced visual noise (fewer borders, less color)
3. More elegant information hierarchy through typography and spacing
4. Subtle but clear tool execution indicators
5. A refined color palette inspired by Claude Code
6. Preservation of all Phase 2 functionality

## 3. Non-Goals

Phase 3 explicitly does not include:

- Switching to a full-screen TUI framework (Textual)
- Adding new interactive features or commands
- Changing the session persistence model
- Modifying the provider or agent loop architecture
- Custom themes or user-configurable colors (may be added later)

## 4. Design Philosophy

### 4.1 Content First

AI responses are the primary content. UI chrome should recede into the background.

**Principles:**
- Text breathes with generous line spacing
- Borders are eliminated in favor of whitespace
- Color serves meaning, not decoration

### 4.2 Restraint

Every visual element must earn its place.

**Principles:**
- One Panel is too many unless absolutely necessary
- Colors from a muted, purposeful palette
- No labels stating the obvious ("assistant", "tool")

### 4.3 Clarity Through Subtlety

Status and progress should be felt, not shouted.

**Principles:**
- Spinners for ongoing work
- Icons for quick state recognition
- Dimmed text for secondary information

## 5. Visual Design System

### 5.1 Color Palette

Replace `rich` default colors with a refined palette:

```python
# Color scheme inspired by Claude Code
COLORS = {
    # Primary scale (grays)
    "dim": "#6B7280",           # Secondary text, hints
    "muted": "#9CA3AF",         # Disabled, tertiary
    "primary": "#374151",       # Primary text, borders

    # Semantic colors (subtle)
    "accent": "#D97706",        # User prompt, highlights
    "success": "#059669",       # Success states
    "error": "#DC2626",         # Errors
    "tool": "#7C3AED",          # Tool indicators (purple)

    # Background (for rare use)
    "bg_subtle": "#F3F4F6",     # Light gray backgrounds
}
```

**Usage rules:**
- 80% of output uses default terminal color (no explicit styling)
- `dim` for metadata, hints, file paths
- `accent` sparingly for the prompt indicator only
- `success/error/tool` only for their specific semantic purposes
- No background colors except for error banners

### 5.2 Typography

**Prompt Indicator:**
```
# Before
>  # plain chevron

# After
›  # right-pointing angle, amber accent color
```

**Headers:**
```
# Before (Phase 2)
╭──────── agentlet chat ────────╮
│ provider  anthropic           │
│ model     claude-sonnet-4-6   │
│ cwd       /path/to/project    │
│ session   2025-...            │
╰───────────────────────────────╯

# After (Phase 3)
agentlet · claude-sonnet-4-6 · /path/to/project · /help
```

**Assistant Output:**
```
# Before (Phase 2)
───────────────────────────────── assistant ─────────────────────────────────
[Markdown content in Panel]

# After (Phase 3)
[Pure Markdown content, no container]
```

### 5.3 Spacing

Use consistent vertical rhythm:

```python
# Spacing scale
SPACING = {
    "xs": 0,    # No blank line (compact)
    "sm": 1,    # Single blank line
    "md": 2,    # Double blank line (section break)
}

# Rules:
# - After prompt submission: xs (immediate start)
# - After assistant response: sm (breathing room)
# - After tool execution block: sm
# - Between session header and first prompt: sm
```

## 6. Component Specifications

### 6.1 Session Header

**Current (Phase 2):**
- `Table.grid` inside `Panel`
- 4 rows of key-value pairs
- Blue border

**Proposed (Phase 3):**
```
agentlet · claude-sonnet-4-6 · /Users/cuizhengliang/projects/demo · /help for commands
```

**Implementation:**
- Single line of text
- Use `·` (middle dot) as separator
- Color: `dim` for entire line
- Truncate path if too long, show `...` in middle

### 6.2 Prompt

**Current (Phase 2):**
```
> message here
```

**Proposed (Phase 3):**
```
› message here
     ^ cursor on newline aligns with content
```

**Implementation:**
- Use `›` (U+203A) as prompt character
- Color: `accent` (amber)
- User input: default color
- Maintain `prompt_toolkit` multiline support

### 6.3 Assistant Output

**Current (Phase 2):**
```
───────────────────────────────── assistant ─────────────────────────────────
│ [Streaming Markdown content]                                               │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
```

**Proposed (Phase 3):**
```
[Pure Markdown content streams here]

No container. No label. Just content.
```

**Implementation:**
- Remove `Rule("assistant")` header
- Remove any Panel wrapping
- Stream `Markdown` directly to console
- Maintain `Live` for smooth updates

### 6.4 Tool Execution

**Current (Phase 2):**
```
[cyan]tool start[/] Read(file_path="/path/to/file")
[green]tool done[/] Read: Lorem ipsum dolor sit amet...
```

**Proposed (Phase 3) - Option A (Minimal):**
```
⠋ Reading file...                    # executing
✓ Read completed                     # success
```

**Proposed (Phase 3) - Option B (Collapsible):**
```
╭─ Tools · 2 ──────────────────────╮  # collapsed by default, expand with `/`
│ ⠋ Read(file="/path/to/file")      │
│ ✓ Grep(pattern="class")           │
╰─ 1.2s ───────────────────────────╯
```

**Recommendation:** Start with Option A for simplicity.

**Implementation:**
```python
# Spinner states (using rich.status.Status)
STATUS_ICONS = {
    "pending": "⠋",   # Braille spinner
    "running": "⠙",   # Animated braille
    "success": "✓",   # Checkmark
    "error": "✗",     # X mark
}

# Tool name display
# - Truncate arguments beyond ~40 chars
# - Show "..." for truncated content
# - Color: `dim` for tool name, default for args
```

### 6.5 Error Messages

**Current (Phase 2):**
```
╭──────────────── Error ─────────────────╮
│ Something went wrong                   │
╰────────────────────────────────────────╯
```

**Proposed (Phase 3):**
```
✗ Error: Something went wrong
```

**Implementation:**
- Single line prefix with `✗` icon
- Color: `error` red
- Multi-line errors: indent subsequent lines to align with first

### 6.6 Notices

**Current (Phase 2):**
```
╭────────────────────────────────────────╮
│ Input cleared. Press Ctrl+C again...  │
╰────────────────────────────────────────╯
```

**Proposed (Phase 3):**
```
⚠ Input cleared. Press Ctrl+C again within 2s to exit.
```

**Implementation:**
- Single line with `⚠` prefix
- Color: `dim` (not yellow - less urgent)
- Keep it subtle, not alarming

### 6.7 Help Display

**Current (Phase 2):**
```
╭────────────── commands ────────────────╮
│ /help    show interactive commands     │
│ /status  show current session details  │
│ ...                                    │
╰────────────────────────────────────────╯
```

**Proposed (Phase 3):**
```
Commands:
  /help      Show interactive commands
  /status    Show session details
  /history   Show recent turns
  /new       Start fresh session
  /clear     Clear terminal
  /exit      Leave session

Alt+Enter for newline
```

**Implementation:**
- No Panel, no border
- Simple left-aligned list
- Command names in regular weight, descriptions in `dim`
- Consistent spacing between columns

### 6.8 Status Command

**Current (Phase 2):**
```
╭────────────── status ──────────────────╮
│ session   2025-03-14-abc123           │
│ provider  anthropic                   │
│ model     claude-sonnet-4-6           │
│ cwd       /path/to/project            │
│ messages  12                          │
╰────────────────────────────────────────╯
```

**Proposed (Phase 3):**
```
Session:  2025-03-14-abc123
Provider: anthropic
Model:    claude-sonnet-4-6
CWD:      /path/to/project
Messages: 12
Tools:    read, grep, glob, web_search, web_fetch
```

**Implementation:**
- No Panel, simple aligned text
- Labels in `dim`, values in default
- Monospace alignment for readability

### 6.9 History Display

**Current (Phase 2):**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ user           assistant               ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ hello          Hi there!               ┃
┃ what is 2+2?   2 + 2 = 4               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Proposed (Phase 3):**
```
Turn 3 · 2 minutes ago
─────────────────────
› hello

Hi there! How can I help?

Turn 4 · just now
─────────────────
› what is 2+2?

2 + 2 = 4
```

**Implementation:**
- Conversation-style formatting
- User input prefixed with `›`
- Assistant output plain
- Turn separators with relative timestamps
- Truncate long content with "... (truncated)"

## 7. Implementation Plan

### 7.1 Files to Modify

```
src/agentlet/cli/
├── presenter.py      # Major rewrite of rendering methods
└── repl.py           # Minor: update prompt symbol if hardcoded
```

### 7.2 Presenter Refactoring

**Remove:**
- `ChatPresenter._build_info_table()` method
- All `Panel` usage except for potential error details
- All `Rule` usage for section headers
- `Table` usage for history

**Add:**
- `Theme` class with color constants
- `_format_tool_status()` helper
- `_format_session_line()` helper
- Spinner management for tool execution

**Modify:**
- `show_session_header()`: single line output
- `show_help()`: plain text list
- `show_status()`: aligned text pairs
- `show_history()`: conversation format
- `show_error()`: single line with icon
- `show_notice()`: single line with icon
- `handle_event()`: tool status with spinner
- `_render_stream()`: no Rule, no Panel

### 7.3 Color Migration

Replace inline color tags with semantic constants:

```python
# Before
self.console.print(f"[cyan]tool start[/] {name}")
self.console.print(f"[green]tool done[/] {name}")

# After
self.console.print(f"{STATUS_ICONS['running']} {name}", style=COLORS["dim"])
self.console.print(f"{STATUS_ICONS['success']} {name}", style=COLORS["success"])
```

## 8. Examples

### 8.1 Complete Session Flow

```
$ agentlet chat
agentlet · claude-sonnet-4-6 · /Users/cuizhengliang/projects/demo · /help for commands

› hello

Hello! How can I help you today?

› what files are in the current directory?

⠋ Listing files...
✓ List completed

I can see the following files in /Users/cuizhengliang/projects/demo:
- README.md
- src/
- tests/
- pyproject.toml

› /help

Commands:
  /help      Show interactive commands
  /status    Show session details
  /history   Show recent turns
  /new       Start fresh session
  /clear     Clear terminal
  /exit      Leave session

Alt+Enter for newline

› /exit

Session closed.
```

### 8.2 Error Handling

```
› /unknown

✗ Unknown command: /unknown

› /status with args

✗ Command /status does not take arguments
```

### 8.3 Multi-turn History

```
› /history

Turn 1 · 10 minutes ago
──────────────────────
› hello

Hello! How can I help?

Turn 2 · 5 minutes ago
─────────────────────
› read the README

⠋ Reading file...
✓ Read completed

This project is a CLI harness for AI agents...

Turn 3 · just now
─────────────────
› thanks

You're welcome!
```

## 9. Exit Criteria

Phase 3 is complete when all of the following are true:

- [ ] No Panel borders in normal operation (except possibly for detailed errors)
- [ ] Single-line session header with dot separators
- [ ] `›` prompt indicator with amber accent color
- [ ] Assistant output streams without Rule/Panel containers
- [ ] Tool execution shows as spinner/checkmark, not text labels
- [ ] Help displays as plain text list, not Panel table
- [ ] History displays in conversation format
- [ ] All colors use the refined palette, no raw "blue/cyan/green"
- [ ] Visual density reduced by ~30% (fewer lines, less chrome)
- [ ] All Phase 2 functionality preserved

## 10. References

- [Claude Code Design Philosophy](https://code.claude.com/docs)
- [Rich Console API](https://rich.readthedocs.io/en/stable/console.html)
- [Rich Status Spinner](https://rich.readthedocs.io/en/stable/reference/status.html)
- [Rich Markdown](https://rich.readthedocs.io/en/stable/markdown.html)
- Phase 2 Design Doc: [phase-2-cli-experience.md](./phase-2-cli-experience.md)
