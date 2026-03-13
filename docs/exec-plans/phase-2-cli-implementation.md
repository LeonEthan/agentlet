# Phase 2 CLI Implementation

Status: in progress

## Summary

This execution plan implements the phase-2 CLI experience on top of the
phase-1 runtime:

- keep one-shot chat behavior for argv and stdin usage
- add an interactive REPL for TTY sessions
- persist interactive transcripts under `~/.agentlet/sessions/` (grouped by cwd hash)
- add normalized streaming and turn events at the runtime boundary
- keep `agent/` free of `rich` and `prompt_toolkit` imports

## Implementation Slices

1. Test hygiene and dependencies
   - add `rich` and `prompt_toolkit`
   - gate real API tests behind an explicit marker and env flag
2. Runtime contracts
   - add provider streaming events
   - add loop turn events
   - preserve one final response handling path
3. CLI/session architecture
   - split one-shot vs interactive mode selection
   - add session persistence, replay, and slash commands
4. Validation
   - expand fake-based unit and smoke coverage
   - update README with the new CLI shape

## Notes

- Session persistence is interactive-only in this phase.
- Resume rehydrates `Context` from normalized transcript records.
- One-shot mode remains final-output-only for scripting stability.
