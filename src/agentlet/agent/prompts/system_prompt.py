"""System prompt helpers.

The prompt stays in a dedicated module so the orchestration code can depend on
one small function instead of hard-coding prompt text inline.
"""

def build_system_prompt() -> str:
    """Return the default system prompt used by the phase-1 agent loop."""
    return (
        "You are agentlet, a concise and reliable Python agent harness. "
        "Prefer direct answers, use tools only when necessary, and keep state transitions explicit."
    )
