from agentlet.agent.context import Context, ToolCall, ToolResult


def test_context_builds_messages_and_tracks_history() -> None:
    context = Context(system_prompt="system prompt")

    first_snapshot = context.build_messages("hello")

    assert [message.role for message in first_snapshot] == ["system", "user"]
    assert first_snapshot[1].content == "hello"

    context.add_assistant_message(
        "working",
        [ToolCall(id="call-1", name="echo", arguments_json='{"text":"hi"}')],
    )
    context.add_tool_result(
        ToolResult(tool_call_id="call-1", name="echo", content="hi")
    )

    second_snapshot = context.build_messages()

    assert [message.role for message in second_snapshot] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_snapshot[2].tool_calls[0].name == "echo"
    assert second_snapshot[3].tool_call_id == "call-1"
