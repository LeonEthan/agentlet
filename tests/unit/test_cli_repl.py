from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentlet.cli.repl as repl_module
from agentlet.agent.agent_loop import AgentTurnResult, TurnEvent
from agentlet.agent.context import Context
from agentlet.cli.commands import TurnSummary
from agentlet.cli.repl import run_repl
from agentlet.cli.sessions import LoadedSession, SessionInfo


class ScriptedPromptSession:
    def __init__(self, inputs: list[object]) -> None:
        self._inputs = list(inputs)

    def prompt(self, prompt_text: str | None = None) -> str:
        raise AssertionError("interactive REPL should use prompt_async()")

    async def prompt_async(self, prompt_text: str | None = None) -> str:
        if not self._inputs:
            raise EOFError
        item = self._inputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return str(item)


class RecordingPresenter:
    def __init__(self) -> None:
        self.session_headers: list[dict[str, object]] = []
        self.help_calls: list[list[str]] = []
        self.status_calls: list[dict[str, object]] = []
        self.history_calls: list[list[TurnSummary]] = []
        self.notices: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.clear_calls = 0
        self.stop_stream_calls = 0
        self.events: list[TurnEvent] = []

    def show_session_header(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
    ) -> None:
        self.session_headers.append(
            {
                "session_id": session_id,
                "provider_name": provider_name,
                "model": model,
                "cwd": cwd,
            }
        )

    def show_help(self, lines: list[str]) -> None:
        self.help_calls.append(lines)

    def show_status(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
        message_count: int,
        tool_names: list[str],
    ) -> None:
        self.status_calls.append(
            {
                "session_id": session_id,
                "provider_name": provider_name,
                "model": model,
                "cwd": cwd,
                "message_count": message_count,
                "tool_names": tool_names,
            }
        )

    def show_history(self, turns: list[TurnSummary], *, limit: int = 20) -> None:
        self.history_calls.append(turns)

    def show_notice(self, message: str) -> None:
        self.notices.append(message)

    def show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    def clear(self) -> None:
        self.clear_calls += 1

    def handle_event(self, event: TurnEvent) -> None:
        self.events.append(event)

    def stop_stream(self) -> None:
        self.stop_stream_calls += 1


class FakeToolRegistry:
    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._tool_names = tool_names or []

    def get_tool_names(self) -> list[str]:
        return list(self._tool_names)


class ScriptedLoop:
    def __init__(self, outcomes: list[object], tool_names: list[str] | None = None) -> None:
        self._outcomes = list(outcomes)
        self.tool_registry = FakeToolRegistry(tool_names or ["echo"])
        self.seen_messages: list[dict[str, object]] = []
        self.system_prompt = "system"

    async def run_turn(
        self,
        user_input: str,
        *,
        context: Context | None = None,
        event_sink=None,
        stream: bool = False,
    ) -> AgentTurnResult:
        self.seen_messages.append(
            {
                "user_input": user_input,
                "stream": stream,
                "context": context,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

        assert context is not None
        context.build_messages(user_input)
        context.add_assistant_message(str(outcome))
        result = AgentTurnResult(
            output=str(outcome),
            context=context,
            iterations=1,
            finish_reason="stop",
        )
        if event_sink is not None:
            event_sink(TurnEvent(kind="turn_started", user_input=user_input))
            event_sink(TurnEvent(kind="assistant_completed", content=str(outcome)))
            event_sink(TurnEvent(kind="turn_completed", result=result))
        return result


class RecordingSessionStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self.started_infos: list[SessionInfo] = []
        self.appended_records: list[dict[str, object]] = []

    def start_session(
        self,
        *,
        provider_name: str,
        model: str,
        api_base: str | None,
        temperature: float,
        max_tokens: int | None,
        system_prompt: str,
    ) -> SessionInfo:
        session_id = f"session-{len(self.started_infos) + 1}"
        info = SessionInfo(
            session_id=session_id,
            transcript_path=self._base_dir / f"{session_id}.jsonl",
            cwd=str(self._base_dir),
            provider_name=provider_name,
            model=model,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        self.started_infos.append(info)
        return info

    def append_records(
        self,
        session: SessionInfo,
        records: list[dict[str, object]],
        *,
        update_latest: bool,
    ) -> None:
        self.appended_records.append(
            {
                "session_id": session.session_id,
                "records": records,
                "update_latest": update_latest,
            }
        )


def _make_loaded_session(store: RecordingSessionStore) -> LoadedSession:
    info = store.start_session(
        provider_name="openai",
        model="gpt-5.4",
        api_base=None,
        temperature=0.0,
        max_tokens=None,
        system_prompt="system",
    )
    return LoadedSession(
        info=info,
        context=Context(system_prompt="system"),
    )


def test_run_repl_commands_cover_help_history_clear_and_new(tmp_path: Path) -> None:
    store = RecordingSessionStore(tmp_path)
    loaded_session = _make_loaded_session(store)
    loop = ScriptedLoop(["first answer"])
    presenter = RecordingPresenter()
    prompt = ScriptedPromptSession(
        ["hello", "/history", "/help", "/clear", "/new", "/history", "/exit"]
    )

    exit_code = asyncio.run(
        run_repl(
            loop=loop,
            prompt_input=prompt,
            presenter=presenter,
            session_store=store,
            cwd=tmp_path,
            loaded_session=loaded_session,
        )
    )

    assert exit_code == 0
    assert presenter.history_calls == [
        [TurnSummary(1, "hello", "first answer")],
        [],
    ]
    assert len(presenter.help_calls) == 1
    assert presenter.clear_calls == 1
    assert presenter.notices == ["Session closed."]
    assert [entry["session_id"] for entry in presenter.session_headers] == [
        "session-1",
        "session-2",
    ]
    assert len(store.started_infos) == 2
    assert len(store.appended_records) == 1
    assert store.appended_records[0]["session_id"] == "session-1"
    assert [item["user_input"] for item in loop.seen_messages] == ["hello"]


def test_run_repl_single_keyboard_interrupt_clears_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RecordingSessionStore(tmp_path)
    loaded_session = _make_loaded_session(store)
    loop = ScriptedLoop([])
    presenter = RecordingPresenter()
    prompt = ScriptedPromptSession([KeyboardInterrupt(), "/exit"])
    monotonic_values = iter([10.0])

    def fake_monotonic() -> float:
        return next(monotonic_values, 10.0)

    monkeypatch.setattr(repl_module, "time", SimpleNamespace(monotonic=fake_monotonic))

    exit_code = asyncio.run(
        run_repl(
            loop=loop,
            prompt_input=prompt,
            presenter=presenter,
            session_store=store,
            cwd=tmp_path,
            loaded_session=loaded_session,
        )
    )

    assert exit_code == 0
    assert presenter.notices == [
        "Input cleared. Press Ctrl+C again within 2s to exit.",
        "Session closed.",
    ]
    assert loop.seen_messages == []
    assert store.appended_records == []


def test_run_repl_double_keyboard_interrupt_exits_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RecordingSessionStore(tmp_path)
    loaded_session = _make_loaded_session(store)
    loop = ScriptedLoop([])
    presenter = RecordingPresenter()
    prompt = ScriptedPromptSession([KeyboardInterrupt(), KeyboardInterrupt()])
    monotonic_values = iter([10.0, 11.0])

    def fake_monotonic() -> float:
        return next(monotonic_values, 11.0)

    monkeypatch.setattr(repl_module, "time", SimpleNamespace(monotonic=fake_monotonic))

    exit_code = asyncio.run(
        run_repl(
            loop=loop,
            prompt_input=prompt,
            presenter=presenter,
            session_store=store,
            cwd=tmp_path,
            loaded_session=loaded_session,
        )
    )

    assert exit_code == 0
    assert presenter.notices == [
        "Input cleared. Press Ctrl+C again within 2s to exit.",
        "Exiting interactive session.",
    ]
    assert loop.seen_messages == []
    assert store.appended_records == []


def test_run_repl_cancelled_turn_does_not_persist_records(tmp_path: Path) -> None:
    store = RecordingSessionStore(tmp_path)
    loaded_session = _make_loaded_session(store)
    loop = ScriptedLoop([KeyboardInterrupt()])
    presenter = RecordingPresenter()
    prompt = ScriptedPromptSession(["hello", "/exit"])

    exit_code = asyncio.run(
        run_repl(
            loop=loop,
            prompt_input=prompt,
            presenter=presenter,
            session_store=store,
            cwd=tmp_path,
            loaded_session=loaded_session,
        )
    )

    assert exit_code == 0
    assert presenter.notices == ["Turn cancelled.", "Session closed."]
    assert presenter.stop_stream_calls == 1
    assert store.appended_records == []


def test_run_repl_failed_turn_can_recover_on_next_input(tmp_path: Path) -> None:
    store = RecordingSessionStore(tmp_path)
    loaded_session = _make_loaded_session(store)
    loop = ScriptedLoop([RuntimeError("boom"), "fixed answer"])
    presenter = RecordingPresenter()
    prompt = ScriptedPromptSession(["hello", "retry", "/history", "/exit"])

    exit_code = asyncio.run(
        run_repl(
            loop=loop,
            prompt_input=prompt,
            presenter=presenter,
            session_store=store,
            cwd=tmp_path,
            loaded_session=loaded_session,
        )
    )

    assert exit_code == 0
    assert presenter.errors == [("Turn failed", "boom")]
    assert presenter.stop_stream_calls == 1
    assert presenter.history_calls == [
        [TurnSummary(1, "retry", "fixed answer")],
    ]
    assert len(store.appended_records) == 1
    assert store.appended_records[0]["session_id"] == "session-1"
    assert [item["user_input"] for item in loop.seen_messages] == ["hello", "retry"]
