"""Runtime app assembly and approval-aware execution flow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from agentlet.core.approvals import ApprovalPolicy
from agentlet.core.context import ContextBuilder, CurrentTaskState
from agentlet.core.loop import (
    AgentLoop,
    ApprovalRequiredTurn,
    CompletedTurn,
    InterruptedTurn,
)
from agentlet.core.types import JSONObject, deep_copy_json_object
from agentlet.llm.base import ModelClient
from agentlet.llm.openai_like import OpenAILikeModelClient, build_openai_like_transport
from agentlet.llm.schemas import ToolChoice
from agentlet.memory import MemoryStore, SessionStore
from agentlet.runtime.events import (
    ResumeRequest,
    RuntimeEvent,
    UserQuestionRequest,
)
from agentlet.runtime.user_io import UserIO
from agentlet.tools.exec.bash import BashTool
from agentlet.tools.fs.edit import EditTool
from agentlet.tools.fs.glob import GlobTool
from agentlet.tools.fs.grep import GrepTool
from agentlet.tools.fs.read import ReadTool
from agentlet.tools.fs.write import WriteTool
from agentlet.tools.registry import ToolRegistry


RuntimeRunOutcome = CompletedTurn | InterruptedTurn
DEFAULT_OPENAI_LIKE_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Resolved file layout used by the runtime app."""

    workspace_root: Path
    session_path: Path
    memory_path: Path
    instructions_path: Path | None = None

    def __post_init__(self) -> None:
        workspace_root = self.workspace_root.resolve()
        if not workspace_root.exists():
            raise ValueError(f"workspace root does not exist: {workspace_root}")
        if not workspace_root.is_dir():
            raise ValueError(f"workspace root is not a directory: {workspace_root}")

        object.__setattr__(self, "workspace_root", workspace_root)
        object.__setattr__(self, "session_path", self.session_path.resolve())
        object.__setattr__(self, "memory_path", self.memory_path.resolve())
        if self.instructions_path is not None:
            object.__setattr__(
                self,
                "instructions_path",
                self.instructions_path.resolve(),
            )

    @classmethod
    def for_workspace(
        cls,
        workspace_root: str | Path,
        *,
        state_dir: str | Path = ".agentlet",
        session_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        instructions_path: str | Path | None = None,
    ) -> "RuntimePaths":
        workspace = Path(workspace_root).resolve()
        state_root = _resolve_path(workspace, state_dir)
        resolved_session_path = _resolve_path(
            workspace,
            session_path or state_root / "session.jsonl",
        )
        resolved_memory_path = _resolve_path(
            workspace,
            memory_path or state_root / "memory.md",
        )
        resolved_instructions_path = (
            _resolve_path(workspace, instructions_path)
            if instructions_path is not None
            else workspace / "AGENTS.md"
        )
        return cls(
            workspace_root=workspace,
            session_path=resolved_session_path,
            memory_path=resolved_memory_path,
            instructions_path=resolved_instructions_path,
        )


@dataclass(frozen=True, slots=True)
class OpenAILikeConfig:
    """Configuration used to assemble the default OpenAI-like model client."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    request_defaults: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        object.__setattr__(
            self,
            "request_defaults",
            deep_copy_json_object(self.request_defaults),
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        request_defaults: JSONObject | None = None,
        timeout_seconds: float = 60.0,
    ) -> "OpenAILikeConfig":
        """Load OpenAI-like provider settings from the process environment."""

        env = os.environ if environ is None else environ
        model = env.get("AGENTLET_MODEL", "").strip()
        if not model:
            raise ValueError("AGENTLET_MODEL must be set.")
        api_key = env.get("AGENTLET_API_KEY", "").strip()
        if not api_key:
            raise ValueError("AGENTLET_API_KEY must be set.")
        return cls(
            model=model,
            api_key=api_key,
            base_url=env.get("AGENTLET_BASE_URL", DEFAULT_OPENAI_LIKE_BASE_URL),
            timeout_seconds=timeout_seconds,
            request_defaults=request_defaults or {},
        )


@dataclass(slots=True)
class RuntimeApp:
    """Runtime-facing app wrapper around the core agent loop."""

    loop: AgentLoop
    user_io: UserIO
    system_instructions: str | None = None

    def run_turn(
        self,
        *,
        current_task: CurrentTaskState | str | None = None,
        resume: ResumeRequest | None = None,
    ) -> RuntimeRunOutcome:
        """Run one task while resolving approvals through ``UserIO``."""

        if current_task is None and resume is None:
            raise ValueError("current_task or resume is required")

        next_task = current_task
        next_resume = resume
        if next_resume is not None:
            self.user_io.emit_event(RuntimeEvent.resumed(next_resume))

        while True:
            outcome = self.loop.run(
                current_task=next_task,
                system_instructions=self.system_instructions,
                resume=next_resume,
            )
            next_task = None
            next_resume = None

            if isinstance(outcome, ApprovalRequiredTurn):
                self.user_io.emit_event(
                    RuntimeEvent.approval_requested(outcome.request)
                )
                approval_response = self.user_io.request_approval(outcome.request)
                next_resume = ResumeRequest.from_approval_response(approval_response)
                self.user_io.emit_event(RuntimeEvent.resumed(next_resume))
                continue

            if isinstance(outcome, InterruptedTurn):
                if outcome.interrupt.kind == "question":
                    request = UserQuestionRequest.from_interrupt(outcome.interrupt)
                    self.user_io.emit_event(
                        RuntimeEvent.question_interrupted(request)
                    )
                    self.user_io.begin_question_interrupt(request)
                    question_response = self.user_io.resolve_question_interrupt(
                        request
                    )
                    next_resume = ResumeRequest.from_question_response(
                        question_response
                    )
                    self.user_io.emit_event(RuntimeEvent.resumed(next_resume))
                    continue
                return outcome

            return outcome


def build_runtime_app(
    *,
    model: ModelClient,
    user_io: UserIO,
    workspace_root: str | Path,
    state_dir: str | Path = ".agentlet",
    session_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    instructions_path: str | Path | None = None,
    registry: ToolRegistry | None = None,
    approval_policy: ApprovalPolicy | None = None,
    context_builder: ContextBuilder | None = None,
    tool_choice: ToolChoice | None = None,
    max_iterations: int = 8,
    bash_timeout_seconds: float | None = None,
) -> RuntimeApp:
    """Assemble the runtime app around one configured model client."""

    paths = RuntimePaths.for_workspace(
        workspace_root,
        state_dir=state_dir,
        session_path=session_path,
        memory_path=memory_path,
        instructions_path=instructions_path,
    )
    return RuntimeApp(
        loop=AgentLoop(
            model=model,
            registry=(
                registry
                if registry is not None
                else build_default_registry(
                    paths.workspace_root,
                    bash_timeout_seconds=bash_timeout_seconds,
                )
            ),
            session_store=SessionStore(paths.session_path),
            memory_store=MemoryStore(paths.memory_path),
            context_builder=context_builder or ContextBuilder(),
            approval_policy=approval_policy or ApprovalPolicy(),
            tool_choice=tool_choice,
            max_iterations=max_iterations,
        ),
        user_io=user_io,
        system_instructions=_load_system_instructions(paths.instructions_path),
    )


def build_openai_like_runtime_app(
    *,
    user_io: UserIO,
    workspace_root: str | Path,
    config: OpenAILikeConfig,
    state_dir: str | Path = ".agentlet",
    session_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    instructions_path: str | Path | None = None,
    approval_policy: ApprovalPolicy | None = None,
    context_builder: ContextBuilder | None = None,
    tool_choice: ToolChoice | None = None,
    max_iterations: int = 8,
    bash_timeout_seconds: float | None = None,
) -> RuntimeApp:
    """Assemble a runtime app using the built-in OpenAI-like provider client."""

    client = OpenAILikeModelClient(
        model=config.model,
        transport=build_openai_like_transport(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        ),
        request_defaults=config.request_defaults,
    )
    return build_runtime_app(
        model=client,
        user_io=user_io,
        workspace_root=workspace_root,
        state_dir=state_dir,
        session_path=session_path,
        memory_path=memory_path,
        instructions_path=instructions_path,
        approval_policy=approval_policy,
        context_builder=context_builder,
        tool_choice=tool_choice,
        max_iterations=max_iterations,
        bash_timeout_seconds=bash_timeout_seconds,
    )


def build_default_runtime_app(
    *,
    user_io: UserIO,
    workspace_root: str | Path,
    state_dir: str | Path = ".agentlet",
    session_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    instructions_path: str | Path | None = None,
    approval_policy: ApprovalPolicy | None = None,
    context_builder: ContextBuilder | None = None,
    tool_choice: ToolChoice | None = None,
    max_iterations: int = 8,
    environ: Mapping[str, str] | None = None,
) -> RuntimeApp:
    """Assemble the default terminal runtime from environment-backed settings."""

    return build_openai_like_runtime_app(
        user_io=user_io,
        workspace_root=workspace_root,
        config=OpenAILikeConfig.from_env(environ),
        state_dir=state_dir,
        session_path=session_path,
        memory_path=memory_path,
        instructions_path=instructions_path,
        approval_policy=approval_policy,
        context_builder=context_builder,
        tool_choice=tool_choice,
        max_iterations=max_iterations,
    )


def build_default_registry(
    workspace_root: str | Path,
    *,
    bash_timeout_seconds: float | None = None,
) -> ToolRegistry:
    """Build the default local-coding tool registry for the terminal app."""

    workspace = Path(workspace_root).resolve()
    return ToolRegistry(
        [
            ReadTool(workspace),
            WriteTool(workspace),
            EditTool(workspace),
            BashTool(workspace, default_timeout_seconds=bash_timeout_seconds),
            GlobTool(workspace),
            GrepTool(workspace),
        ]
    )


def _load_system_instructions(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"instructions path is not a file: {path}")
    content = path.read_text(encoding="utf-8")
    return content if content.strip() else None


def _resolve_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


__all__ = [
    "OpenAILikeConfig",
    "DEFAULT_OPENAI_LIKE_BASE_URL",
    "RuntimeApp",
    "RuntimePaths",
    "RuntimeRunOutcome",
    "build_default_registry",
    "build_default_runtime_app",
    "build_openai_like_runtime_app",
    "build_runtime_app",
]
