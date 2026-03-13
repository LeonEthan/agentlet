# Phase 1 Foundation Design

Status: draft

## 1. Context

This phase defines the smallest useful `agentlet` harness:

- a basic agent loop
- an independent `Context` class
- a provider boundary backed by `LiteLLM`
- an OpenAI-compatible path as the first concrete provider for local testing

This design is influenced by two references:

- OpenAI Harness Engineering: keep `AGENTS.md` as the map, push detailed decisions into `docs/`
- `HKUDS/nanobot`: keep the core package shape small, separate agent loop from context building, and isolate provider integration behind a narrow interface

Reference links:

- [OpenAI Harness Engineering](https://openai.com/zh-Hans-CN/index/harness-engineering/)
- [HKUDS/nanobot](https://github.com/HKUDS/nanobot/tree/main)

## 2. Goals

Phase 1 must deliver:

1. A single-process agent loop that can:
   - accept user input
   - build a model-ready message list
   - call an LLM provider
   - optionally execute tool calls
   - return the final assistant response
2. A standalone `Context` abstraction responsible only for message assembly and turn mutation.
3. A provider interface with one concrete adapter: `LiteLLMProvider`.
4. A CLI entrypoint suitable for local manual testing.
5. A testable architecture where the loop, context, and provider can be unit-tested independently.

## 3. Non-Goals

Phase 1 explicitly does not include:

- multi-channel gateways
- background workers or message bus
- persistent sessions or long-term memory
- subagents
- MCP integration
- provider registry for many vendors
- advanced retries, scheduling, observability, or auth workflows

The goal is a clean nucleus, not a feature-complete framework.

## 4. Design Principles

We will copy the shape, not the weight, of `nanobot`.

- `AgentLoop` owns iteration.
- `Context` owns prompt and message construction.
- provider adapters own SDK and API details.
- tools are invoked through a narrow registry boundary.
- internal types and protocols may live close to the loop instead of being split into extra layers too early.
- phase-1 code should optimize for readability over extensibility theater.

## 5. Proposed Repository Shape

Phase 1 should build toward this minimal subset:

```text
src/agentlet/
├── agent/
│   ├── agent_loop.py
│   ├── context.py
│   ├── tools/
│   │   └── registry.py
│   ├── providers/
│   │   ├── registry.py
│   │   └── litellm_provider.py
│   └── prompts/
│       └── system_prompt.py
└── cli/
    └── main.py
```

Notes:

- `agent/agent_loop.py` should be readable in one sitting.
- `agent/context.py` should stay independent from provider and tool execution details.
- `agent/providers/litellm_provider.py` is the only place that knows `litellm`.
- `agent/providers/registry.py` can stay very small in phase 1 and may only resolve one default provider.
- colocating these modules under `agent/` is preferred over introducing extra architectural layers before they pay for themselves.

## 6. Core Abstractions

### 6.1 Message Model

We only need an OpenAI-like message format in phase 1.

Recommended internal shapes:

- `Role`: `system | user | assistant | tool`
- `Message`:
  - `role`
  - `content`
  - `name: str | None`
  - `tool_call_id: str | None`
- `ToolCall`:
  - `id`
  - `name`
  - `arguments_json`
- `ToolResult`:
  - `tool_call_id`
  - `name`
  - `content`

These may be implemented as dataclasses first, likely colocated in `agent/context.py` or nearby small modules if they start to grow. We should avoid introducing `pydantic` until external validation pressure becomes real.

### 6.2 Context

`Context` is intentionally independent from `AgentLoop`.

Responsibilities:

- hold the system prompt
- hold current turn history
- build provider-ready message payloads
- append assistant messages
- append tool results

Non-responsibilities:

- call the model
- execute tools
- persist history
- decide loop termination

Minimal interface:

```python
class Context:
    def __init__(self, system_prompt: str, history: list[Message] | None = None): ...
    def build_messages(self, user_input: str) -> list[Message]: ...
    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None: ...
    def add_tool_result(self, result: ToolResult) -> None: ...
```

Design intent:

- `Context` should answer one question only: "What messages should the provider receive next?"
- it should be unit-testable without any LLM mocking
- if we later add persistence, `Context` should still remain a pure in-memory turn builder

### 6.3 LLM Provider Port

The loop should not know `litellm` details.

Provider contract:

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
```

`LLMResponse` should normalize backend output:

- `content: str | None`
- `tool_calls: list[ToolCall]`
- `finish_reason: str | None`
- `usage: TokenUsage | None`

We normalize once at the provider edge so the rest of the system never sees raw SDK payloads.

In phase 1 this protocol can live inside `agent/providers/litellm_provider.py` or `agent/providers/registry.py`. There is no need to split it into an extra layer yet.

### 6.4 Tool Boundary

We keep tools simple in phase 1.

Tool contract:

```python
class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult: ...
    def get_tool_schemas(self) -> list[ToolSpec]: ...
```

Initial recommendation:

- start with an in-memory registry
- phase 1 can ship with zero or one demo tool
- the loop must support tool calls even if the initial CLI only tests plain chat
- keep the registry under `agent/tools/registry.py`; avoid a generic plugin system for now

## 7. Agent Loop

`AgentLoop` is the orchestration center. It should own only the runtime sequence.

Minimal loop:

1. Receive a user input string.
2. Ask `Context` to build the initial message list.
3. Call `LLMProvider.complete(...)`.
4. If the provider returns plain assistant content, store it in `Context` and return it.
5. If the provider returns tool calls:
   - add the assistant tool-call message to `Context`
   - execute each tool through `ToolExecutor`
   - add each tool result back into `Context`
   - rebuild the next provider request from `Context`
   - continue until final assistant content or max iterations
6. Stop after a small fixed iteration budget, for example `max_iterations = 8`.

Reference pseudocode:

```python
async def run_turn(user_input: str) -> AgentTurnResult:
    messages = context.build_messages(user_input)

    for _ in range(max_iterations):
        response = await provider.complete(messages, tools=tool_executor.get_tool_schemas())

        context.add_assistant_message(response.content, response.tool_calls)

        if not response.tool_calls:
            return AgentTurnResult.final(response.content or "")

        for call in response.tool_calls:
            result = await tool_executor.execute(call)
            context.add_tool_result(result)

        messages = context.build_messages("")

    raise MaxIterationsExceeded()
```

Implementation notes:

- `Context.build_messages("")` after tool execution is acceptable in phase 1 if the method is documented clearly.
- if this feels awkward in code, split into `start_turn(user_input)` and `snapshot_messages()`.
- the loop must fail loudly on malformed tool arguments rather than silently guessing.

## 8. Provider Strategy

Phase 1 uses `LiteLLM` as the integration layer, but only for one narrow case first:

- OpenAI API
- OpenAI-compatible base URL
- one default chat-completions style model path

Concrete adapter:

- `LiteLLMProvider`

Responsibilities:

- translate internal `Message` objects into the shape `litellm` expects
- pass model name, API key, optional base URL, and tool schemas
- normalize `litellm` responses into `LLMResponse`

Configuration surface for phase 1:

- `model`
- `api_key`
- `api_base` optional
- `temperature`
- `max_tokens`

Deliberate constraints:

- no provider auto-detection in phase 1
- no large provider registry yet
- no vendor-specific branching outside the adapter

Operational note:

- some OpenAI-compatible backends still require provider-prefixed model names when routed through `LiteLLM`
- example: DeepSeek should be configured as `deepseek/deepseek-chat`, not just `deepseek-chat`

Provider registry role in phase 1:

- convert a short provider name such as `openai` into a provider instance
- hold the default provider selection logic
- stay tiny enough that removing it later would be cheap

Why this path:

- it keeps local testing easy
- it leaves room for true multi-provider support later
- it avoids hard-coupling the service layer to OpenAI SDK types

## 9. CLI Shape

Phase 1 should expose one obvious runtime command plus a bootstrap command, for example:

```bash
uv run python -m agentlet.cli.main init --model gpt-5.4
```

and:

```bash
uv run python -m agentlet.cli.main chat --model gpt-5.4
```

Behavior:

- `init` creates `~/.agentlet/setting.json` with the canonical agentlet config shape
- read a user message from argv or stdin
- load `~/.agentlet/setting.json` before parsing chat defaults
- construct `Context`, provider, and tool registry
- run one turn
- print the final assistant response

This is enough to validate the loop before adding a REPL or session persistence.

Environment behavior:

- `setting.json` values are treated as defaults for provider name, model, API key, base URL, temperature, and max tokens
- local CLI behavior should come from `setting.json` plus built-in defaults
- this keeps runtime configuration explicit and avoids hidden override layers

Example local test:

```bash
uv run python -m agentlet.cli.main chat "Hello"
```

## 10. Testing Strategy

The test plan should mirror the boundaries:

- `tests/unit/test_context.py`
  - builds correct message order
  - appends assistant and tool messages correctly
- `tests/unit/test_agent_loop.py`
  - stops on plain assistant response
  - executes tool calls and continues
  - enforces max-iteration limit
- `tests/unit/test_litellm_provider.py`
  - normalizes OpenAI-like responses correctly
  - maps tool calls correctly
- `tests/smoke/test_cli_chat.py`
  - one end-to-end path with a fake provider

Phase 1 should prefer fake providers in tests over live network calls.

For manual validation outside automated tests:

- use `~/.agentlet/setting.json` for credentials and endpoint configuration
- verify at least one real text-only request against an OpenAI-compatible backend
- keep automated test coverage fake-based to avoid network flakiness

## 11. Implementation Sequence

Recommended order:

1. define the minimal message, tool-call, and response types near the agent package
2. implement `Context`
3. implement `AgentLoop` with fake provider tests
4. implement `LiteLLMProvider` and the small provider registry
5. add the CLI command
6. add one smoke test

This order keeps the core loop stable before bringing in `litellm`.

## 12. Open Questions

These do not block phase 1, but we should settle them before coding:

- Should `Context` own the full history list, or should `AgentLoop` own history and let `Context` only render snapshots?
- Do we want sync plus async APIs, or async-only from day one?
- Should tool schemas use raw OpenAI JSON Schema shape, or a small internal `ToolSpec` converted at the provider edge?

Current recommendation:

- `Context` owns in-memory history
- async-only APIs
- internal `ToolSpec`, converted at the provider edge

## 13. Exit Criteria

Phase 1 is complete when all of the following are true:

- one CLI command can run a single turn against an OpenAI-compatible endpoint
- one CLI command can read user-level `~/.agentlet/setting.json` defaults for local real-provider testing
- `AgentLoop`, `Context`, and provider adapter are separate modules
- tool-call round trips work through the loop
- the core modules have unit tests
- no code outside `agent/providers/litellm_provider.py` imports `litellm` directly
