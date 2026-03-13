# agentlet

A minimal Python agent harness.

## Status

Current phase:

- single-process agent loop with tool support
- interactive TTY chat with streaming output
- resumable cwd-scoped sessions under `~/.agentlet/sessions/` (grouped by working directory hash)
- independent `Context`
- `LiteLLM` provider integration
- user-level `~/.agentlet/setting.json` defaults for local testing

## Setup

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

## Real API Testing

Initialize the user-level settings file once:

```bash
agentlet init \
  --api-key your_api_key \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o-mini
```

This writes `~/.agentlet/settings.json` with the canonical JSON shape:

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "your_api_key",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.0,
  "max_tokens": null
}
```

Exported shell variables still override the file when you need a temporary change.

Run a real request:

```bash
uv run python -m agentlet.cli.main chat "Hello, introduce yourself briefly."
```

Run the live API test suite explicitly:

```bash
AGENTLET_RUN_REAL_API_TESTS=1 uv run pytest tests/test_real_api.py -v
```

## DeepSeek Example

DeepSeek works through the OpenAI-compatible path, but the model name must follow LiteLLM's provider-prefixed format.

```bash
agentlet init \
  --api-key your_deepseek_api_key \
  --api-base https://api.deepseek.com/v1 \
  --model deepseek/deepseek-chat
```

Run:

```bash
uv run python -m agentlet.cli.main chat "Hello"
```

## CLI Usage

One-shot:

```bash
agentlet chat "hello"
echo "hello" | agentlet chat
agentlet chat --print < prompt.txt
```

Interactive:

```bash
agentlet init
agentlet chat
agentlet chat --continue
agentlet chat --session 20260312T120000000000Z-deadbeef
agentlet chat --new-session
```

Interactive commands:

- `/help`
- `/status`
- `/history`
- `/new`
- `/clear`
- `/exit`

## Notes

- SOCKS proxy support is included through `httpx[socks]`.
- `LiteLLM` may require provider-prefixed model names for some backends.
- Exported shell variables win over values stored in `~/.agentlet/settings.json`.
- Interactive sessions persist only completed turns; cancelled or failed turns are not committed.
- Session headers persist non-sensitive provider settings so resumed chats keep the same model endpoint and sampling limits.
- Detailed design notes live in [`docs/design-docs/phase-1-foundation.md`](docs/design-docs/phase-1-foundation.md) and [`docs/design-docs/phase-2-cli-experience.md`](docs/design-docs/phase-2-cli-experience.md).
