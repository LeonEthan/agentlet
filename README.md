# agentlet

A minimal Python agent harness.

## Status

Current phase:

- single-turn agent loop
- independent `Context`
- `LiteLLM` provider integration
- OpenAI-compatible local testing path
- project-level `.env` loading for real API testing

## Setup

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
python3 -m pytest tests
```

## Real API Testing

The CLI automatically loads `.env` from the current directory or parent directories.
Existing shell environment variables are not overwritten.

Example `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
AGENTLET_MODEL=gpt-4o-mini
```

Run a real request:

```bash
uv run python -m agentlet.cli.main chat "Hello, introduce yourself briefly."
```

## DeepSeek Example

DeepSeek works through the OpenAI-compatible path, but the model name must follow LiteLLM's provider-prefixed format.

Example `.env`:

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
AGENTLET_MODEL=deepseek/deepseek-chat
```

Run:

```bash
uv run python -m agentlet.cli.main chat "Hello"
```

## Notes

- SOCKS proxy support is included through `httpx[socks]`.
- `LiteLLM` may require provider-prefixed model names for some backends.
- Detailed design notes live in [`docs/design-docs/phase-1-foundation.md`](docs/design-docs/phase-1-foundation.md).
