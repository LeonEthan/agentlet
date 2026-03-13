# agentlet

A minimal Python agent harness.

## Status

Current phase:

- single-process agent loop with tool support
- interactive TTY chat with streaming output
- persisted cwd-scoped session transcripts under `~/.agentlet/sessions/` (grouped by working directory hash)
- independent `Context`
- `LiteLLM` provider integration
- user-level `~/.agentlet/settings.json` defaults for local testing

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Settings

Run `agentlet init` to create your settings file:

```bash
# OpenAI (default)
agentlet init \
  --model gpt-5.4
```

Then edit `~/.agentlet/settings.json` to add `api_key` and `api_base` when your provider requires them.

See [Configuration Guide](#configuration) below for more providers.

### 3. Start Chatting

```bash
# Interactive mode (TTY)
agentlet chat

# One-shot mode
agentlet chat "Hello, introduce yourself briefly."

# Pipe input
echo "Explain quantum computing" | agentlet chat
```

## Configuration

### Settings File

The `agentlet init` command creates `~/.agentlet/settings.json`:

```json
{
  "provider": "openai",
  "model": "gpt-5.4",
  "api_key": "your_api_key",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.0,
  "max_tokens": null
}
```

### Configuration Priority

Settings are resolved in this priority order (highest first):

1. **Settings file** (`~/.agentlet/settings.json`)
2. **Built-in defaults**

| Setting | Default |
|---------|---------|
| provider | `openai` |
| model | `gpt-5.4` |
| api_key | - |
| api_base | - |
| temperature | `0.0` |
| max_tokens | `null` |

### Provider Examples

#### Anthropic Claude

```bash
agentlet init \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

#### DeepSeek

```bash
agentlet init \
  --provider openai \
  --model deepseek/deepseek-chat
```

#### Azure OpenAI

```bash
agentlet init \
  --provider azure \
  --model gpt-5.4
```

#### Google Gemini

```bash
agentlet init \
  --provider gemini \
  --model gemini/gemini-1.5-flash
```

#### Groq

```bash
agentlet init \
  --provider groq \
  --model groq/llama-3.1-70b-versatile
```

#### Together AI

```bash
agentlet init \
  --provider together_ai \
  --model together_ai/llama-3.1-70b
```

## CLI Usage

### One-Shot Mode

For scripting and single-turn interactions:

```bash
# Direct message
agentlet chat "Explain Python decorators"

# Pipe from stdin
echo "What is the capital of France?" | agentlet chat

# Read from file
agentlet chat --print < prompt.txt

# Force non-interactive output
agentlet chat --print "Simple output without formatting"
```

### Interactive Mode

Start an interactive session with full streaming output:

```bash
# New session
agentlet chat
```

#### Interactive Slash Commands

Once in interactive mode, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Display current session and model info |
| `/history` | Show recent turn summaries |
| `/new` | Start a fresh session |
| `/clear` | Clear terminal screen (preserves session) |
| `/exit` | Exit the session |

#### Interactive Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` (during generation) | Cancel current turn, keep session |
| `Ctrl+C` (while idle) | Clear input buffer |
| `Ctrl+C` twice quickly | Exit session |
| `Ctrl+D` | Exit session cleanly |

### Session Management

Sessions are automatically persisted to `~/.agentlet/sessions/{cwd_hash}/`:

```bash
# List your sessions (manual - files are in ~/.agentlet/sessions/)
ls ~/.agentlet/sessions/*/
```

Session behavior:
- Each working directory has isolated sessions (identified by path hash)
- Only completed turns are persisted (cancelled/failed turns are discarded)
- Session headers store non-sensitive settings (model, temperature, etc.)
- Each `agentlet chat` launch starts a fresh interactive session
- History is global at `~/.agentlet/history`

## Development

### Run Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# Smoke tests
uv run pytest tests/smoke/

# Live API tests (requires API key)
AGENTLET_RUN_REAL_API_TESTS=1 uv run pytest tests/test_real_api.py -v
```

### Project Structure

```
agentlet/
├── src/agentlet/
│   ├── agent/          # Core agent loop and runtime
│   │   ├── agent_loop.py
│   │   ├── context.py
│   │   ├── providers/  # LLM provider adapters
│   │   ├── tools/      # Tool registry
│   │   └── prompts/    # System prompts
│   ├── cli/            # CLI interface
│   │   ├── main.py     # Entry point
│   │   ├── chat_app.py # Mode selection
│   │   ├── sessions.py # Session persistence
│   │   ├── repl.py     # Interactive loop
│   │   └── presenter.py # Rich rendering
│   └── settings.py     # Settings management
├── tests/
│   ├── unit/           # Unit tests
│   └── smoke/          # Integration tests
└── docs/               # Design docs
```

## Notes

## Documentation

- **[CLI Usage Guide](docs/cli-usage.md)** - Complete guide for configuration, commands, and examples
- **[Phase 1 Design](docs/design-docs/phase-1-foundation.md)** - Core architecture and runtime design
- **[Phase 2 Design](docs/design-docs/phase-2-cli-experience.md)** - CLI and interactive experience design

## Notes

- SOCKS proxy support is included through `httpx[socks]`
- `LiteLLM` may require provider-prefixed model names for some backends
- Interactive sessions persist only completed turns; cancelled or failed turns are not committed
- Session headers persist non-sensitive provider settings for transcript inspection and debugging
