# agentlet

A minimal Python agent harness.

## Status

Current phase:

- single-process agent loop with tool support
- interactive TTY chat with streaming output
- resumable cwd-scoped sessions under `~/.agentlet/sessions/` (grouped by working directory hash)
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
  --api-key your_api_key \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o-mini

# Or use environment variables temporarily
export OPENAI_API_KEY=your_api_key
agentlet chat "Hello"
```

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
  "model": "gpt-4o-mini",
  "api_key": "your_api_key",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.0,
  "max_tokens": null
}
```

### Configuration Priority

Settings are resolved in this priority order (highest first):

1. **Environment variables** (temporary overrides)
2. **Settings file** (`~/.agentlet/settings.json`)
3. **Built-in defaults**

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| provider | `AGENTLET_PROVIDER` | `openai` |
| model | `AGENTLET_MODEL` | `gpt-4o-mini` |
| api_key | `AGENTLET_API_KEY` → `{PROVIDER}_API_KEY` → `OPENAI_API_KEY` | - |
| api_base | `AGENTLET_BASE_URL` → `{PROVIDER}_BASE_URL` → `OPENAI_BASE_URL` | - |
| temperature | - | `0.0` |
| max_tokens | - | `null` |

### Provider Examples

#### Anthropic Claude

```bash
agentlet init \
  --provider anthropic \
  --api-key $ANTHROPIC_API_KEY \
  --model claude-3-5-sonnet-20241022
```

#### DeepSeek

```bash
agentlet init \
  --api-key $DEEPSEEK_API_KEY \
  --api-base https://api.deepseek.com/v1 \
  --model deepseek/deepseek-chat
```

#### Azure OpenAI

```bash
agentlet init \
  --provider azure \
  --api-key $AZURE_API_KEY \
  --api-base https://your-resource.openai.azure.com/
```

#### Google Gemini

```bash
agentlet init \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --model gemini/gemini-1.5-flash
```

#### Groq

```bash
agentlet init \
  --provider groq \
  --api-key $GROQ_API_KEY \
  --model groq/llama-3.1-70b-versatile
```

#### Together AI

```bash
agentlet init \
  --provider together_ai \
  --api-key $TOGETHERAI_API_KEY \
  --api-base https://api.together.xyz/v1
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

# Resume latest session in current directory
agentlet chat --continue

# Resume specific session by ID
agentlet chat --session 20260312T120000000000Z-deadbeef

# Force new session (ignore --continue)
agentlet chat --new-session
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

# Find session ID from latest pointer
cat ~/.agentlet/sessions/$(echo -n $(pwd) | md5 | cut -c1-16)/latest
```

Session behavior:
- Each working directory has isolated sessions (identified by path hash)
- Only completed turns are persisted (cancelled/failed turns are discarded)
- Session headers store non-sensitive settings (model, temperature, etc.)
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
- Session headers persist non-sensitive provider settings so resumed chats keep the same model endpoint and sampling limits
