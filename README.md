# agentlet

A minimal Python agent harness.

## Status

Current phase:

- Single-process agent loop with tool support
- Interactive TTY chat with streaming output
- Persisted cwd-scoped session transcripts under `~/.agentlet/sessions/` (grouped by working directory hash)
- Independent `Context` with conversation history management
- `LiteLLM` provider integration supporting multiple LLM backends
- User-level `~/.agentlet/settings.json` configuration
- Configurable tool policies (write, bash, network permissions)
- Interactive approval prompts for unsafe operations

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Settings

Run `agentlet init` to create your settings file:

```bash
# OpenAI (default)
agentlet init
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
  "api_key": null,
  "api_base": null,
  "temperature": 0.0,
  "max_tokens": null,
  "max_iterations": 8,
  "max_html_extract_bytes": 2000000,
  "allow_write": null,
  "allow_bash": null,
  "allow_network": null
}
```

### Configuration Priority

Settings are resolved in this priority order (highest first):

1. **CLI arguments** (highest priority)
2. **Settings file** (`~/.agentlet/settings.json`)
3. **Built-in defaults**

| Setting | Default | Description |
|---------|---------|-------------|
| provider | `openai` | LLM provider name |
| model | `gpt-5.4` | Model identifier |
| api_key | `null` | API key for authentication |
| api_base | `null` | Custom API base URL |
| temperature | `0.0` | Sampling temperature |
| max_tokens | `null` | Maximum tokens per response |
| max_iterations | `8` | Maximum tool iterations per turn |
| max_html_extract_bytes | `2000000` | Byte limit for HTML text extraction |
| allow_write | `null` | Enable Write/Edit tools (defaults to true) |
| allow_bash | `null` | Enable Bash tool (defaults to true) |
| allow_network | `null` | Enable WebSearch/WebFetch tools (defaults to true) |

### Provider Examples

#### Anthropic Claude

```bash
agentlet init \
  --provider anthropic \
  --model claude-sonnet-4-6
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

#### Cohere

```bash
agentlet init \
  --provider cohere \
  --model cohere/command-r
```

#### Mistral

```bash
agentlet init \
  --provider mistral \
  --model mistral/mistral-large
```

#### Fireworks

```bash
agentlet init \
  --provider fireworks \
  --model fireworks/llama-3.1-70b
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

### Chat Options

| Option | Description |
|--------|-------------|
| `--provider` | Override the provider for this run |
| `--model` | Override the model for this run |
| `--temperature` | Override the temperature for this run |
| `--max-tokens` | Set maximum tokens for this run |
| `--max-iterations` | Maximum tool iterations per turn (default: 8) |
| `--max-html-extract-bytes` | Byte limit for HTML extraction |
| `--auto-approve` | Automatically approve all tool actions |
| `--deny-write` | Disable Write and Edit tools |
| `--deny-bash` | Disable Bash tool |
| `--deny-network` | Disable WebSearch and WebFetch tools |
| `--print` | Force one-shot print mode |

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
| `Enter` | Submit message |
| `Alt+Enter` | Insert newline in message |
| `Ctrl+C` (during generation) | Cancel current turn, keep session |
| `Ctrl+C` (while idle) | Clear input buffer (press twice within 2s to exit) |
| `Ctrl+D` or `Ctrl+C` twice | Exit session cleanly |

### Session Management

Sessions are automatically persisted to `~/.agentlet/sessions/{cwd_hash}/`:

```bash
# List your sessions
ls ~/.agentlet/sessions/*/
```

Session behavior:
- Each working directory has isolated sessions (identified by SHA256 hash of the path)
- Only completed turns are persisted (cancelled/failed turns are discarded)
- Session headers store non-sensitive settings (model, temperature, system prompt)
- Each `agentlet chat` launch starts a fresh interactive session
- Command history is global at `~/.agentlet/history`
- Session transcripts are stored as JSONL files with schema versioning

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
│   ├── agent/               # Core agent loop and runtime
│   │   ├── agent_loop.py    # Agent orchestration loop
│   │   ├── context.py       # Conversation context management
│   │   ├── providers/       # LLM provider adapters
│   │   │   ├── litellm_provider.py  # LiteLLM integration
│   │   │   └── registry.py          # Provider registry
│   │   ├── tools/           # Built-in tools
│   │   │   ├── bash.py
│   │   │   ├── builtins.py
│   │   │   ├── local_fs.py
│   │   │   ├── policy.py
│   │   │   ├── registry.py
│   │   │   └── web.py
│   │   └── prompts/         # System prompts
│   │       └── system_prompt.py
│   ├── cli/                 # CLI interface
│   │   ├── main.py          # Entry point and argument parsing
│   │   ├── chat_app.py      # Chat mode selection and wiring
│   │   ├── commands.py      # Slash command parsing
│   │   ├── sessions.py      # Session persistence and loading
│   │   ├── repl.py          # Interactive REPL loop
│   │   ├── presenter.py     # Rich console output
│   │   ├── prompt.py        # Prompt session setup
│   │   └── approvals.py     # Interactive tool approval
│   └── settings.py          # Settings management
├── tests/
│   ├── unit/                # Unit tests
│   ├── smoke/               # Integration tests
│   └── test_real_api.py     # Live API tests
└── docs/                    # Design docs
    └── design-docs/
```

## Built-in Tools

The agentlet comes with several built-in tools that can be enabled/disabled via settings or CLI flags:

### Always Enabled (Read-Only)
- **Read** - Read file contents
- **Glob** - Find files matching a pattern
- **Grep** - Search file contents

### Network Tools (gated by `allow_network`)
- **WebSearch** - Search the web using DuckDuckGo
- **WebFetch** - Fetch and extract text from web pages

### File Modification Tools (gated by `allow_write`)
- **Write** - Write content to files
- **Edit** - Edit existing files

### System Tools (gated by `allow_bash`)
- **Bash** - Execute shell commands

## Documentation

- **[CLI Usage Guide](docs/cli-usage.md)** - Complete guide for configuration, commands, and examples
- **[Phase 1 Design](docs/design-docs/phase-1-foundation.md)** - Core architecture and runtime design
- **[Phase 2 Design](docs/design-docs/phase-2-cli-experience.md)** - CLI and interactive experience design
- **[Phase 3 TUI Refinement](docs/design-docs/phase-3-tui-refinement.md)** - Terminal UI improvements

## Notes

- SOCKS proxy support is included through `httpx[socks]`
- `LiteLLM` may require provider-prefixed model names for some backends
- Interactive sessions persist only completed turns; cancelled or failed turns are not committed
- Session headers persist non-sensitive provider settings for transcript inspection and debugging
- Settings file uses `0o600` permissions on Unix systems for security
- Legacy `setting.json` files are automatically migrated to `settings.json`
