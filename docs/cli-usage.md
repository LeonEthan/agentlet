# CLI Usage Guide

Guide for the `agentlet` CLI.

## Configuration

### Settings File

Agentlet reads configuration from `~/.agentlet/settings.json`. Create or update it with:

```bash
agentlet init [options]
```

Supported options:

| Option | Description | Example |
|--------|-------------|---------|
| `--provider` | Provider name | `openai`, `anthropic`, `azure` |
| `--model` | Model identifier | `gpt-5.4`, `claude-3-5-sonnet` |
| `--temperature` | Sampling temperature | `0.0` to `1.0` |
| `--max-tokens` | Maximum tokens per response | `4096` |
| `--force` | Overwrite existing settings | - |

Settings shape:

```json
{
  "provider": "openai",
  "model": "gpt-5.4",
  "api_key": "sk-...",
  "api_base": "https://api.openai.com/v1",
  "temperature": 0.0,
  "max_tokens": null
}
```

### Resolution Rules

Runtime configuration comes from:

1. `~/.agentlet/settings.json`
2. Built-in defaults for `provider`, `model`, and `temperature`

There is no environment-variable override layer.
Credentials and provider-specific base URLs must be edited directly in `~/.agentlet/settings.json`.

### Provider Examples

OpenAI:

```bash
agentlet init \
  --model gpt-5.4
```

Anthropic:

```bash
agentlet init \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

Azure OpenAI:

```bash
agentlet init \
  --provider azure \
  --model gpt-5.4
```

Gemini:

```bash
agentlet init \
  --provider gemini \
  --model gemini/gemini-1.5-flash
```

## Command Reference

### `agentlet init`

Create or update `~/.agentlet/settings.json`.

```bash
agentlet init --model gpt-5.4
agentlet init --model gpt-5.4 --force
agentlet init --temperature 0.7 --max-tokens 2048
```

### `agentlet chat`

Send a one-shot prompt or start an interactive TTY session.

Usage:

```bash
agentlet chat [options] [message]
```

Options:

| Option | Description |
|--------|-------------|
| `--print` | Force one-shot output mode |
| `--provider` | Override provider for this run |
| `--model` | Override model for this run |
| `--temperature` | Override sampling temperature for this run |
| `--max-tokens` | Override max tokens for this run |

Removed options:

| Option | Status |
|--------|--------|
| `--continue` | Removed |
| `--session` | Removed |
| `--new-session` | Removed |
| `--api-key` | Removed |
| `--api-base` | Removed |

## Usage Modes

### One-Shot Mode

```bash
agentlet chat "Explain Python decorators"
echo "What is the capital of France?" | agentlet chat
agentlet chat --print < prompt.txt
agentlet chat --print "Simple text output"
```

One-shot overrides:

```bash
agentlet chat \
  --model gpt-4o \
  --temperature 0.5 \
  "Write a Python function to calculate fibonacci"
```

### Interactive Mode

When `stdin` is a TTY and no message is provided, `agentlet chat` starts interactive mode:

```bash
agentlet chat
```

Interactive commands:

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/status` | Show session info |
| `/history` | Show recent conversation turns |
| `/new` | Start a new session |
| `/clear` | Clear terminal screen |
| `/exit` | Exit session |

Keyboard shortcuts:

| Shortcut | When | Action |
|----------|------|--------|
| `Ctrl+C` | During generation | Cancel turn, keep session |
| `Ctrl+C` | Idle | Clear input buffer |
| `Ctrl+C` `Ctrl+C` | Idle (quickly) | Exit session |
| `Ctrl+D` | Empty prompt | Exit session |

## Session Storage

Interactive transcripts are stored under `~/.agentlet/sessions/{cwd_hash}/`:

- `cwd_hash` is the first 16 characters of the MD5 hash of the resolved working directory
- only completed turns are persisted
- `latest` is still maintained as metadata, but there is no CLI resume flag
- each `agentlet chat` launch starts a fresh session

Layout:

```text
~/.agentlet/
├── sessions/
│   └── {cwd_hash}/
│       ├── latest
│       └── {timestamp}-{id}.jsonl
└── history
```

## Troubleshooting

Settings file:

```bash
ls -la ~/.agentlet/settings.json
agentlet init --model gpt-5.4 --force
```

Connection settings:

```bash
${EDITOR:-vi} ~/.agentlet/settings.json
agentlet chat "Test"
```
