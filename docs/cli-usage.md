# CLI Usage Guide

Complete guide for using the `agentlet` CLI.

## Table of Contents

- [Configuration](#configuration)
  - [Settings File](#settings-file)
  - [Environment Variables](#environment-variables)
  - [Provider-Specific Setup](#provider-specific-setup)
- [Command Reference](#command-reference)
  - [agentlet init](#agentlet-init)
  - [agentlet chat](#agentlet-chat)
- [Usage Modes](#usage-modes)
  - [One-Shot Mode](#one-shot-mode)
  - [Interactive Mode](#interactive-mode)
- [Session Management](#session-management)
- [Examples](#examples)

---

## Configuration

### Settings File

Agentlet stores configuration in `~/.agentlet/settings.json`. Create it with:

```bash
agentlet init [options]
```

**Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--api-key` | API key for the provider | `sk-...` |
| `--api-base` | Base URL for API requests | `https://api.openai.com/v1` |
| `--provider` | Provider name | `openai`, `anthropic`, `azure` |
| `--model` | Model identifier | `gpt-5.4`, `claude-3-5-sonnet` |
| `--temperature` | Sampling temperature | `0.0` to `1.0` |
| `--max-tokens` | Maximum tokens per response | `4096` |
| `--force` | Overwrite existing settings | - |

**Settings File Format:**

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

### Environment Variables

Environment variables override settings file values. This is useful for temporary changes or CI/CD environments.

**Generic Variables:**

| Variable | Purpose |
|----------|---------|
| `AGENTLET_PROVIDER` | Override provider |
| `AGENTLET_MODEL` | Override model |
| `AGENTLET_API_KEY` | Override API key (highest priority) |
| `AGENTLET_BASE_URL` | Override base URL |

**Provider-Specific Variables:**

Agentlet follows LiteLLM's standard environment variable naming:

| Provider | API Key Env | Base URL Env |
|----------|-------------|--------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Azure | `AZURE_API_KEY` | `AZURE_API_BASE` |
| Gemini | `GEMINI_API_KEY` | - |
| Groq | `GROQ_API_KEY` | - |
| Together AI | `TOGETHERAI_API_KEY` | `TOGETHERAI_BASE_URL` |
| Cohere | `COHERE_API_KEY` | - |
| Mistral | `MISTRAL_API_KEY` | - |
| Fireworks | `FIREWORKS_API_KEY` | `FIREWORKS_BASE_URL` |
| Anyscale | `ANYSCALE_API_KEY` | `ANYSCALE_BASE_URL` |

**Priority Order:**

1. `AGENTLET_API_KEY` / `AGENTLET_BASE_URL` (project-specific override)
2. Provider-specific env (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
3. OpenAI-compatible fallback (`OPENAI_API_KEY`, `OPENAI_BASE_URL`)
4. Settings file value
5. Built-in default

### Provider-Specific Setup

#### OpenAI (Default)

```bash
agentlet init \
  --api-key $OPENAI_API_KEY \
  --api-base https://api.openai.com/v1 \
  --model gpt-5.4
```

**Environment fallback:** `OPENAI_API_KEY`, `OPENAI_BASE_URL`

#### Anthropic Claude

```bash
agentlet init \
  --provider anthropic \
  --api-key $ANTHROPIC_API_KEY \
  --model claude-3-5-sonnet-20241022
```

**Note:** Anthropic models don't use `--api-base`.

**Environment fallback:** `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`

#### DeepSeek

```bash
agentlet init \
  --api-key $DEEPSEEK_API_KEY \
  --api-base https://api.deepseek.com/v1 \
  --model deepseek/deepseek-chat
```

**Note:** DeepSeek uses LiteLLM provider prefix format.

#### Azure OpenAI

```bash
agentlet init \
  --provider azure \
  --api-key $AZURE_API_KEY \
  --api-base https://your-resource.openai.azure.com/
```

**Environment fallback:** `AZURE_API_KEY`, `AZURE_API_BASE`

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
  --api-base https://api.together.xyz/v1 \
  --model together_ai/llama-3.1-70b
```

---

## Command Reference

### agentlet init

Initialize or update the settings file.

```bash
# Create new settings (fails if exists)
agentlet init --api-key sk-... --model gpt-5.4

# Update existing settings
agentlet init --api-key sk-... --force

# Set temperature and token limit
agentlet init \
  --api-key sk-... \
  --temperature 0.7 \
  --max-tokens 2048
```

### agentlet chat

Send messages to the LLM.

**Usage:**

```bash
agentlet chat [options] [message]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--print` | Force one-shot output mode |

---

## Usage Modes

### One-Shot Mode

Use for scripting, automation, or single questions.

**Message as argument:**

```bash
agentlet chat "Explain Python decorators"
```

**Pipe from stdin:**

```bash
echo "What is the capital of France?" | agentlet chat
cat prompt.txt | agentlet chat
```

**Force non-interactive output:**

```bash
agentlet chat --print "Simple text output"
```

**One-shot with overrides:**

```bash
agentlet chat \
  --model gpt-4o \
  --temperature 0.5 \
  "Write a Python function to calculate fibonacci"
```

### Interactive Mode

Default when run in a TTY without a message argument.

**Start new session:**

```bash
agentlet chat
```

**Resume latest session:**

```bash
agentlet chat --continue
```

#### Interactive Commands

Type these during an interactive session:

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/status` | Show session info (model, provider, session ID) |
| `/history` | Show recent conversation turns |
| `/new` | Start a new session |
| `/clear` | Clear terminal screen |
| `/exit` | Exit session |

#### Keyboard Shortcuts

| Shortcut | When | Action |
|----------|------|--------|
| `Ctrl+C` | During generation | Cancel turn, keep session |
| `Ctrl+C` | Idle | Clear input buffer |
| `Ctrl+C` `Ctrl+C` | Idle (quickly) | Exit session |
| `Ctrl+D` | Empty prompt | Exit session |

---

## Session Management

Sessions are stored in `~/.agentlet/sessions/{cwd_hash}/`:

- `cwd_hash`: First 16 chars of MD5 hash of current working directory
- Each directory gets isolated session storage
- Sessions persist only completed turns
- Latest session pointer stored in `latest` file

**Session Files:**

```
~/.agentlet/
├── sessions/
│   └── {cwd_hash}/
│       ├── latest                 # Points to latest session ID
│       └── {timestamp}-{id}.jsonl # Session transcript
└── history                        # Global prompt history
```

**Find Your Sessions:**

```bash
# Get current directory hash
cwd_hash=$(python3 -c "import hashlib; print(hashlib.md5('$(pwd)'.encode()).hexdigest()[:16])")

# List sessions for current directory
ls ~/.agentlet/sessions/$cwd_hash/

# View latest session ID
cat ~/.agentlet/sessions/$cwd_hash/latest
```

---

## Examples

### Basic One-Shot

```bash
# Simple question
agentlet chat "What is 2+2?"

# With heredoc
agentlet chat << 'EOF'
Write a Python function that:
1. Takes a list of numbers
2. Returns the sum
3. Handles empty lists gracefully
EOF
```

### Scripting Examples

```bash
# Process files
for file in *.py; do
    agentlet chat --print "Review this code for bugs: $(cat $file)"
done

# Batch prompts
while read prompt; do
    agentlet chat --print "$prompt" >> responses.txt
done < prompts.txt
```

### Interactive Workflow

```bash
# Start a coding session
cd my-project
agentlet chat

# Later, continue from same directory
agentlet chat --continue

# Or start fresh
agentlet chat --new-session
```

### Provider Switching

```bash
# Use OpenAI for quick tasks
agentlet chat "Hello"

# Use Claude for complex reasoning
agentlet chat --provider anthropic --model claude-3-5-sonnet "Analyze this..."

# Use local model via base URL
agentlet chat --api-base http://localhost:11434/v1 --model llama2 "Hello"
```


---

## Troubleshooting

### Settings Not Found

```bash
# Check if settings file exists
ls -la ~/.agentlet/settings.json

# Re-initialize
agentlet init --api-key sk-... --force
```

### API Key Issues

```bash
# Test with explicit key
agentlet chat --api-key sk-... "Test"

# Check environment variables
env | grep -i api_key
```

### Connection Issues

```bash
# Test with explicit base URL
agentlet chat --api-base https://api.openai.com/v1 "Test"

# Enable SOCKS proxy (if configured)
export HTTPS_PROXY=socks5://localhost:1080
```
