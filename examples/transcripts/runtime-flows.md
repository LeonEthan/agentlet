# Runtime Flow Transcripts

These transcripts mirror the behavior covered by
`tests/e2e/test_main_loop_e2e.py`.

## Local Coding Path

User task:

```text
Create notes.md
```

Assistant:

```text
I should write the file.
```

Tool call:

```json
{"name":"Write","arguments":{"path":"notes.md","content":"hello"}}
```

Runtime approval:

```text
Allow `Write` to run with the proposed arguments?
```

User decision:

```text
approved
```

Assistant:

```text
Proceeding with the approved write.
```

Tool result:

```text
Created file: notes.md
```

Assistant:

```text
notes.md is ready.
```

## Approval Refusal Path

User task:

```text
Create notes.md
```

Runtime approval:

```text
rejected
```

Tool result surfaced back to the model:

```text
Tool `Write` was not executed because approval was rejected.
```

Assistant:

```text
Skipping the write.
```

## Interrupt Path

User task:

```text
Pick a file to edit.
```

Assistant tool call:

```json
{
  "name": "AskUserQuestion",
  "arguments": {
    "prompt": "Which file should I edit?",
    "request_id": "question_1",
    "options": [
      {"value": "readme", "label": "README.md"},
      {"value": "arch", "label": "docs/ARCHITECTURE.md"}
    ]
  }
}
```

Runtime question:

```text
Which file should I edit?
```

User answer:

```text
readme
```

Assistant:

```text
Use README.md.
```
