# Logfire

Optional observability with [Pydantic Logfire](https://pydantic.dev/logfire).

## Setup

```bash
uv add textagents[logfire]
```

```python
import textagents

agent = textagents.load_agent(
    "agent.txt",
    logfire_token="your-token"  # or set LOGFIRE_TOKEN env var
)
```

## What's Traced

- Agent runs with timing
- LLM requests/responses
- Retries and validation errors
- Full inputs and outputs
