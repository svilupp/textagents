# Agent Files

Agent files use TOML front-matter followed by a prompt template.

## Structure

```toml
---
[agent]
model = "openai:gpt-4o"      # Required
name = "my_agent"            # Optional, defaults to filename
retries = 2                  # Optional, retry on validation failure

[agent.settings]
temperature = 0              # Model parameters

[agent.output_type]
# Field definitions
---
Your prompt with {placeholders}
```

## Output Types

Fields default to `bool` when type is omitted:

```toml
[agent.output_type.is_valid]
description = "Whether valid."  # bool

[agent.output_type.reasoning]
type = "str"
description = "Explanation."

[agent.output_type.score]
type = "int"
ge = 1
le = 10

[agent.output_type.confidence]
type = "float"
ge = 0.0
le = 1.0

[agent.output_type.tags]
type = "list[str]"
max_items = 5
```

### Constraints

| Constraint | Types | Description |
|------------|-------|-------------|
| `min_length`, `max_length` | str | Character limits |
| `pattern` | str | Regex pattern |
| `ge`, `le`, `gt`, `lt` | int, float | Numeric bounds |
| `min_items`, `max_items` | list | List length |
| `enum` | any | Allowed values |
| `optional` | any | Allow None |

### Enum Example

```toml
[agent.output_type.sentiment]
type = "str"
enum = ["positive", "negative", "neutral"]
```

## Input Types

Inputs are inferred from `{placeholders}`. Optionally document them:

```toml
[agent.input_type]
text = { type = "str", description = "Text to analyze" }
count = { type = "int", optional = true }
```

### Type Coercion

Strings are coerced: `"42"` → `42`, `"true"` → `True`.

### File Loading

Use `@` prefix to load from file:

```python
agent.run(text="@document.txt")
```

### Magic Variables

| Variable | Example |
|----------|---------|
| `{CURRENT_DATE}` | 2024-01-15 |
| `{CURRENT_TIME}` | 14:30:00 |
| `{CURRENT_DATETIME}` | 2024-01-15 14:30:00 |

## Reasoning Field

The `reasoning` field is placed first in the output model, encouraging chain-of-thought:

```toml
[agent.output_type.reasoning]
type = "str"
max_length = 500
description = "Think step-by-step before judging."

[agent.output_type.is_valid]
description = "Final judgment."
```
