# TextAgents

**Let agents write better agents.**

Define LLM agents in text files with TOML front-matter. Your coding agents can copy examples, tweak prompts, and build evaluation harnesses alongside your application—the runtime handles all the idiomatic PydanticAI usage.

!!! warning "Alpha Release"
    This package is in early development. Expect bugs, breaking changes, and incomplete features. Use at your own risk and please [report issues](https://github.com/svilupp/textagents/issues).

## Install

```bash
uv add textagents
```

## Quick Example

```toml title="judge.txt"
---
[agent]
model = "openai:gpt-4o"

[agent.output_type.reasoning]
type = "str"
description = "Step-by-step analysis."

[agent.output_type.is_valid]
description = "Whether valid."
---
Evaluate: {input}
```

```python
import textagents

agent = textagents.load_agent("judge.txt")
result = agent.run_sync(input="The sky is blue.")
print(result.is_valid)  # True
```

Or via CLI:

```bash
uv run textagents run judge.txt --input "The sky is blue."
```

## Next Steps

- [Agent Files](guide/agent-files.md) - Full TOML specification
- [CLI](guide/cli.md) - Command-line usage
- [Examples](examples/safety-judge.md) - Real-world agents
