# API Reference

## load_agent

::: textagents.load_agent
    options:
      show_root_heading: true
      heading_level: 3

```python
import textagents

# Basic
agent = textagents.load_agent("agent.txt")

# With model override
agent = textagents.load_agent("agent.txt", model_override="openai:gpt-4o-mini")

# With Logfire
agent = textagents.load_agent("agent.txt", logfire_token="...")
```

## TextAgent

::: textagents.TextAgent
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - run
        - run_sync
        - name
        - model
        - input_names
        - required_inputs

```python
# Async
result = await agent.run(text="Hello")

# Sync
result = agent.run_sync(text="Hello")

# Properties
agent.name            # Agent name
agent.model           # Model string
agent.required_inputs # List of required input names
```

## Results

Results are Pydantic model instances:

```python
result.reasoning      # Access fields
result.model_dump()   # Convert to dict
result.model_dump_json()  # Convert to JSON
```
