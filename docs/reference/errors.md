# Errors

## Exception Hierarchy

```
TextAgentsError (base)
├── AgentDefinitionError
├── MissingInputError
├── InputTypeError
├── OutputValidationError
└── TemplateError
```

::: textagents.TextAgentsError
    options:
      show_root_heading: true
      heading_level: 3

::: textagents.AgentDefinitionError
    options:
      show_root_heading: true
      heading_level: 3

::: textagents.MissingInputError
    options:
      show_root_heading: true
      heading_level: 3

::: textagents.InputTypeError
    options:
      show_root_heading: true
      heading_level: 3

::: textagents.OutputValidationError
    options:
      show_root_heading: true
      heading_level: 3

::: textagents.TemplateError
    options:
      show_root_heading: true
      heading_level: 3

## Error Handling

```python
import textagents

try:
    agent = textagents.load_agent("agent.txt")
    result = await agent.run()
except textagents.AgentDefinitionError:
    print("Fix your agent file")
except textagents.MissingInputError as e:
    print(f"Provide input: {e}")
except textagents.InputTypeError as e:
    print(f"Fix input type: {e}")
except textagents.OutputValidationError:
    print("LLM produced invalid output")
except textagents.TextAgentsError as e:
    print(f"TextAgents error: {e}")
```
