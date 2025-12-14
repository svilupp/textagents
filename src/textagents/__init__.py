"""TextAgents - Declarative, file-based PydanticAI agents.

TextAgents extends TextPrompts to create declarative agents defined
in text files with TOML front-matter. The core use case is LLM-as-judge
with boolean criteria, but it supports any structured output.

Example:
    ```python
    import textagents

    # Load an agent from a file
    text_agent = textagents.load_agent("safety_judge.txt")

    # Run with inputs (async)
    result = await text_agent.run(
        user_input="What's the weather?",
        model_output="I can help with that!",
    )

    # Access structured output
    print(result.reasoning)
    print(result.is_safe)  # True or False
    ```

CLI:
    ```bash
    textagents run safety_judge.txt --user_input "Hello" --model_output "Hi!"
    textagents info safety_judge.txt
    textagents validate safety_judge.txt
    ```
"""

from .agent import TextAgent
from .errors import (
    AgentDefinitionError,
    InputTypeError,
    MissingInputError,
    OutputValidationError,
    TemplateError,
    TextAgentsError,
)
from .loader import load_agent

__all__ = [
    # Main API
    "load_agent",
    "TextAgent",
    # Errors
    "TextAgentsError",
    "AgentDefinitionError",
    "MissingInputError",
    "InputTypeError",
    "OutputValidationError",
    "TemplateError",
]

__version__ = "0.1.0"
