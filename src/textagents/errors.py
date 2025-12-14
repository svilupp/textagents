"""Custom exceptions for TextAgents.

All exceptions inherit from TextAgentsError for easy catching.
Each exception provides helpful error messages with context about
what went wrong, what was expected, and how to fix it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class TextAgentsError(Exception):
    """Base exception for all TextAgents errors."""

    pass


class AgentDefinitionError(TextAgentsError):
    """Invalid agent definition file.

    Raised when:
    - Missing required [agent] section
    - Missing model specification
    - No output_type fields defined
    - Invalid type specification
    - Unsupported type
    """

    @classmethod
    def missing_section(cls, section: str) -> AgentDefinitionError:
        """Create error for missing required section."""
        return cls(
            f"Missing required section '[{section}]' in agent definition.\n\n"
            f"Expected:\n"
            f"  [{section}]\n"
            f'  model = "openai:gpt-5"\n'
        )

    @classmethod
    def missing_field(
        cls, section: str, field: str, example: str
    ) -> AgentDefinitionError:
        """Create error for missing required field."""
        return cls(
            f"Missing required field '{field}' in [{section}] section.\n\n"
            f"Expected:\n"
            f"  [{section}]\n"
            f"  {field} = {example}\n"
        )

    @classmethod
    def no_output_fields(cls) -> AgentDefinitionError:
        """Create error for empty output_type."""
        return cls(
            "No fields defined in [agent.output_type].\n\n"
            "At least one output field is required. Example:\n\n"
            "  [agent.output_type]\n"
            '  reasoning = { type = "str", description = "Explanation" }\n'
            '  is_valid = { description = "Whether valid" }\n'
        )

    @classmethod
    def unsupported_type(
        cls,
        field_name: str,
        type_str: str,
        supported: Sequence[str],
    ) -> AgentDefinitionError:
        """Create error for unsupported type."""
        return cls(
            f"Unsupported type '{type_str}' for field '{field_name}'.\n\n"
            f"Supported types: {', '.join(supported)}\n"
        )

    @classmethod
    def no_prompt_body(cls) -> AgentDefinitionError:
        """Create error for missing prompt body."""
        return cls(
            "No prompt body found after the TOML header.\n\n"
            "The agent definition must include a prompt template after '---'. Example:\n\n"
            "  ---\n"
            "  [agent]\n"
            '  model = "openai:gpt-5"\n'
            "  ...\n"
            "  ---\n"
            "  Evaluate the following input: {input}\n"
        )

    @classmethod
    def no_placeholders(cls) -> AgentDefinitionError:
        """Create error for prompt without placeholders."""
        return cls(
            "Prompt body has no {placeholders}.\n\n"
            "The prompt must contain at least one {variable} to interpolate.\n"
            "Example: 'Evaluate: {input}'\n"
        )

    @classmethod
    def invalid_input_type(
        cls,
        input_name: str,
        type_str: str,
    ) -> AgentDefinitionError:
        """Create error for invalid input_type value."""
        return cls(
            f"Invalid input_type for '{input_name}': expected table or string, got {type_str}.\n\n"
            "Valid examples:\n"
            "  [agent.input_type]\n"
            '  user_input = "str"\n'
            '  count = { type = "int", optional = true }\n'
        )


class MissingInputError(TextAgentsError):
    """Required input not provided.

    Raised when:
    - Required input variable not in run() kwargs
    - Input file (@ syntax) not found
    """

    @classmethod
    def missing_required(
        cls,
        missing: Sequence[str],
        provided: Sequence[str],
        all_inputs: Sequence[str],
        optional_placeholders: Sequence[str] | None = None,
    ) -> MissingInputError:
        """Create error for missing required inputs."""
        missing_str = ", ".join(f"'{m}'" for m in missing)
        provided_str = ", ".join(f"'{p}'" for p in provided) if provided else "(none)"
        all_str = ", ".join(f"'{a}'" for a in all_inputs)

        optional_note = ""
        if optional_placeholders:
            optional_list = ", ".join(f"'{o}'" for o in optional_placeholders)
            optional_note = (
                "\nOptional inputs were marked optional but appear in the prompt/instructions, "
                f"so they still need values: {optional_list}"
            )

        return cls(
            f"Missing required input(s): {missing_str}\n\n"
            f"Provided: {provided_str}\n"
            f"Expected: {all_str}{optional_note}\n"
        )

    @classmethod
    def file_not_found(cls, input_name: str, file_path: str) -> MissingInputError:
        """Create error for missing input file."""
        return cls(
            f"Input file not found for '{input_name}': {file_path}\n\n"
            f"When using @filepath syntax, the file must exist.\n"
        )


class InputTypeError(TextAgentsError):
    """Input cannot be coerced to expected type.

    Raised when:
    - String cannot be parsed to int/float/bool
    - Type mismatch for complex types
    """

    @classmethod
    def cannot_coerce(
        cls, input_name: str, value: object, target_type: str
    ) -> InputTypeError:
        """Create error for type coercion failure."""
        return cls(
            f"Cannot convert input '{input_name}' to {target_type}.\n\n"
            f"Received: {value!r} (type: {type(value).__name__})\n"
            f"Expected: {target_type}\n"
        )


class OutputValidationError(TextAgentsError):
    """Output validation failed after all retries.

    Wraps the underlying PydanticAI validation error with context.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class TemplateError(TextAgentsError):
    """Template interpolation failed.

    Raised when:
    - Placeholder in template not provided in inputs
    - Placeholder not in input_type and not a magic variable
    """

    @classmethod
    def missing_placeholder(
        cls, placeholder: str, available: Sequence[str]
    ) -> TemplateError:
        """Create error for unresolved placeholder."""
        available_str = (
            ", ".join(f"'{a}'" for a in available) if available else "(none)"
        )
        return cls(
            f"Template placeholder '{{{placeholder}}}' not found in inputs.\n\n"
            f"Available inputs: {available_str}\n"
            f"Magic variables: CURRENT_DATE, CURRENT_TIME, CURRENT_DATETIME\n"
        )
