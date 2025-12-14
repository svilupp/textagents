"""Build automatic output validators for TextAgents.

This module creates output validator functions that are registered
with PydanticAI agents to provide additional validation and helpful
retry messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry

from .parser import AgentSpec, FieldDefinition

if TYPE_CHECKING:
    from pydantic_ai import RunContext

OutputT = TypeVar("OutputT", bound=BaseModel)


def add_output_validator(
    agent: Agent[None, OutputT],
    spec: AgentSpec,
) -> None:
    """Add automatic output validator to the agent.

    The validator checks:
    - Required fields are present and not None
    - Enum values are within allowed set
    - String length constraints
    - Numeric bounds

    While Pydantic handles most validation, this provides clearer
    error messages that help the LLM retry correctly.

    Args:
        agent: The PydanticAI agent to add validator to
        spec: The agent specification with field definitions
    """
    # Build lookup for field definitions
    field_defs = {f.name: f for f in spec.output_fields}

    @agent.output_validator
    def _validate_output(
        ctx: RunContext[None],  # noqa: ARG001
        output: OutputT,
    ) -> OutputT:
        """Validate output and provide helpful retry messages."""
        errors: list[str] = []

        for field_name, field_def in field_defs.items():
            value = getattr(output, field_name, None)

            # Check required fields
            if not field_def.optional and value is None:
                errors.append(f"'{field_name}' is required but was None")
                continue

            if value is None:
                continue  # Optional field, skip validation

            # Validate specific constraints
            field_errors = _validate_field(field_def, value)
            errors.extend(field_errors)

        if errors:
            raise ModelRetry("Output validation failed:\n- " + "\n- ".join(errors))

        return output


def _validate_field(field_def: FieldDefinition, value: Any) -> list[str]:
    """Validate a single field value against its definition.

    Args:
        field_def: The field definition
        value: The field value

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    # Enum validation
    if field_def.enum is not None and value not in field_def.enum:
        errors.append(
            f"'{field_def.name}' must be one of {list(field_def.enum)}, got {value!r}"
        )

    # String length validation
    if isinstance(value, str):
        if field_def.max_length is not None and len(value) > field_def.max_length:
            errors.append(
                f"'{field_def.name}' exceeds max_length of {field_def.max_length} "
                f"(got {len(value)} chars)"
            )
        if field_def.min_length is not None and len(value) < field_def.min_length:
            errors.append(
                f"'{field_def.name}' below min_length of {field_def.min_length} "
                f"(got {len(value)} chars)"
            )

    # Numeric bounds validation
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if field_def.ge is not None and value < field_def.ge:
            errors.append(f"'{field_def.name}' must be >= {field_def.ge}, got {value}")
        if field_def.le is not None and value > field_def.le:
            errors.append(f"'{field_def.name}' must be <= {field_def.le}, got {value}")
        if field_def.gt is not None and value <= field_def.gt:
            errors.append(f"'{field_def.name}' must be > {field_def.gt}, got {value}")
        if field_def.lt is not None and value >= field_def.lt:
            errors.append(f"'{field_def.name}' must be < {field_def.lt}, got {value}")

    # List length validation
    if isinstance(value, list):
        if field_def.max_items is not None and len(value) > field_def.max_items:
            errors.append(
                f"'{field_def.name}' exceeds max_items of {field_def.max_items} "
                f"(got {len(value)} items)"
            )
        if field_def.min_items is not None and len(value) < field_def.min_items:
            errors.append(
                f"'{field_def.name}' below min_items of {field_def.min_items} "
                f"(got {len(value)} items)"
            )

    return errors
