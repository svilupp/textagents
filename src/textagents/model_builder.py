"""Build dynamic Pydantic models from field definitions.

This module creates Pydantic BaseModel subclasses at runtime from
the parsed field definitions. The generated models are used as
output_type for PydanticAI agents.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, create_model

from .parser import AgentSpec, FieldDefinition

# Mapping from TOML type strings to Python types
TYPE_MAP: dict[str, type[Any]] = {
    "bool": bool,
    "str": str,
    "int": int,
    "float": float,
    "list[str]": list[str],
    "list[int]": list[int],
}


def build_output_model(spec: AgentSpec) -> type[BaseModel]:
    """Build a Pydantic model from AgentSpec.

    Creates a dynamic Pydantic model with fields matching the
    output_type specification. Fields are ordered with 'reasoning'
    first (if present) to encourage chain-of-thought.

    Args:
        spec: The parsed agent specification

    Returns:
        A Pydantic BaseModel subclass
    """
    fields: dict[str, tuple[type[Any], Any]] = {}

    # Sort fields: reasoning first, then alphabetically
    sorted_fields = sorted(
        spec.output_fields,
        key=lambda f: (0 if f.name == "reasoning" else 1, f.name),
    )

    for field_def in sorted_fields:
        python_type = _get_python_type(field_def)
        field_info = _build_field_info(field_def)

        # Handle optional fields
        if field_def.optional:
            python_type = python_type | None

        fields[field_def.name] = (python_type, field_info)  # type: ignore[assignment]

    # Create the model
    model = create_model(  # type: ignore[no-matching-overload]
        spec.output_type_name,
        __doc__=spec.output_type_description or "",
        **fields,
    )

    return model


def _get_python_type(field_def: FieldDefinition) -> type[Any]:
    """Convert field definition to Python type.

    Handles:
    - Basic types (bool, str, int, float)
    - List types (list[str], list[int])
    - Enum constraints (converted to Literal)

    Args:
        field_def: The field definition

    Returns:
        The Python type
    """
    # Handle enum - convert to Literal type
    if field_def.enum is not None:
        # Create Literal type from enum values
        # Literal requires at least one argument
        return Literal[field_def.enum]  # type: ignore[return-value]

    return TYPE_MAP[field_def.type]


def _build_field_info(field_def: FieldDefinition) -> Any:
    """Build Pydantic Field with constraints.

    Args:
        field_def: The field definition

    Returns:
        Pydantic Field instance
    """
    kwargs: dict[str, Any] = {}

    # Description
    if field_def.description:
        kwargs["description"] = field_def.description

    # Default for optional fields
    if field_def.optional:
        kwargs["default"] = None

    # String constraints
    if field_def.min_length is not None:
        kwargs["min_length"] = field_def.min_length
    if field_def.max_length is not None:
        kwargs["max_length"] = field_def.max_length
    if field_def.pattern is not None:
        kwargs["pattern"] = field_def.pattern

    # Numeric constraints
    if field_def.ge is not None:
        kwargs["ge"] = field_def.ge
    if field_def.le is not None:
        kwargs["le"] = field_def.le
    if field_def.gt is not None:
        kwargs["gt"] = field_def.gt
    if field_def.lt is not None:
        kwargs["lt"] = field_def.lt

    # List constraints (mapped to min_length/max_length for sequences)
    if field_def.min_items is not None:
        kwargs["min_length"] = field_def.min_items
    if field_def.max_items is not None:
        kwargs["max_length"] = field_def.max_items

    return Field(**kwargs)


def get_field_metadata(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Extract field metadata from a Pydantic model.

    Useful for validation and introspection.

    Args:
        model: The Pydantic model class

    Returns:
        Dict mapping field names to their metadata
    """
    metadata: dict[str, dict[str, Any]] = {}

    for name, field_info in model.model_fields.items():
        field_meta: dict[str, Any] = {
            "required": field_info.is_required(),
            "type": str(field_info.annotation),
        }

        if field_info.description:
            field_meta["description"] = field_info.description

        # Extract constraints from metadata
        for constraint in field_info.metadata:
            if hasattr(constraint, "gt"):
                field_meta["gt"] = constraint.gt
            if hasattr(constraint, "ge"):
                field_meta["ge"] = constraint.ge
            if hasattr(constraint, "lt"):
                field_meta["lt"] = constraint.lt
            if hasattr(constraint, "le"):
                field_meta["le"] = constraint.le
            if hasattr(constraint, "min_length"):
                field_meta["min_length"] = constraint.min_length
            if hasattr(constraint, "max_length"):
                field_meta["max_length"] = constraint.max_length
            if hasattr(constraint, "pattern"):
                field_meta["pattern"] = constraint.pattern

        # Check for Literal types (enum)
        annotation = field_info.annotation
        if annotation is not None:
            # Handle Optional types
            origin = getattr(annotation, "__origin__", None)
            if origin is Literal:
                field_meta["enum"] = list(get_args(annotation))

        metadata[name] = field_meta

    return metadata
