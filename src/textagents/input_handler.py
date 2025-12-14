"""Handle input processing for TextAgents.

This module provides:
- File loading with @filepath syntax
- Magic variable interpolation (CURRENT_DATE, etc.)
- Type coercion for input values
- Input validation
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import InputTypeError, MissingInputError, TemplateError
from .parser import AgentSpec, InputDefinition

# Magic variables that are auto-filled if not provided (UPPERCASE)
MAGIC_VARIABLES: dict[str, Callable[[], str]] = {
    "CURRENT_DATE": lambda: datetime.now().strftime("%Y-%m-%d"),
    "CURRENT_TIME": lambda: datetime.now().strftime("%H:%M:%S"),
    "CURRENT_DATETIME": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}


def process_inputs(
    raw_inputs: dict[str, Any],
    spec: AgentSpec,
) -> dict[str, Any]:
    """Process and validate inputs for template interpolation.

    This function:
    1. Loads file contents for @filepath values
    2. Applies magic variables if not provided
    3. Coerces types where possible
    4. Validates required inputs are present

    Args:
        raw_inputs: Raw input values from user
        spec: The agent specification

    Returns:
        Processed inputs ready for template interpolation

    Raises:
        MissingInputError: If required inputs are missing
        InputTypeError: If type coercion fails
    """
    processed: dict[str, Any] = {}

    # Build input definitions lookup
    input_defs = {d.name: d for d in spec.input_definitions}

    # Process provided inputs
    for name, value in raw_inputs.items():
        # Handle @file syntax
        if isinstance(value, str) and value.startswith("@"):
            value = _load_file_input(name, value[1:])

        # Coerce to expected type if defined
        if name in input_defs:
            value = _coerce_type(value, input_defs[name])

        processed[name] = value

    # Get all placeholders that need values
    all_placeholders = spec.all_placeholders

    # Apply magic variables for missing placeholders
    for placeholder in all_placeholders:
        if placeholder not in processed and placeholder in MAGIC_VARIABLES:
            processed[placeholder] = MAGIC_VARIABLES[placeholder]()

    # Validate required inputs
    _validate_required_inputs(processed, input_defs, all_placeholders)

    return processed


def _load_file_input(input_name: str, file_path: str) -> str:
    """Load input value from a file.

    Args:
        input_name: Name of the input (for error messages)
        file_path: Path to the file (without @ prefix)

    Returns:
        File contents as string

    Raises:
        MissingInputError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise MissingInputError.file_not_found(input_name, file_path)

    return path.read_text()


def _coerce_type(value: Any, definition: InputDefinition) -> Any:
    """Coerce input value to the expected type.

    Args:
        value: The input value
        definition: The input definition with expected type

    Returns:
        Coerced value

    Raises:
        InputTypeError: If coercion fails
    """
    target_type = definition.type

    # Already correct type
    if target_type == "str":
        return str(value)

    if target_type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise InputTypeError.cannot_coerce(definition.name, value, "int")

    if target_type == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise InputTypeError.cannot_coerce(definition.name, value, "float")

    if target_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "1", "yes", "on"):
                return True
            if lower in ("false", "0", "no", "off"):
                return False
        raise InputTypeError.cannot_coerce(definition.name, value, "bool")

    # For complex types (list[str], list[int]), pass through
    return value


def _validate_required_inputs(
    processed: dict[str, Any],
    input_defs: dict[str, InputDefinition],
    all_placeholders: set[str],
) -> None:
    """Validate that all required inputs are provided.

    Args:
        processed: Processed input values
        input_defs: Input definitions from spec
        all_placeholders: All placeholders from template

    Raises:
        MissingInputError: If required inputs are missing
    """
    missing: list[str] = []
    optional_in_prompt: list[str] = []

    # Check defined inputs
    for name, definition in input_defs.items():
        if not definition.optional and name not in processed:
            missing.append(name)

    # Check placeholders that aren't magic variables or defined inputs
    for placeholder in all_placeholders:
        if placeholder in processed:
            continue

        if placeholder in MAGIC_VARIABLES:
            continue

        # If placeholder is a defined optional input, track it as missing so we fail fast
        if placeholder in input_defs and input_defs[placeholder].optional:
            optional_in_prompt.append(placeholder)
            missing.append(placeholder)
            continue

        if placeholder not in input_defs:
            # Undefined placeholder - required by virtue of being in the template
            missing.append(placeholder)

    if missing:
        all_expected = list(input_defs.keys()) + [
            p
            for p in all_placeholders
            if p not in input_defs and p not in MAGIC_VARIABLES
        ]
        raise MissingInputError.missing_required(
            missing,
            list(processed.keys()),
            all_expected,
            optional_placeholders=optional_in_prompt or None,
        )


def interpolate_template(template: str, inputs: dict[str, Any]) -> str:
    """Interpolate a template string with input values.

    Uses Python's str.format() for interpolation and converts missing
    placeholders into TemplateError for clearer error handling.

    Args:
        template: Template string with {placeholders}
        inputs: Input values to interpolate

    Returns:
        Interpolated string

    Raises:
        TemplateError: If a placeholder is missing from inputs
    """
    try:
        return template.format(**inputs)
    except KeyError as exc:  # pragma: no cover - exercised via tests
        placeholder = str(exc).strip("'\"")
        raise TemplateError.missing_placeholder(
            placeholder, list(inputs.keys())
        ) from exc
