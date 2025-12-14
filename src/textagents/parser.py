"""Parse agent definition files into AgentSpec.

This module handles parsing TOML front-matter from text files and
extracting the agent configuration, input/output type definitions,
and prompt template.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AgentDefinitionError

# Pre-compiled regex pattern for placeholder extraction
_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")

# Supported types for output fields
SUPPORTED_TYPES = frozenset(
    {
        "bool",
        "str",
        "int",
        "float",
        "list[str]",
        "list[int]",
    }
)

# Reserved keys in output_type that are not field definitions
OUTPUT_TYPE_METADATA_KEYS = frozenset({"name", "description"})


@dataclass(frozen=True, slots=True)
class InputDefinition:
    """Definition of an input variable."""

    name: str
    type: str = "str"
    description: str | None = None
    optional: bool = False


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    """Definition of an output field."""

    name: str
    type: str = "bool"
    description: str | None = None
    optional: bool = False
    enum: tuple[Any, ...] | None = None
    # Constraints
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_items: int | None = None
    max_items: int | None = None


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Parsed agent specification from TOML."""

    # Required
    model: str
    prompt_template: str
    output_fields: tuple[FieldDefinition, ...]

    # Optional agent config
    name: str | None = None
    instructions: str | None = None
    retries: int = 2

    # Model settings
    settings: dict[str, Any] = field(default_factory=dict)

    # Input definitions
    input_definitions: tuple[InputDefinition, ...] = field(default_factory=tuple)

    # Output type metadata
    output_type_name: str = "AgentOutput"
    output_type_description: str | None = None

    # Source file (for error messages and naming)
    source_path: Path | None = None

    @property
    def placeholders(self) -> set[str]:
        """Extract all {placeholder} names from the prompt template."""
        return set(_PLACEHOLDER_PATTERN.findall(self.prompt_template))

    @property
    def instruction_placeholders(self) -> set[str]:
        """Extract all {placeholder} names from instructions if present."""
        if not self.instructions:
            return set()
        return set(_PLACEHOLDER_PATTERN.findall(self.instructions))

    @property
    def all_placeholders(self) -> set[str]:
        """All placeholders from both prompt and instructions."""
        return self.placeholders | self.instruction_placeholders


def parse_agent_file(path: Path) -> AgentSpec:
    """Parse an agent definition file.

    Args:
        path: Path to the agent file (.txt with TOML front-matter)

    Returns:
        Parsed AgentSpec

    Raises:
        FileNotFoundError: If file doesn't exist
        AgentDefinitionError: If definition is invalid
    """
    content = path.read_text()
    meta, prompt_body = _parse_front_matter(content)
    return parse_agent_spec(meta, prompt_body, path)


def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Parse TOML front-matter from content.

    Front-matter is delimited by --- lines:
    ---
    [agent]
    model = "..."
    ---
    Prompt body here

    Args:
        content: Full file content

    Returns:
        Tuple of (parsed TOML dict, prompt body)
    """
    # Check for front-matter delimiters
    if not content.startswith("---"):
        # No front-matter, treat entire content as prompt
        return {}, content

    # Find the closing ---
    lines = content.split("\n")
    end_idx = None

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        # No closing delimiter, treat as no front-matter
        return {}, content

    # Extract TOML and body
    toml_lines = lines[1:end_idx]
    toml_str = "\n".join(toml_lines)
    body_lines = lines[end_idx + 1 :]
    body = "\n".join(body_lines)

    # Parse TOML
    try:
        meta = tomllib.loads(toml_str)
    except tomllib.TOMLDecodeError as e:
        raise AgentDefinitionError(f"Invalid TOML in front-matter: {e}") from e

    return meta, body


def parse_agent_spec(
    meta: dict[str, Any],
    prompt_body: str,
    source_path: Path | None = None,
) -> AgentSpec:
    """Parse agent specification from TOML metadata and prompt body.

    Args:
        meta: Parsed TOML metadata dictionary
        prompt_body: The prompt template text
        source_path: Optional source file path

    Returns:
        Parsed AgentSpec

    Raises:
        AgentDefinitionError: If definition is invalid
    """
    # Validate required sections
    if "agent" not in meta:
        raise AgentDefinitionError.missing_section("agent")

    agent_config = meta["agent"]

    # Validate required fields
    if "model" not in agent_config:
        raise AgentDefinitionError.missing_field("agent", "model", '"openai:gpt-5"')

    # Validate prompt body
    prompt_body = prompt_body.strip()
    if not prompt_body:
        raise AgentDefinitionError.no_prompt_body()

    # Extract placeholders
    placeholders = set(_PLACEHOLDER_PATTERN.findall(prompt_body))

    # Also check instructions for placeholders
    instructions = agent_config.get("instructions")
    if instructions:
        placeholders |= set(_PLACEHOLDER_PATTERN.findall(instructions))

    if not placeholders:
        raise AgentDefinitionError.no_placeholders()

    # Parse output_type
    output_type_config = agent_config.get("output_type", {})
    output_fields = _parse_output_fields(output_type_config)

    if not output_fields:
        raise AgentDefinitionError.no_output_fields()

    # Parse input_type
    input_type_config = agent_config.get("input_type", {})
    input_definitions = _parse_input_definitions(input_type_config)

    # Extract settings
    settings = agent_config.get("settings", {})

    return AgentSpec(
        model=agent_config["model"],
        prompt_template=prompt_body,
        output_fields=tuple(output_fields),
        name=agent_config.get("name"),
        instructions=instructions,
        retries=agent_config.get("retries", 2),
        settings=dict(settings) if settings else {},
        input_definitions=tuple(input_definitions),
        output_type_name=output_type_config.get("name", "AgentOutput"),
        output_type_description=output_type_config.get("description"),
        source_path=source_path,
    )


def _parse_output_fields(config: dict[str, Any]) -> list[FieldDefinition]:
    """Parse output field definitions from output_type config.

    Supports both inline and section-based definitions:
    - Inline: { type = "str", description = "..." }
    - Section: [agent.output_type.field_name] with type, description, etc.

    Args:
        config: The output_type configuration dict

    Returns:
        List of FieldDefinition objects, with 'reasoning' first if present
    """
    fields: list[FieldDefinition] = []

    for name, value in config.items():
        # Skip metadata keys
        if name in OUTPUT_TYPE_METADATA_KEYS:
            continue

        # Must be a dict (field definition)
        if not isinstance(value, dict):
            continue

        field_def = _parse_single_field(name, value)
        fields.append(field_def)

    # Sort: reasoning first, then alphabetically
    fields.sort(key=lambda f: (0 if f.name == "reasoning" else 1, f.name))

    return fields


def _parse_single_field(name: str, config: dict[str, Any]) -> FieldDefinition:
    """Parse a single field definition.

    Args:
        name: Field name
        config: Field configuration dict

    Returns:
        FieldDefinition

    Raises:
        AgentDefinitionError: If type is unsupported
    """
    type_str = config.get("type", "bool")

    # Validate type
    if type_str not in SUPPORTED_TYPES:
        raise AgentDefinitionError.unsupported_type(
            name, type_str, list(SUPPORTED_TYPES)
        )

    # Parse enum if present
    enum_values = config.get("enum")
    if enum_values is not None:
        enum_values = tuple(enum_values)

    return FieldDefinition(
        name=name,
        type=type_str,
        description=config.get("description"),
        optional=config.get("optional", False),
        enum=enum_values,
        min_length=config.get("min_length"),
        max_length=config.get("max_length"),
        pattern=config.get("pattern"),
        ge=config.get("ge"),
        le=config.get("le"),
        gt=config.get("gt"),
        lt=config.get("lt"),
        min_items=config.get("min_items"),
        max_items=config.get("max_items"),
    )


def _parse_input_definitions(config: dict[str, Any]) -> list[InputDefinition]:
    """Parse input definitions from input_type config.

    Supports both inline and section-based definitions.

    Args:
        config: The input_type configuration dict

    Returns:
        List of InputDefinition objects
    """
    inputs: list[InputDefinition] = []

    for name, value in config.items():
        if isinstance(value, dict):
            inputs.append(
                InputDefinition(
                    name=name,
                    type=value.get("type", "str"),
                    description=value.get("description"),
                    optional=value.get("optional", False),
                )
            )
        elif isinstance(value, str):
            # Simple string value means just a type
            inputs.append(InputDefinition(name=name, type=value))
        else:
            raise AgentDefinitionError.invalid_input_type(name, type(value).__name__)

    return inputs
