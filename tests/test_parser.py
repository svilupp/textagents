"""Tests for the parser module."""

from __future__ import annotations

from typing import Any

import pytest

from textagents.errors import AgentDefinitionError
from textagents.parser import (
    FieldDefinition,
    InputDefinition,
    parse_agent_spec,
)


class TestParseAgentSpec:
    """Tests for parse_agent_spec function."""

    def test_minimal_valid_spec(self, minimal_agent_meta: dict[str, Any]) -> None:
        """Test parsing a minimal valid agent specification."""
        spec = parse_agent_spec(minimal_agent_meta, "Evaluate: {input}")

        assert spec.model == "openai:gpt-5"
        assert spec.prompt_template == "Evaluate: {input}"
        assert len(spec.output_fields) == 2
        assert spec.placeholders == {"input"}

    def test_reasoning_field_first(self, minimal_agent_meta: dict[str, Any]) -> None:
        """Test that reasoning field is sorted first."""
        spec = parse_agent_spec(minimal_agent_meta, "Test: {input}")

        # reasoning should be first
        assert spec.output_fields[0].name == "reasoning"
        assert spec.output_fields[1].name == "is_valid"

    def test_bool_default_type(self, minimal_agent_meta: dict[str, Any]) -> None:
        """Test that fields without type default to bool."""
        spec = parse_agent_spec(minimal_agent_meta, "Test: {input}")

        # is_valid has no type, should default to bool
        is_valid_field = next(f for f in spec.output_fields if f.name == "is_valid")
        assert is_valid_field.type == "bool"

    def test_explicit_type(self, minimal_agent_meta: dict[str, Any]) -> None:
        """Test that explicit type is respected."""
        spec = parse_agent_spec(minimal_agent_meta, "Test: {input}")

        # reasoning has explicit str type
        reasoning_field = next(f for f in spec.output_fields if f.name == "reasoning")
        assert reasoning_field.type == "str"

    def test_field_description(self, minimal_agent_meta: dict[str, Any]) -> None:
        """Test that field descriptions are parsed."""
        spec = parse_agent_spec(minimal_agent_meta, "Test: {input}")

        reasoning_field = next(f for f in spec.output_fields if f.name == "reasoning")
        assert reasoning_field.description == "Brief reasoning."

    def test_missing_agent_section(self) -> None:
        """Test error when [agent] section is missing."""
        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec({}, "Test: {input}")

        assert "Missing required section '[agent]'" in str(exc_info.value)

    def test_missing_model(self) -> None:
        """Test error when model is missing."""
        meta = {"agent": {"output_type": {"field": {}}}}

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "Test: {input}")

        assert "Missing required field 'model'" in str(exc_info.value)

    def test_no_output_fields(self) -> None:
        """Test error when no output fields are defined."""
        meta = {"agent": {"model": "openai:gpt-5"}}

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "Test: {input}")

        assert "No fields defined" in str(exc_info.value)

    def test_empty_prompt_body(self) -> None:
        """Test error when prompt body is empty."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {"field": {}},
            }
        }

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "")

        assert "No prompt body" in str(exc_info.value)

    def test_no_placeholders(self) -> None:
        """Test error when prompt has no placeholders."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {"field": {}},
            }
        }

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "Static prompt without placeholders")

        assert "no {placeholders}" in str(exc_info.value)

    def test_unsupported_type(self) -> None:
        """Test error for unsupported field type."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {"field": {"type": "dict[str,str]"}},
            }
        }

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "Test: {input}")

        assert "Unsupported type" in str(exc_info.value)

    def test_enum_values(self) -> None:
        """Test parsing enum field."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {
                    "severity": {
                        "type": "int",
                        "enum": [1, 2, 3, 4, 5],
                    }
                },
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")
        field = spec.output_fields[0]

        assert field.enum == (1, 2, 3, 4, 5)

    def test_invalid_input_type_raises(self) -> None:
        """Input definitions must be string or table."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {"result": {"type": "bool"}},
                "input_type": {"bad": 123},
            }
        }

        with pytest.raises(AgentDefinitionError) as exc_info:
            parse_agent_spec(meta, "Prompt: {bad}")

        assert "invalid input_type" in str(exc_info.value).lower()

    def test_field_constraints(self) -> None:
        """Test parsing field constraints."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {
                    "text": {
                        "type": "str",
                        "min_length": 10,
                        "max_length": 500,
                        "pattern": "^[A-Z]",
                    },
                    "score": {
                        "type": "float",
                        "ge": 0.0,
                        "le": 1.0,
                    },
                },
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")

        text_field = next(f for f in spec.output_fields if f.name == "text")
        assert text_field.min_length == 10
        assert text_field.max_length == 500
        assert text_field.pattern == "^[A-Z]"

        score_field = next(f for f in spec.output_fields if f.name == "score")
        assert score_field.ge == 0.0
        assert score_field.le == 1.0

    def test_optional_field(self) -> None:
        """Test parsing optional field."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {
                    "required_field": {},
                    "optional_field": {"optional": True},
                },
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")

        required = next(f for f in spec.output_fields if f.name == "required_field")
        optional = next(f for f in spec.output_fields if f.name == "optional_field")

        assert required.optional is False
        assert optional.optional is True

    def test_input_type_parsing(self) -> None:
        """Test parsing input_type definitions."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "input_type": {
                    "required_input": {"type": "str", "description": "Required"},
                    "optional_input": {"type": "int", "optional": True},
                },
                "output_type": {"field": {}},
            }
        }

        spec = parse_agent_spec(meta, "Test: {required_input}")

        assert len(spec.input_definitions) == 2

        required = next(i for i in spec.input_definitions if i.name == "required_input")
        optional = next(i for i in spec.input_definitions if i.name == "optional_input")

        assert required.type == "str"
        assert required.optional is False
        assert optional.type == "int"
        assert optional.optional is True

    def test_instructions_placeholders(self) -> None:
        """Test that placeholders in instructions are detected."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "instructions": "Today is {CURRENT_DATE}. User: {user_name}",
                "output_type": {"field": {}},
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")

        assert "CURRENT_DATE" in spec.instruction_placeholders
        assert "user_name" in spec.instruction_placeholders
        assert spec.all_placeholders == {"input", "CURRENT_DATE", "user_name"}

    def test_output_type_metadata(self) -> None:
        """Test parsing output_type name and description."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "output_type": {
                    "name": "CustomOutput",
                    "description": "A custom output type",
                    "field": {},
                },
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")

        assert spec.output_type_name == "CustomOutput"
        assert spec.output_type_description == "A custom output type"

    def test_settings_parsing(self) -> None:
        """Test parsing model settings."""
        meta = {
            "agent": {
                "model": "openai:gpt-5",
                "settings": {"temperature": 0, "max_tokens": 500},
                "output_type": {"field": {}},
            }
        }

        spec = parse_agent_spec(meta, "Test: {input}")

        assert spec.settings == {"temperature": 0, "max_tokens": 500}


class TestFieldDefinition:
    """Tests for FieldDefinition dataclass."""

    def test_default_values(self) -> None:
        """Test default values for FieldDefinition."""
        field = FieldDefinition(name="test")

        assert field.type == "bool"
        assert field.description is None
        assert field.optional is False
        assert field.enum is None

    def test_all_values(self) -> None:
        """Test FieldDefinition with all values set."""
        field = FieldDefinition(
            name="test",
            type="str",
            description="A test field",
            optional=True,
            enum=("a", "b", "c"),
            min_length=1,
            max_length=100,
        )

        assert field.name == "test"
        assert field.type == "str"
        assert field.description == "A test field"
        assert field.optional is True
        assert field.enum == ("a", "b", "c")
        assert field.min_length == 1
        assert field.max_length == 100


class TestInputDefinition:
    """Tests for InputDefinition dataclass."""

    def test_default_values(self) -> None:
        """Test default values for InputDefinition."""
        inp = InputDefinition(name="test")

        assert inp.type == "str"
        assert inp.description is None
        assert inp.optional is False

    def test_all_values(self) -> None:
        """Test InputDefinition with all values set."""
        inp = InputDefinition(
            name="test",
            type="int",
            description="A test input",
            optional=True,
        )

        assert inp.name == "test"
        assert inp.type == "int"
        assert inp.description == "A test input"
        assert inp.optional is True
