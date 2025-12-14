"""Integration tests for TextAgent using PydanticAI's TestModel.

These tests verify the full flow without making real API calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textagents import load_agent
from textagents.agent import create_text_agent
from textagents.parser import AgentSpec, FieldDefinition


@pytest.fixture
def mock_pydantic_agent():
    """Mock PydanticAI Agent to avoid needing API keys during load."""
    with patch("textagents.agent.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent_class


class TestTextAgentWithTestModel:
    """Integration tests using PydanticAI's TestModel."""

    def test_create_agent_from_spec(self, mock_pydantic_agent: MagicMock) -> None:
        """Test creating a TextAgent from spec."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Evaluate: {input}",
            output_fields=(
                FieldDefinition(name="reasoning", type="str"),
                FieldDefinition(name="is_valid", type="bool"),
            ),
        )

        text_agent = create_text_agent(spec)

        assert text_agent.model == "openai:gpt-5"
        assert text_agent.output_model is not None
        assert "reasoning" in text_agent.output_model.model_fields
        assert "is_valid" in text_agent.output_model.model_fields

    def test_output_model_fields_correct(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that output model has correct field types."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file)

        # Check the output model structure
        model = text_agent.output_model
        fields = model.model_fields

        assert "reasoning" in fields
        assert "is_valid" in fields

    def test_agent_properties(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test TextAgent properties."""
        agent_file = tmp_path / "test_agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file)

        assert text_agent.model == "openai:gpt-5"
        assert text_agent.name == "test_agent"
        assert "input" in text_agent.input_names
        assert "input" in text_agent.required_inputs

    def test_full_agent_properties(
        self, tmp_path: Path, full_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test full-featured agent properties."""
        agent_file = tmp_path / "full_agent.txt"
        agent_file.write_text(full_agent_content)

        text_agent = load_agent(agent_file)

        # Verify output model structure
        assert "reasoning" in text_agent.output_model.model_fields
        assert "is_valid" in text_agent.output_model.model_fields
        assert "score" in text_agent.output_model.model_fields
        assert "confidence" in text_agent.output_model.model_fields


class TestAgentInputValidation:
    """Test input validation in the agent."""

    def test_required_inputs_property(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that required_inputs correctly identifies required fields."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.input_type]
required_field = { type = "str" }
optional_field = { type = "str", optional = true }

[agent.output_type.result]
---
Test: {required_field} {optional_field}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        assert "required_field" in text_agent.required_inputs
        assert "optional_field" not in text_agent.required_inputs

    def test_input_names_includes_all(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that input_names includes all inputs."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.input_type]
defined_input = { type = "str" }

[agent.output_type.result]
---
Test: {defined_input} {placeholder_input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # Both defined and placeholder inputs should be in input_names
        assert "defined_input" in text_agent.input_names
        assert "placeholder_input" in text_agent.input_names


class TestAgentMagicVariables:
    """Test magic variable handling."""

    def test_magic_variables_not_required(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that CURRENT_DATE etc are not in required_inputs."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.output_type.result]
---
Today is {CURRENT_DATE}. Input: {user_input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # CURRENT_DATE should NOT be in required_inputs (it's magic)
        assert "CURRENT_DATE" not in text_agent.required_inputs
        # user_input should be required
        assert "user_input" in text_agent.required_inputs

    def test_all_magic_variables_excluded(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test all magic variables are excluded from required."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.output_type.result]
---
Date: {CURRENT_DATE} Time: {CURRENT_TIME} DateTime: {CURRENT_DATETIME}
Input: {input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # Magic variables should not be required
        assert "CURRENT_DATE" not in text_agent.required_inputs
        assert "CURRENT_TIME" not in text_agent.required_inputs
        assert "CURRENT_DATETIME" not in text_agent.required_inputs
        # Regular input should be required
        assert "input" in text_agent.required_inputs


class TestOutputModelGeneration:
    """Test dynamic Pydantic model generation."""

    def test_bool_field_default(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that fields without type default to bool."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.output_type.is_valid]
description = "Just a description, no type"
---
Test: {input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # The field should be bool type
        field_info = text_agent.output_model.model_fields["is_valid"]
        assert field_info.annotation is bool

    def test_enum_field_creates_literal(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that enum fields create Literal type."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.output_type.severity]
type = "int"
enum = [1, 2, 3, 4, 5]
---
Test: {input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # Create an instance to verify Literal type works
        model = text_agent.output_model
        instance = model(severity=3)
        assert instance.severity == 3

    def test_optional_field_allows_none(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that optional fields accept None."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.output_type.notes]
type = "str"
optional = true
---
Test: {input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        # Create an instance with None
        model = text_agent.output_model
        instance = model(notes=None)
        assert instance.notes is None
