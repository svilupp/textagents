"""Tests for the model_builder module."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from textagents.model_builder import build_output_model
from textagents.parser import AgentSpec, FieldDefinition


class TestBuildOutputModel:
    """Tests for build_output_model function."""

    def test_minimal_model(self) -> None:
        """Test building a minimal model with one bool field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="is_valid", description="Whether valid"),
            ),
        )

        model = build_output_model(spec)

        assert issubclass(model, BaseModel)
        assert "is_valid" in model.model_fields

        # Test instantiation
        instance = model(is_valid=True)
        assert instance.is_valid is True

    def test_reasoning_first(self) -> None:
        """Test that reasoning field is first in the model."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="is_valid"),
                FieldDefinition(name="reasoning", type="str"),
            ),
        )

        model = build_output_model(spec)

        # Get field order
        field_names = list(model.model_fields.keys())
        assert field_names[0] == "reasoning"
        assert field_names[1] == "is_valid"

    def test_string_type(self) -> None:
        """Test building model with string field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="text", type="str", description="Text field"),
            ),
        )

        model = build_output_model(spec)
        instance = model(text="hello")

        assert instance.text == "hello"

    def test_int_type(self) -> None:
        """Test building model with int field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="count", type="int"),),
        )

        model = build_output_model(spec)
        instance = model(count=42)

        assert instance.count == 42

    def test_float_type(self) -> None:
        """Test building model with float field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="score", type="float"),),
        )

        model = build_output_model(spec)
        instance = model(score=0.95)

        assert instance.score == 0.95

    def test_list_str_type(self) -> None:
        """Test building model with list[str] field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="tags", type="list[str]"),),
        )

        model = build_output_model(spec)
        instance = model(tags=["a", "b", "c"])

        assert instance.tags == ["a", "b", "c"]

    def test_enum_field(self) -> None:
        """Test building model with enum constraint."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(
                    name="severity",
                    type="int",
                    enum=(1, 2, 3, 4, 5),
                ),
            ),
        )

        model = build_output_model(spec)

        # Valid values work
        instance = model(severity=3)
        assert instance.severity == 3

        # Invalid values should fail
        with pytest.raises(ValidationError):
            model(severity=10)

    def test_optional_field(self) -> None:
        """Test building model with optional field."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="required", type="str"),
                FieldDefinition(name="optional", type="str", optional=True),
            ),
        )

        model = build_output_model(spec)

        # Can create without optional field
        instance = model(required="hello")
        assert instance.required == "hello"
        assert instance.optional is None

        # Can create with optional field
        instance = model(required="hello", optional="world")
        assert instance.optional == "world"

    def test_max_length_constraint(self) -> None:
        """Test building model with max_length constraint."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="text", type="str", max_length=10),),
        )

        model = build_output_model(spec)

        # Valid length works
        instance = model(text="short")
        assert instance.text == "short"

        # Too long should fail
        with pytest.raises(ValidationError):
            model(text="this is way too long")

    def test_numeric_constraints(self) -> None:
        """Test building model with numeric constraints."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="score", type="float", ge=0.0, le=1.0),
            ),
        )

        model = build_output_model(spec)

        # Valid values work
        instance = model(score=0.5)
        assert instance.score == 0.5

        # Out of range should fail
        with pytest.raises(ValidationError):
            model(score=1.5)

        with pytest.raises(ValidationError):
            model(score=-0.1)

    def test_model_name(self) -> None:
        """Test that model uses output_type_name."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="field"),),
            output_type_name="CustomOutput",
        )

        model = build_output_model(spec)
        assert model.__name__ == "CustomOutput"

    def test_model_description(self) -> None:
        """Test that model uses output_type_description."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="field"),),
            output_type_description="A custom description",
        )

        model = build_output_model(spec)
        assert model.__doc__ == "A custom description"

    def test_field_description_in_schema(self) -> None:
        """Test that field descriptions appear in JSON schema."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(
                    name="field",
                    description="This is a test field",
                ),
            ),
        )

        model = build_output_model(spec)
        schema = model.model_json_schema()

        assert schema["properties"]["field"]["description"] == "This is a test field"
