"""Tests for the validator_builder module."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from textagents.parser import AgentSpec, FieldDefinition
from textagents.validator_builder import _validate_field


class TestValidateField:
    """Tests for _validate_field function."""

    def test_enum_valid(self) -> None:
        """Test enum validation with valid value."""
        field_def = FieldDefinition(
            name="severity",
            type="int",
            enum=(1, 2, 3, 4, 5),
        )

        errors = _validate_field(field_def, 3)
        assert errors == []

    def test_enum_invalid(self) -> None:
        """Test enum validation with invalid value."""
        field_def = FieldDefinition(
            name="severity",
            type="int",
            enum=(1, 2, 3, 4, 5),
        )

        errors = _validate_field(field_def, 10)
        assert len(errors) == 1
        assert "'severity' must be one of [1, 2, 3, 4, 5]" in errors[0]
        assert "got 10" in errors[0]

    def test_enum_string_valid(self) -> None:
        """Test enum validation with string values."""
        field_def = FieldDefinition(
            name="status",
            type="str",
            enum=("pending", "approved", "rejected"),
        )

        errors = _validate_field(field_def, "approved")
        assert errors == []

    def test_enum_string_invalid(self) -> None:
        """Test enum validation with invalid string value."""
        field_def = FieldDefinition(
            name="status",
            type="str",
            enum=("pending", "approved", "rejected"),
        )

        errors = _validate_field(field_def, "unknown")
        assert len(errors) == 1
        assert "'status' must be one of" in errors[0]

    def test_max_length_valid(self) -> None:
        """Test max_length validation with valid string."""
        field_def = FieldDefinition(
            name="text",
            type="str",
            max_length=10,
        )

        errors = _validate_field(field_def, "short")
        assert errors == []

    def test_max_length_invalid(self) -> None:
        """Test max_length validation with too long string."""
        field_def = FieldDefinition(
            name="text",
            type="str",
            max_length=10,
        )

        errors = _validate_field(field_def, "this is way too long")
        assert len(errors) == 1
        assert "'text' exceeds max_length of 10" in errors[0]
        assert "got 20 chars" in errors[0]

    def test_min_length_valid(self) -> None:
        """Test min_length validation with valid string."""
        field_def = FieldDefinition(
            name="text",
            type="str",
            min_length=5,
        )

        errors = _validate_field(field_def, "hello world")
        assert errors == []

    def test_min_length_invalid(self) -> None:
        """Test min_length validation with too short string."""
        field_def = FieldDefinition(
            name="text",
            type="str",
            min_length=5,
        )

        errors = _validate_field(field_def, "hi")
        assert len(errors) == 1
        assert "'text' below min_length of 5" in errors[0]
        assert "got 2 chars" in errors[0]

    def test_ge_valid(self) -> None:
        """Test ge (greater than or equal) validation with valid value."""
        field_def = FieldDefinition(
            name="score",
            type="float",
            ge=0.0,
        )

        errors = _validate_field(field_def, 0.5)
        assert errors == []

        # Boundary value
        errors = _validate_field(field_def, 0.0)
        assert errors == []

    def test_ge_invalid(self) -> None:
        """Test ge validation with invalid value."""
        field_def = FieldDefinition(
            name="score",
            type="float",
            ge=0.0,
        )

        errors = _validate_field(field_def, -0.1)
        assert len(errors) == 1
        assert "'score' must be >= 0.0" in errors[0]

    def test_le_valid(self) -> None:
        """Test le (less than or equal) validation with valid value."""
        field_def = FieldDefinition(
            name="score",
            type="float",
            le=1.0,
        )

        errors = _validate_field(field_def, 0.5)
        assert errors == []

        # Boundary value
        errors = _validate_field(field_def, 1.0)
        assert errors == []

    def test_le_invalid(self) -> None:
        """Test le validation with invalid value."""
        field_def = FieldDefinition(
            name="score",
            type="float",
            le=1.0,
        )

        errors = _validate_field(field_def, 1.1)
        assert len(errors) == 1
        assert "'score' must be <= 1.0" in errors[0]

    def test_gt_valid(self) -> None:
        """Test gt (greater than) validation with valid value."""
        field_def = FieldDefinition(
            name="value",
            type="int",
            gt=0,
        )

        errors = _validate_field(field_def, 1)
        assert errors == []

    def test_gt_invalid(self) -> None:
        """Test gt validation with invalid value (boundary)."""
        field_def = FieldDefinition(
            name="value",
            type="int",
            gt=0,
        )

        # Boundary value should fail (must be greater than, not equal)
        errors = _validate_field(field_def, 0)
        assert len(errors) == 1
        assert "'value' must be > 0" in errors[0]

    def test_lt_valid(self) -> None:
        """Test lt (less than) validation with valid value."""
        field_def = FieldDefinition(
            name="value",
            type="int",
            lt=100,
        )

        errors = _validate_field(field_def, 99)
        assert errors == []

    def test_lt_invalid(self) -> None:
        """Test lt validation with invalid value (boundary)."""
        field_def = FieldDefinition(
            name="value",
            type="int",
            lt=100,
        )

        # Boundary value should fail (must be less than, not equal)
        errors = _validate_field(field_def, 100)
        assert len(errors) == 1
        assert "'value' must be < 100" in errors[0]

    def test_max_items_valid(self) -> None:
        """Test max_items validation with valid list."""
        field_def = FieldDefinition(
            name="tags",
            type="list[str]",
            max_items=5,
        )

        errors = _validate_field(field_def, ["a", "b", "c"])
        assert errors == []

    def test_max_items_invalid(self) -> None:
        """Test max_items validation with too many items."""
        field_def = FieldDefinition(
            name="tags",
            type="list[str]",
            max_items=3,
        )

        errors = _validate_field(field_def, ["a", "b", "c", "d", "e"])
        assert len(errors) == 1
        assert "'tags' exceeds max_items of 3" in errors[0]
        assert "got 5 items" in errors[0]

    def test_min_items_valid(self) -> None:
        """Test min_items validation with valid list."""
        field_def = FieldDefinition(
            name="tags",
            type="list[str]",
            min_items=2,
        )

        errors = _validate_field(field_def, ["a", "b", "c"])
        assert errors == []

    def test_min_items_invalid(self) -> None:
        """Test min_items validation with too few items."""
        field_def = FieldDefinition(
            name="tags",
            type="list[str]",
            min_items=3,
        )

        errors = _validate_field(field_def, ["a"])
        assert len(errors) == 1
        assert "'tags' below min_items of 3" in errors[0]
        assert "got 1 items" in errors[0]

    def test_combined_constraints(self) -> None:
        """Test multiple constraints on the same field."""
        field_def = FieldDefinition(
            name="score",
            type="float",
            ge=0.0,
            le=1.0,
        )

        # Valid value
        errors = _validate_field(field_def, 0.5)
        assert errors == []

        # Below minimum
        errors = _validate_field(field_def, -0.5)
        assert len(errors) == 1

        # Above maximum
        errors = _validate_field(field_def, 1.5)
        assert len(errors) == 1

    def test_no_constraints(self) -> None:
        """Test field with no constraints."""
        field_def = FieldDefinition(
            name="text",
            type="str",
        )

        errors = _validate_field(field_def, "any value works")
        assert errors == []

    def test_bool_not_treated_as_numeric(self) -> None:
        """Test that bool values are not validated as numeric."""
        field_def = FieldDefinition(
            name="flag",
            type="bool",
            ge=0,  # This shouldn't apply to bool
        )

        # True is 1 in Python, but should not trigger numeric validation
        errors = _validate_field(field_def, True)
        assert errors == []

        errors = _validate_field(field_def, False)
        assert errors == []

    def test_int_with_constraints(self) -> None:
        """Test integer field with numeric constraints."""
        field_def = FieldDefinition(
            name="count",
            type="int",
            ge=1,
            le=10,
        )

        # Valid value
        errors = _validate_field(field_def, 5)
        assert errors == []

        # Below minimum
        errors = _validate_field(field_def, 0)
        assert len(errors) == 1

        # Above maximum
        errors = _validate_field(field_def, 11)
        assert len(errors) == 1

    def test_empty_list(self) -> None:
        """Test empty list with min_items constraint."""
        field_def = FieldDefinition(
            name="items",
            type="list[str]",
            min_items=1,
        )

        errors = _validate_field(field_def, [])
        assert len(errors) == 1
        assert "got 0 items" in errors[0]

    def test_empty_string(self) -> None:
        """Test empty string with min_length constraint."""
        field_def = FieldDefinition(
            name="text",
            type="str",
            min_length=1,
        )

        errors = _validate_field(field_def, "")
        assert len(errors) == 1
        assert "got 0 chars" in errors[0]


class TestAddOutputValidator:
    """Tests for add_output_validator integration."""

    def test_validator_added_to_agent(self) -> None:
        """Test that validator is added to agent."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="is_valid"),),
        )

        # Create a mock agent with output_validator decorator
        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Verify validator was registered
        assert validator_func is not None

    def test_validator_passes_valid_output(self) -> None:
        """Test that validator passes valid output through."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="is_valid"),
                FieldDefinition(name="score", type="int", ge=1, le=5),
            ),
        )

        # Create a mock agent and capture the validator
        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Create a valid output model
        class TestOutput(BaseModel):
            is_valid: bool
            score: int

        output = TestOutput(is_valid=True, score=3)
        mock_ctx = MagicMock()

        # Should return output unchanged
        result = validator_func(mock_ctx, output)
        assert result == output

    def test_validator_raises_on_required_none(self) -> None:
        """Test that validator raises ModelRetry for required None field."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent, ModelRetry

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(FieldDefinition(name="required_field", type="str"),),
        )

        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Create output with None for required field
        class TestOutput(BaseModel):
            required_field: str | None

        output = TestOutput(required_field=None)
        mock_ctx = MagicMock()

        with pytest.raises(ModelRetry) as exc_info:
            validator_func(mock_ctx, output)

        assert "'required_field' is required but was None" in str(exc_info.value)

    def test_validator_skips_optional_none(self) -> None:
        """Test that validator skips validation for optional None fields."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="optional_field", type="str", optional=True),
            ),
        )

        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Create output with None for optional field
        class TestOutput(BaseModel):
            optional_field: str | None

        output = TestOutput(optional_field=None)
        mock_ctx = MagicMock()

        # Should pass without error
        result = validator_func(mock_ctx, output)
        assert result == output

    def test_validator_raises_on_constraint_violation(self) -> None:
        """Test that validator raises ModelRetry for constraint violations."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent, ModelRetry

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="score", type="float", ge=0.0, le=1.0),
            ),
        )

        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Create output with invalid score
        class TestOutput(BaseModel):
            score: float

        output = TestOutput(score=1.5)
        mock_ctx = MagicMock()

        with pytest.raises(ModelRetry) as exc_info:
            validator_func(mock_ctx, output)

        assert "'score' must be <= 1.0" in str(exc_info.value)

    def test_validator_multiple_errors(self) -> None:
        """Test that validator collects multiple errors."""
        from unittest.mock import MagicMock

        from pydantic_ai import Agent, ModelRetry

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Test: {input}",
            output_fields=(
                FieldDefinition(name="text", type="str", min_length=10),
                FieldDefinition(name="score", type="float", ge=0.0, le=1.0),
            ),
        )

        mock_agent = MagicMock(spec=Agent)
        validator_func = None

        def capture_validator(func):
            nonlocal validator_func
            validator_func = func
            return func

        mock_agent.output_validator = capture_validator

        from textagents.validator_builder import add_output_validator

        add_output_validator(mock_agent, spec)

        # Create output with multiple violations
        class TestOutput(BaseModel):
            text: str
            score: float

        output = TestOutput(text="short", score=2.0)
        mock_ctx = MagicMock()

        with pytest.raises(ModelRetry) as exc_info:
            validator_func(mock_ctx, output)

        error_msg = str(exc_info.value)
        assert "'text' below min_length" in error_msg
        assert "'score' must be <= 1.0" in error_msg
