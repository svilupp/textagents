"""Tests for the errors module."""

from __future__ import annotations

from textagents.errors import (
    AgentDefinitionError,
    InputTypeError,
    MissingInputError,
    OutputValidationError,
    TemplateError,
    TextAgentsError,
)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_errors_inherit_from_base(self) -> None:
        """Test that all custom errors inherit from TextAgentsError."""
        errors = [
            AgentDefinitionError,
            MissingInputError,
            InputTypeError,
            OutputValidationError,
            TemplateError,
        ]

        for error_class in errors:
            assert issubclass(error_class, TextAgentsError)

    def test_catch_all_with_base(self) -> None:
        """Test that base class catches all custom errors."""
        errors = [
            AgentDefinitionError("test"),
            MissingInputError("test"),
            InputTypeError("test"),
            OutputValidationError("test"),
            TemplateError("test"),
        ]

        for error in errors:
            try:
                raise error
            except TextAgentsError:
                pass  # Should catch all


class TestAgentDefinitionError:
    """Tests for AgentDefinitionError."""

    def test_missing_section(self) -> None:
        """Test missing_section class method."""
        error = AgentDefinitionError.missing_section("agent")

        assert "Missing required section" in str(error)
        assert "[agent]" in str(error)

    def test_missing_field(self) -> None:
        """Test missing_field class method."""
        error = AgentDefinitionError.missing_field("agent", "model", '"openai:gpt-5"')

        assert "Missing required field 'model'" in str(error)
        assert "[agent]" in str(error)

    def test_no_output_fields(self) -> None:
        """Test no_output_fields class method."""
        error = AgentDefinitionError.no_output_fields()

        assert "No fields defined" in str(error)
        assert "agent.output_type" in str(error)

    def test_unsupported_type(self) -> None:
        """Test unsupported_type class method."""
        error = AgentDefinitionError.unsupported_type(
            "field", "dict[str,str]", ["bool", "str", "int"]
        )

        assert "Unsupported type" in str(error)
        assert "dict[str,str]" in str(error)
        assert "field" in str(error)

    def test_no_prompt_body(self) -> None:
        """Test no_prompt_body class method."""
        error = AgentDefinitionError.no_prompt_body()

        assert "No prompt body" in str(error)

    def test_no_placeholders(self) -> None:
        """Test no_placeholders class method."""
        error = AgentDefinitionError.no_placeholders()

        assert "no {placeholders}" in str(error)


class TestMissingInputError:
    """Tests for MissingInputError."""

    def test_missing_required(self) -> None:
        """Test missing_required class method."""
        error = MissingInputError.missing_required(
            ["input1", "input2"],
            ["provided1"],
            ["input1", "input2", "provided1"],
        )

        assert "Missing required input" in str(error)
        assert "input1" in str(error)
        assert "input2" in str(error)
        assert "Provided:" in str(error)

    def test_file_not_found(self) -> None:
        """Test file_not_found class method."""
        error = MissingInputError.file_not_found("input", "/path/to/file.txt")

        assert "Input file not found" in str(error)
        assert "input" in str(error)
        assert "/path/to/file.txt" in str(error)


class TestInputTypeError:
    """Tests for InputTypeError."""

    def test_cannot_coerce(self) -> None:
        """Test cannot_coerce class method."""
        error = InputTypeError.cannot_coerce("count", "not a number", "int")

        assert "Cannot convert" in str(error)
        assert "count" in str(error)
        assert "int" in str(error)


class TestOutputValidationError:
    """Tests for OutputValidationError."""

    def test_with_cause(self) -> None:
        """Test OutputValidationError with cause."""
        cause = ValueError("original error")
        error = OutputValidationError("Validation failed", cause)

        assert error.cause is cause
        assert "Validation failed" in str(error)


class TestTemplateError:
    """Tests for TemplateError."""

    def test_missing_placeholder(self) -> None:
        """Test missing_placeholder class method."""
        error = TemplateError.missing_placeholder(
            "missing", ["available1", "available2"]
        )

        assert "missing" in str(error)
        assert "available1" in str(error)
        assert "CURRENT_DATE" in str(error)  # Should mention magic vars
