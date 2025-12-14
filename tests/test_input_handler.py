"""Tests for the input_handler module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from textagents.errors import InputTypeError, MissingInputError, TemplateError
from textagents.input_handler import (
    MAGIC_VARIABLES,
    interpolate_template,
    process_inputs,
)
from textagents.parser import AgentSpec, FieldDefinition, InputDefinition


class TestProcessInputs:
    """Tests for process_inputs function."""

    def test_basic_inputs(self) -> None:
        """Test processing basic string inputs."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({"input": "hello"}, spec)

        assert result["input"] == "hello"

    def test_file_input(self, tmp_path: Path) -> None:
        """Test loading input from file with @ syntax."""
        # Create a test file
        test_file = tmp_path / "input.txt"
        test_file.write_text("content from file")

        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({"input": f"@{test_file}"}, spec)

        assert result["input"] == "content from file"

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
        )

        with pytest.raises(MissingInputError) as exc_info:
            process_inputs({"input": "@nonexistent.txt"}, spec)

        assert "Input file not found" in str(exc_info.value)

    def test_magic_variable_current_date(self) -> None:
        """Test CURRENT_DATE magic variable."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Date: {CURRENT_DATE}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({}, spec)

        # Should match YYYY-MM-DD format
        assert re.match(r"\d{4}-\d{2}-\d{2}", result["CURRENT_DATE"])

    def test_magic_variable_current_time(self) -> None:
        """Test CURRENT_TIME magic variable."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Time: {CURRENT_TIME}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({}, spec)

        # Should match HH:MM:SS format
        assert re.match(r"\d{2}:\d{2}:\d{2}", result["CURRENT_TIME"])

    def test_magic_variable_current_datetime(self) -> None:
        """Test CURRENT_DATETIME magic variable."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="DateTime: {CURRENT_DATETIME}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({}, spec)

        # Should match YYYY-MM-DD HH:MM:SS format
        assert re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result["CURRENT_DATETIME"]
        )

    def test_magic_variable_not_overwritten(self) -> None:
        """Test that explicit input overrides magic variable."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Date: {CURRENT_DATE}",
            output_fields=(FieldDefinition(name="field"),),
        )

        result = process_inputs({"CURRENT_DATE": "2024-01-01"}, spec)

        assert result["CURRENT_DATE"] == "2024-01-01"

    def test_missing_required_input(self) -> None:
        """Test error when required input is missing."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {required_input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="required_input", optional=False),),
        )

        with pytest.raises(MissingInputError) as exc_info:
            process_inputs({}, spec)

        assert "Missing required input" in str(exc_info.value)
        assert "required_input" in str(exc_info.value)

    def test_optional_input_missing_but_in_template_fails_fast(self) -> None:
        """Optional inputs referenced in template must still be provided."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {required} Optional: {optional}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(
                InputDefinition(name="required", optional=False),
                InputDefinition(name="optional", optional=True),
            ),
        )

        with pytest.raises(MissingInputError) as exc_info:
            process_inputs({"required": "value"}, spec)

        assert "optional" in str(exc_info.value)

        # Supplying it should succeed
        result = process_inputs({"required": "value", "optional": "x"}, spec)
        assert result["optional"] == "x"

    def test_missing_placeholder_raises_template_error(self) -> None:
        """Template interpolation raises TemplateError on missing placeholder."""
        with pytest.raises(TemplateError) as exc_info:
            interpolate_template("Hello {name}", {"other": "value"})

        assert "name" in str(exc_info.value)

    def test_type_coercion_str(self) -> None:
        """Test type coercion to string."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="str"),),
        )

        result = process_inputs({"input": 123}, spec)

        assert result["input"] == "123"
        assert isinstance(result["input"], str)

    def test_type_coercion_int(self) -> None:
        """Test type coercion to int."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="int"),),
        )

        result = process_inputs({"input": "42"}, spec)

        assert result["input"] == 42
        assert isinstance(result["input"], int)

    def test_type_coercion_int_fails(self) -> None:
        """Test error when int coercion fails."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="int"),),
        )

        with pytest.raises(InputTypeError) as exc_info:
            process_inputs({"input": "not a number"}, spec)

        assert "Cannot convert" in str(exc_info.value)

    def test_type_coercion_float(self) -> None:
        """Test type coercion to float."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="float"),),
        )

        result = process_inputs({"input": "3.14"}, spec)

        assert result["input"] == 3.14
        assert isinstance(result["input"], float)

    def test_type_coercion_bool_true(self) -> None:
        """Test type coercion to bool (true values)."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="bool"),),
        )

        for value in ["true", "True", "TRUE", "1", "yes", "on"]:
            result = process_inputs({"input": value}, spec)
            assert result["input"] is True

    def test_type_coercion_bool_false(self) -> None:
        """Test type coercion to bool (false values)."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="bool"),),
        )

        for value in ["false", "False", "FALSE", "0", "no", "off"]:
            result = process_inputs({"input": value}, spec)
            assert result["input"] is False

    def test_type_coercion_bool_invalid(self) -> None:
        """Test error when bool coercion fails."""
        spec = AgentSpec(
            model="openai:gpt-5",
            prompt_template="Input: {input}",
            output_fields=(FieldDefinition(name="field"),),
            input_definitions=(InputDefinition(name="input", type="bool"),),
        )

        with pytest.raises(InputTypeError) as exc_info:
            process_inputs({"input": "maybe"}, spec)

        assert "Cannot convert" in str(exc_info.value)


class TestInterpolateTemplate:
    """Tests for interpolate_template function."""

    def test_basic_interpolation(self) -> None:
        """Test basic template interpolation."""
        result = interpolate_template("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_placeholders(self) -> None:
        """Test template with multiple placeholders."""
        result = interpolate_template(
            "{greeting} {name}, you are {age} years old.",
            {"greeting": "Hello", "name": "Alice", "age": 30},
        )
        assert result == "Hello Alice, you are 30 years old."

    def test_repeated_placeholder(self) -> None:
        """Test template with repeated placeholder."""
        result = interpolate_template(
            "{name} likes {name}'s code.",
            {"name": "Alice"},
        )
        assert result == "Alice likes Alice's code."


class TestMagicVariables:
    """Tests for magic variables."""

    def test_all_magic_variables_defined(self) -> None:
        """Test that all expected magic variables are defined."""
        expected = {"CURRENT_DATE", "CURRENT_TIME", "CURRENT_DATETIME"}
        assert set(MAGIC_VARIABLES.keys()) == expected

    def test_magic_variables_are_uppercase(self) -> None:
        """Test that all magic variables are uppercase."""
        for name in MAGIC_VARIABLES:
            assert name == name.upper()

    def test_magic_variables_are_callable(self) -> None:
        """Test that all magic variables are callable and return strings."""
        for name, func in MAGIC_VARIABLES.items():
            result = func()
            assert isinstance(result, str), f"{name} should return a string"
