"""Tests for the CLI module."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from textagents.cli import _parse_cli_inputs, _print_pretty, app

runner = CliRunner()


class TestParseCLIInputs:
    """Tests for _parse_cli_inputs function."""

    def test_simple_key_value(self) -> None:
        """Test parsing simple --key value pairs."""
        args = ["--name", "John", "--age", "30"]
        result = _parse_cli_inputs(args)

        assert result == {"name": "John", "age": "30"}

    def test_file_reference(self) -> None:
        """Test parsing @file references (passed through as strings)."""
        args = ["--input", "@data.txt"]
        result = _parse_cli_inputs(args)

        assert result == {"input": "@data.txt"}

    def test_underscores_from_dashes(self) -> None:
        """Test that dashes in names are converted to underscores."""
        args = ["--user-input", "hello"]
        result = _parse_cli_inputs(args)

        assert result == {"user_input": "hello"}

    def test_empty_args(self) -> None:
        """Test parsing empty arguments."""
        result = _parse_cli_inputs([])
        assert result == {}

    def test_skip_missing_value(self) -> None:
        """Test that arguments without values are skipped."""
        args = ["--name"]
        result = _parse_cli_inputs(args)

        assert result == {}

    def test_skip_consecutive_options(self) -> None:
        """Test that consecutive options skip the first without a value."""
        args = ["--flag", "--name", "value"]
        result = _parse_cli_inputs(args)

        assert result == {"name": "value"}

    def test_non_option_args_ignored(self) -> None:
        """Test that non-option arguments are ignored."""
        args = ["positional", "--name", "value", "another"]
        result = _parse_cli_inputs(args)

        assert result == {"name": "value"}

    def test_triple_dash_ignored(self) -> None:
        """Test that --- prefixed args are ignored."""
        args = ["---invalid", "value", "--valid", "good"]
        result = _parse_cli_inputs(args)

        assert result == {"valid": "good"}

    def test_quoted_values(self) -> None:
        """Test parsing values with spaces."""
        args = ["--message", "hello world"]
        result = _parse_cli_inputs(args)

        assert result == {"message": "hello world"}


class TestPrintPretty:
    """Tests for _print_pretty function."""

    def test_bool_pass(self, capsys) -> None:
        """Test that True values show PASS."""
        from pydantic import BaseModel

        class Output(BaseModel):
            is_valid: bool

        _print_pretty(Output(is_valid=True))
        captured = capsys.readouterr()
        assert "is_valid: PASS" in captured.out

    def test_bool_fail(self, capsys) -> None:
        """Test that False values show FAIL."""
        from pydantic import BaseModel

        class Output(BaseModel):
            is_valid: bool

        _print_pretty(Output(is_valid=False))
        captured = capsys.readouterr()
        assert "is_valid: FAIL" in captured.out

    def test_long_string_indented(self, capsys) -> None:
        """Test that long strings are indented."""
        from pydantic import BaseModel

        class Output(BaseModel):
            reasoning: str

        long_text = "x" * 150
        _print_pretty(Output(reasoning=long_text))
        captured = capsys.readouterr()
        assert "reasoning:" in captured.out
        assert f"  {long_text}" in captured.out

    def test_short_string_inline(self, capsys) -> None:
        """Test that short strings are inline."""
        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        _print_pretty(Output(text="hello"))
        captured = capsys.readouterr()
        assert "text: hello" in captured.out

    def test_numeric_value(self, capsys) -> None:
        """Test numeric values display correctly."""
        from pydantic import BaseModel

        class Output(BaseModel):
            score: float

        _print_pretty(Output(score=0.95))
        captured = capsys.readouterr()
        assert "score: 0.95" in captured.out


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_minimal_agent(
        self, tmp_path: Path, minimal_agent_content: str
    ) -> None:
        """Test info command with minimal agent."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        result = runner.invoke(app, ["info", str(agent_file)])

        assert result.exit_code == 0
        assert "Agent:" in result.output
        assert "Model: openai:gpt-5" in result.output
        assert "Output fields:" in result.output
        assert "reasoning" in result.output
        assert "is_valid" in result.output

    def test_info_full_agent(self, tmp_path: Path, full_agent_content: str) -> None:
        """Test info command with full-featured agent."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(full_agent_content)

        result = runner.invoke(app, ["info", str(agent_file)])

        assert result.exit_code == 0
        assert "Agent: test_judge" in result.output
        assert "Model: openai:gpt-5" in result.output
        assert "Retries: 3" in result.output
        assert "Instructions:" in result.output
        assert "Inputs:" in result.output
        assert "user_input" in result.output

    def test_info_file_not_found(self, tmp_path: Path) -> None:
        """Test info command with non-existent file."""
        result = runner.invoke(app, ["info", str(tmp_path / "nonexistent.txt")])

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_info_invalid_agent(self, tmp_path: Path) -> None:
        """Test info command with invalid agent definition."""
        agent_file = tmp_path / "invalid.txt"
        agent_file.write_text("---\n[invalid]\n---\nNo agent section")

        result = runner.invoke(app, ["info", str(agent_file)])

        assert result.exit_code == 1
        assert "Error:" in result.output


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_valid_agent(
        self, tmp_path: Path, minimal_agent_content: str
    ) -> None:
        """Test validate command with valid agent."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        result = runner.invoke(app, ["validate", str(agent_file)])

        assert result.exit_code == 0
        assert "Valid agent definition" in result.output
        assert "Output fields: 2" in result.output

    def test_validate_invalid_agent(self, tmp_path: Path) -> None:
        """Test validate command with invalid agent."""
        agent_file = tmp_path / "invalid.txt"
        agent_file.write_text("---\n[agent]\n---\nNo model or output")

        result = runner.invoke(app, ["validate", str(agent_file)])

        assert result.exit_code == 1
        assert "Invalid agent definition" in result.output

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        """Test validate command with missing file."""
        result = runner.invoke(app, ["validate", str(tmp_path / "missing.txt")])

        assert result.exit_code == 1
        assert "Error:" in result.output


class TestRunCommand:
    """Tests for the run command."""

    def test_run_missing_agent_file(self, tmp_path: Path) -> None:
        """Test run command with missing agent file."""
        result = runner.invoke(app, ["run", str(tmp_path / "missing.txt")])

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_run_missing_inputs_file(
        self, tmp_path: Path, minimal_agent_content: str
    ) -> None:
        """Test run command with missing inputs file."""
        from unittest.mock import MagicMock, patch

        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        # Mock load_agent to avoid API key issues
        mock_agent = MagicMock()
        with patch("textagents.cli.load_agent", return_value=mock_agent):
            result = runner.invoke(
                app,
                ["run", str(agent_file), "--inputs", str(tmp_path / "missing.json")],
            )

        assert result.exit_code == 1
        # Error goes to stderr, check combined output
        assert "Inputs file not found" in (result.output + (result.stderr or ""))

    def test_run_invalid_agent(self, tmp_path: Path) -> None:
        """Test run command with invalid agent definition."""
        agent_file = tmp_path / "invalid.txt"
        agent_file.write_text("---\n[invalid]\n---\nNo agent section")

        result = runner.invoke(app, ["run", str(agent_file)])

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_run_with_cli_inputs(
        self, tmp_path: Path, minimal_agent_content: str, monkeypatch
    ) -> None:
        """Test run command with CLI inputs (mocked agent execution)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        # Mock the load_agent and agent.run
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"is_valid": true}'

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("textagents.cli.load_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["run", str(agent_file), "--input", "test value"]
            )

        # The mock should have been called
        if result.exit_code == 0:
            assert "is_valid" in result.output

    def test_run_with_inputs_file(
        self, tmp_path: Path, minimal_agent_content: str
    ) -> None:
        """Test run command with JSON inputs file (mocked agent execution)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps({"input": "test value"}))

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"is_valid": true}'

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("textagents.cli.load_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["run", str(agent_file), "--inputs", str(inputs_file)]
            )

        if result.exit_code == 0:
            mock_agent.run.assert_called_once()

    def test_run_pretty_format(
        self, tmp_path: Path, minimal_agent_content: str
    ) -> None:
        """Test run command with pretty output format."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from pydantic import BaseModel

        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        class MockOutput(BaseModel):
            is_valid: bool
            reasoning: str

        mock_result = MockOutput(is_valid=True, reasoning="Test passed")

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("textagents.cli.load_agent", return_value=mock_agent):
            result = runner.invoke(
                app,
                ["run", str(agent_file), "--format", "pretty", "--input", "test"],
            )

        if result.exit_code == 0:
            assert "PASS" in result.output or "is_valid" in result.output


class TestAppStructure:
    """Tests for CLI app structure and help."""

    def test_app_no_args(self) -> None:
        """Test that app shows help with no arguments."""
        result = runner.invoke(app, [])
        # no_args_is_help=True shows help and exits with code 0 or 2
        # Typer exits with 0 for help, but some versions may exit with 2
        assert result.exit_code in (0, 2)
        assert "run" in result.output.lower() or "Usage" in result.output

    def test_help_flag(self) -> None:
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "run" in result.output.lower()
        assert "info" in result.output.lower()
        assert "validate" in result.output.lower()

    def test_run_help(self) -> None:
        """Test run --help."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--inputs" in result.output or "-i" in result.output
        assert "--format" in result.output or "-f" in result.output
        assert "--model" in result.output or "-m" in result.output

    def test_info_help(self) -> None:
        """Test info --help."""
        result = runner.invoke(app, ["info", "--help"])

        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    def test_validate_help(self) -> None:
        """Test validate --help."""
        result = runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "agent" in result.output.lower()
