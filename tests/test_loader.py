"""Tests for the loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textagents.errors import AgentDefinitionError
from textagents.loader import load_agent


@pytest.fixture
def mock_pydantic_agent():
    """Mock PydanticAI Agent to avoid needing API keys."""
    with patch("textagents.agent.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent_class


class TestLoadAgent:
    """Tests for load_agent function."""

    def test_load_minimal_agent(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test loading a minimal agent definition."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file)

        assert text_agent.model == "openai:gpt-5"
        assert text_agent.name == "agent"  # from filename
        assert "input" in text_agent.input_names

    def test_load_compact_syntax(
        self,
        tmp_path: Path,
        minimal_agent_compact_content: str,
        mock_pydantic_agent: MagicMock,
    ) -> None:
        """Test loading agent with compact {} syntax."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_compact_content)

        text_agent = load_agent(agent_file)

        assert text_agent.model == "openai:gpt-5"
        # Should have both fields
        assert len(text_agent.spec.output_fields) == 2

    def test_load_full_agent(
        self, tmp_path: Path, full_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test loading a full-featured agent."""
        agent_file = tmp_path / "test_judge.txt"
        agent_file.write_text(full_agent_content)

        text_agent = load_agent(agent_file)

        assert text_agent.name == "test_judge"
        assert text_agent.model == "openai:gpt-5"
        assert text_agent.spec.retries == 3
        assert text_agent.spec.instructions is not None

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_agent("nonexistent.txt")

    def test_model_override(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test model override."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file, model_override="anthropic:claude-3")

        assert text_agent.model == "anthropic:claude-3"

    def test_agent_has_output_model(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that loaded agent has correct output model."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file)

        # Check output model has expected fields
        assert hasattr(text_agent.output_model, "model_fields")
        assert "reasoning" in text_agent.output_model.model_fields
        assert "is_valid" in text_agent.output_model.model_fields

    def test_agent_has_pydantic_agent(
        self, tmp_path: Path, minimal_agent_content: str, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test that loaded agent exposes underlying PydanticAI agent."""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(minimal_agent_content)

        text_agent = load_agent(agent_file)

        # Should have .agent accessor
        assert text_agent.agent is not None

    def test_required_inputs(
        self, tmp_path: Path, mock_pydantic_agent: MagicMock
    ) -> None:
        """Test required_inputs property."""
        content = """---
[agent]
model = "openai:gpt-5"

[agent.input_type]
required = { type = "str" }
optional = { type = "str", optional = true }

[agent.output_type.field]
---
Input: {required} {optional}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        text_agent = load_agent(agent_file)

        assert "required" in text_agent.required_inputs
        assert "optional" not in text_agent.required_inputs

    def test_invalid_agent_definition(self, tmp_path: Path) -> None:
        """Test error for invalid agent definition."""
        # Missing model
        content = """---
[agent]
output_type = { field = {} }
---
Test: {input}
"""
        agent_file = tmp_path / "agent.txt"
        agent_file.write_text(content)

        with pytest.raises(AgentDefinitionError):
            load_agent(agent_file)
