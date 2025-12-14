"""Shared test fixtures for TextAgents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return the path to valid fixtures directory."""
    return fixtures_dir / "valid"


@pytest.fixture
def invalid_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return the path to invalid fixtures directory."""
    return fixtures_dir / "invalid"


@pytest.fixture
def minimal_agent_content() -> str:
    """Minimal valid agent definition."""
    return """---
[agent]
model = "openai:gpt-5"

[agent.output_type.reasoning]
type = "str"
description = "Brief reasoning."

[agent.output_type.is_valid]
description = "Whether valid."
---
Evaluate: {input}
"""


@pytest.fixture
def minimal_agent_compact_content() -> str:
    """Minimal valid agent definition using compact syntax."""
    return """---
[agent]
model = "openai:gpt-5"

[agent.output_type]
reasoning = { type = "str", description = "Brief reasoning." }
is_valid = { description = "Whether valid." }
---
Evaluate: {input}
"""


@pytest.fixture
def full_agent_content() -> str:
    """Full-featured agent definition."""
    return """---
[agent]
name = "test_judge"
model = "openai:gpt-5"
retries = 3
instructions = "Be strict and precise. Today is {CURRENT_DATE}."

[agent.settings]
temperature = 0

[agent.input_type]
user_input = { type = "str", description = "User message" }
context = { type = "str", optional = true }

[agent.output_type]
name = "TestOutput"
description = "Test output model"

[agent.output_type.reasoning]
type = "str"
max_length = 500

[agent.output_type.is_valid]
description = "Validity check"

[agent.output_type.score]
type = "int"
enum = [1, 2, 3, 4, 5]

[agent.output_type.confidence]
type = "float"
ge = 0.0
le = 1.0
---
Input: {user_input}
Context: {context}
"""


@pytest.fixture
def minimal_agent_meta() -> dict[str, Any]:
    """Parsed TOML metadata for minimal agent."""
    return {
        "agent": {
            "model": "openai:gpt-5",
            "output_type": {
                "reasoning": {"type": "str", "description": "Brief reasoning."},
                "is_valid": {"description": "Whether valid."},
            },
        }
    }


@pytest.fixture
def tmp_agent_file(tmp_path: Path, minimal_agent_content: str) -> Path:
    """Create a temporary agent file."""
    agent_file = tmp_path / "test_agent.txt"
    agent_file.write_text(minimal_agent_content)
    return agent_file
