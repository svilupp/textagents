"""Integration tests that call real LLM APIs.

These tests are skipped by default and only run when:
1. OPENAI_API_KEY is set
2. pytest is run with --run-real-api flag

To run: pytest tests/test_real_api.py --run-real-api -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()

# Skip all tests in this module if API key not available or flag not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("RUN_REAL_API_TESTS"),
    reason="Real API tests require OPENAI_API_KEY and RUN_REAL_API_TESTS=1",
)


@pytest.fixture
def examples_dir() -> Path:
    """Return the path to examples directory."""
    return Path(__file__).parent.parent / "examples"


class TestRealAPIIntegration:
    """Tests that actually call the OpenAI API."""

    @pytest.mark.asyncio
    async def test_minimal_judge(self, examples_dir: Path) -> None:
        """Test minimal judge with real API."""
        from textagents import load_agent

        agent = load_agent(examples_dir / "minimal_judge.txt")

        result = await agent.run(input="The sky is blue.")

        # Should have the expected fields
        assert hasattr(result, "reasoning")
        assert hasattr(result, "is_valid")
        assert isinstance(result.reasoning, str)
        assert isinstance(result.is_valid, bool)
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_safety_judge(self, examples_dir: Path) -> None:
        """Test safety judge with real API."""
        from textagents import load_agent

        agent = load_agent(examples_dir / "safety_judge.txt")

        result = await agent.run(
            user_input="What's the weather?",
            model_output="I'd be happy to help! Could you tell me your city?",
        )

        # Should have all safety fields
        assert hasattr(result, "reasoning")
        assert hasattr(result, "is_safe")
        assert hasattr(result, "no_hate")
        assert hasattr(result, "no_pii")
        assert hasattr(result, "is_helpful")

        # This should be a safe response
        assert result.is_safe is True
        assert result.no_hate is True
        assert result.no_pii is True

    def test_minimal_judge_sync(self, examples_dir: Path) -> None:
        """Test sync API."""
        from textagents import load_agent

        agent = load_agent(examples_dir / "minimal_judge.txt")

        result = agent.run_sync(input="2 + 2 = 4")

        assert isinstance(result.is_valid, bool)
        assert len(result.reasoning) > 0
