"""Example script to test TextAgents with a real LLM."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

EXAMPLES_DIR = Path(__file__).parent
PROJECT_ROOT = EXAMPLES_DIR.parent

# Load API keys from .env BEFORE importing textagents
load_dotenv(PROJECT_ROOT / ".env")

import textagents  # noqa: E402


async def main():
    # Load the minimal judge
    agent = textagents.load_agent(EXAMPLES_DIR / "minimal_judge.txt")

    print(f"Loaded agent: {agent.name}")
    print(f"Model: {agent.model}")
    print(f"Required inputs: {agent.required_inputs}")
    print()

    # Run with a test input
    result = await agent.run(input="The sky is blue.")

    print("Result:")
    print(f"  reasoning: {result.reasoning}")
    print(f"  is_valid: {result.is_valid}")
    print()

    # Try the safety judge
    safety_agent = textagents.load_agent(EXAMPLES_DIR / "safety_judge.txt")

    print(f"Loaded agent: {safety_agent.name}")
    print(f"Required inputs: {safety_agent.required_inputs}")
    print()

    safety_result = await safety_agent.run(
        user_input="What's the weather like today?",
        model_output="I'd be happy to help! The weather varies by location. Could you tell me your city?",
    )

    print("Safety Result:")
    print(f"  reasoning: {safety_result.reasoning}")
    print(f"  is_safe: {safety_result.is_safe}")
    print(f"  no_hate: {safety_result.no_hate}")
    print(f"  no_pii: {safety_result.no_pii}")
    print(f"  is_helpful: {safety_result.is_helpful}")


if __name__ == "__main__":
    asyncio.run(main())
