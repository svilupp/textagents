"""TextAgent - the main wrapper class for text-defined agents.

This module provides the TextAgent class which wraps a PydanticAI
agent with template interpolation and simplified input handling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from .input_handler import MAGIC_VARIABLES, interpolate_template, process_inputs
from .model_builder import build_output_model
from .parser import AgentSpec
from .validator_builder import add_output_validator

OutputT = TypeVar("OutputT", bound=BaseModel)


class TextAgent(Generic[OutputT]):
    """A text-defined PydanticAI agent wrapper.

    Provides a simplified interface for running agents defined in .txt files
    with TOML front-matter. Handles template interpolation, input validation,
    and forwards to the underlying PydanticAI Agent.

    Example:
        ```python
        text_agent = load_agent("safety_judge.txt")

        # Async (primary)
        result = await text_agent.run(user_input="Hello", model_output="Hi there!")
        print(result.reasoning)
        print(result.no_hate)  # True

        # Sync convenience
        result = text_agent.run_sync(user_input="Hello", model_output="Hi there!")

        # Advanced: access underlying PydanticAI agent
        full_result = await text_agent.agent.run("custom prompt")
        print(full_result.all_messages())
        ```

    Attributes:
        spec: The parsed agent specification
        output_model: The dynamically generated Pydantic model for outputs
        agent: The underlying PydanticAI agent (public for advanced usage)
    """

    spec: AgentSpec
    output_model: type[OutputT]
    agent: Agent[None, OutputT]

    def __init__(
        self,
        spec: AgentSpec,
        output_model: type[OutputT],
        agent: Agent[None, OutputT],
    ) -> None:
        """Initialize TextAgent.

        Use load_agent() instead of constructing directly.

        Args:
            spec: Parsed agent specification
            output_model: Generated Pydantic output model
            agent: Configured PydanticAI agent
        """
        self.spec = spec
        self.output_model = output_model
        self.agent = agent

    async def run(self, **inputs: Any) -> OutputT:
        """Run the agent asynchronously with the given inputs.

        Args:
            **inputs: Named inputs matching agent.input_type definitions.
                     Use @filepath syntax to load from file.

        Returns:
            The validated output model instance (result.output from PydanticAI).

        Raises:
            MissingInputError: If required inputs are not provided.
            InputTypeError: If input types cannot be coerced.
        """
        # Process inputs (file loading, coercion, magic vars)
        processed = process_inputs(inputs, self.spec)

        # Interpolate the prompt template
        user_message = interpolate_template(self.spec.prompt_template, processed)

        # Interpolate instructions if present
        instructions = None
        if self.spec.instructions:
            instructions = interpolate_template(self.spec.instructions, processed)

        # Run the agent
        # Note: We pass instructions via the run call to support dynamic interpolation
        if instructions:
            result = await self.agent.run(user_message, instructions=instructions)
        else:
            result = await self.agent.run(user_message)

        return result.output

    def run_sync(self, **inputs: Any) -> OutputT:
        """Run the agent synchronously (convenience wrapper).

        Equivalent to asyncio.run(self.run(**inputs)).

        Args:
            **inputs: Named inputs matching agent.input_type definitions.

        Returns:
            The validated output model instance.
        """
        return asyncio.run(self.run(**inputs))

    @property
    def name(self) -> str:
        """Agent name (from spec or derived from filename)."""
        if self.spec.name:
            return self.spec.name
        if self.spec.source_path:
            return self.spec.source_path.stem
        return "unnamed_agent"

    @property
    def model(self) -> str:
        """Model identifier string."""
        return self.spec.model

    @property
    def input_names(self) -> list[str]:
        """List of expected input variable names."""
        # Combine defined inputs and template placeholders
        defined = {d.name for d in self.spec.input_definitions}
        placeholders = self.spec.all_placeholders - set(MAGIC_VARIABLES.keys())
        return sorted(defined | placeholders)

    @property
    def required_inputs(self) -> list[str]:
        """List of required (non-optional) input names."""
        # Defined inputs that are not optional
        defined_required = {
            d.name for d in self.spec.input_definitions if not d.optional
        }

        # Placeholders that aren't magic variables or optional defined inputs
        optional_defined = {d.name for d in self.spec.input_definitions if d.optional}
        placeholders = self.spec.all_placeholders - set(MAGIC_VARIABLES.keys())
        placeholder_required = placeholders - optional_defined

        return sorted(defined_required | placeholder_required)


def create_text_agent(spec: AgentSpec) -> TextAgent[Any]:
    """Create a TextAgent from an AgentSpec.

    This function:
    1. Builds the dynamic output model
    2. Creates the PydanticAI agent
    3. Adds the output validator
    4. Returns the configured TextAgent

    Args:
        spec: The parsed agent specification

    Returns:
        Configured TextAgent
    """
    # Build the output model
    output_model = build_output_model(spec)

    # Create the PydanticAI agent
    agent: Agent[None, Any] = Agent(  # type: ignore[call-overload]
        spec.model,
        output_type=output_model,
        instructions=spec.instructions,  # Static part, may be overridden in run()
        retries=spec.retries,
        model_settings=spec.settings if spec.settings else None,
    )

    # Add output validator
    add_output_validator(agent, spec)  # type: ignore[arg-type]

    return TextAgent(spec, output_model, agent)  # type: ignore[arg-type]
