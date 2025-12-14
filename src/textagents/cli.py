"""Command-line interface for TextAgents.

Provides commands to run, inspect, and validate text-defined agents.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from .errors import TextAgentsError
from .loader import load_agent
from .parser import parse_agent_file

# Load .env for API keys
load_dotenv()

app = typer.Typer(
    name="textagents",
    help="Run text-defined PydanticAI agents from the command line.",
    no_args_is_help=True,
)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": True,
        "ignore_unknown_options": True,
    }
)
def run(
    ctx: typer.Context,
    agent_file: Annotated[Path, typer.Argument(help="Path to agent definition file")],
    inputs_file: Annotated[
        Path | None,
        typer.Option("--inputs", "-i", help="JSON file with input values"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or pretty"),
    ] = "json",
    model_override: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override the model from the agent file"),
    ] = None,
) -> None:
    """Run an agent with the specified inputs.

    Inputs can be provided via:
    - --inputs: JSON file with all inputs
    - CLI arguments: --input_name "value" or --input_name @file.txt
    """

    async def _run() -> None:
        try:
            # Load the agent
            text_agent = load_agent(agent_file, model_override=model_override)

            # Collect inputs
            inputs: dict[str, Any] = {}

            # Load from JSON file if provided
            if inputs_file:
                if not inputs_file.exists():
                    typer.echo(f"Error: Inputs file not found: {inputs_file}", err=True)
                    raise typer.Exit(1)
                inputs = json.loads(inputs_file.read_text())

            # Parse additional CLI arguments from context
            cli_inputs = _parse_cli_inputs(ctx.args)
            inputs.update(cli_inputs)

            # Run the agent
            result = await text_agent.run(**inputs)

            # Output the result
            if output_format == "pretty":
                _print_pretty(result)
            else:
                typer.echo(result.model_dump_json(indent=2))

        except TextAgentsError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    asyncio.run(_run())


@app.command()
def info(
    agent_file: Annotated[Path, typer.Argument(help="Path to agent definition file")],
) -> None:
    """Show information about an agent definition."""
    try:
        spec = parse_agent_file(agent_file)

        typer.echo(f"Agent: {spec.name or agent_file.stem}")
        typer.echo(f"Model: {spec.model}")
        typer.echo(f"Retries: {spec.retries}")

        if spec.instructions:
            typer.echo(f"\nInstructions: {spec.instructions[:100]}...")

        typer.echo("\nInputs:")
        if spec.input_definitions:
            for inp in spec.input_definitions:
                optional = " (optional)" if inp.optional else ""
                desc = f" - {inp.description}" if inp.description else ""
                typer.echo(f"  {inp.name}: {inp.type}{optional}{desc}")
        else:
            # Show placeholders from template
            from .input_handler import MAGIC_VARIABLES

            placeholders = spec.all_placeholders - set(MAGIC_VARIABLES.keys())
            for p in sorted(placeholders):
                typer.echo(f"  {p}: str (from template)")

        typer.echo("\nOutput fields:")
        for field in spec.output_fields:
            optional = " (optional)" if field.optional else ""
            desc = f" - {field.description}" if field.description else ""
            enum_str = f" [{', '.join(map(str, field.enum))}]" if field.enum else ""
            typer.echo(f"  {field.name}: {field.type}{enum_str}{optional}{desc}")

        typer.echo(f"\nPrompt template ({len(spec.prompt_template)} chars):")
        preview = spec.prompt_template[:200]
        if len(spec.prompt_template) > 200:
            preview += "..."
        typer.echo(f"  {preview}")

    except TextAgentsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def validate(
    agent_file: Annotated[Path, typer.Argument(help="Path to agent definition file")],
) -> None:
    """Validate an agent definition without running it."""
    try:
        spec = parse_agent_file(agent_file)

        # Also try building the output model
        from .model_builder import build_output_model

        build_output_model(spec)

        typer.echo(f"Valid agent definition: {agent_file}")
        typer.echo(f"  Name: {spec.name or agent_file.stem}")
        typer.echo(f"  Model: {spec.model}")
        typer.echo(f"  Output fields: {len(spec.output_fields)}")
        typer.echo(f"  Input placeholders: {len(spec.all_placeholders)}")

    except TextAgentsError as e:
        typer.echo(f"Invalid agent definition: {e}", err=True)
        raise typer.Exit(1) from None
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


def _parse_cli_inputs(extra_args: list[str]) -> dict[str, str]:
    """Parse extra CLI arguments as input values.

    Handles:
    - --name value
    - --name @file.txt (file loading handled later)

    Args:
        extra_args: Extra arguments from typer context

    Returns:
        Dict of input name to value
    """
    inputs: dict[str, str] = {}

    i = 0
    while i < len(extra_args):
        arg = extra_args[i]

        # Look for --name value patterns
        if arg.startswith("--") and not arg.startswith("---"):
            name = arg[2:].replace("-", "_")

            # Skip if no value follows
            if i + 1 >= len(extra_args):
                i += 1
                continue

            value = extra_args[i + 1]

            # Skip if value looks like another option
            if value.startswith("--"):
                i += 1
                continue

            inputs[name] = value
            i += 2
        else:
            i += 1

    return inputs


def _print_pretty(result: Any) -> None:
    """Print result in human-readable format."""
    for field_name, value in result.model_dump().items():
        if isinstance(value, bool):
            icon = "PASS" if value else "FAIL"
            typer.echo(f"{field_name}: {icon}")
        elif isinstance(value, str) and len(value) > 100:
            typer.echo(f"{field_name}:")
            typer.echo(f"  {value}")
        else:
            typer.echo(f"{field_name}: {value}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
