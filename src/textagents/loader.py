"""Load TextAgents from files.

This module provides the main entry point for loading text-defined
agents from files, with optional Logfire instrumentation.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import TextAgent, create_text_agent
from .parser import parse_agent_file

# Track if Logfire has been configured
_logfire_configured = False


def load_agent(
    path: str | Path,
    *,
    model_override: str | None = None,
    logfire_token: str | None = None,
) -> TextAgent[Any]:
    """Load a TextAgent from a file.

    Args:
        path: Path to the agent definition file (.txt with TOML front-matter)
        model_override: Override the model specified in the file
        logfire_token: Logfire token for auto-instrumentation
                      (or set LOGFIRE_TOKEN env var)

    Returns:
        A configured TextAgent ready to run.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        AgentDefinitionError: If the agent definition is invalid.

    Example:
        ```python
        # Basic usage
        text_agent = load_agent("judges/safety.txt")

        # Override model for testing
        text_agent = load_agent(
            "judges/safety.txt",
            model_override="openai:gpt-4.1-nano"
        )
        ```
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

    # Configure Logfire if token available
    _maybe_configure_logfire(logfire_token)

    # Parse the agent file
    spec = parse_agent_file(path)

    # Apply model override if specified
    if model_override:
        spec = replace(spec, model=model_override)

    # Create and return the TextAgent
    return create_text_agent(spec)


def _maybe_configure_logfire(token: str | None = None) -> bool:
    """Configure Logfire instrumentation if token is available.

    Only configures once per process to avoid duplicate instrumentation.

    Args:
        token: Explicit token, or reads from LOGFIRE_TOKEN env var

    Returns:
        True if Logfire was configured, False otherwise
    """
    global _logfire_configured

    if _logfire_configured:
        return True

    token = token or os.environ.get("LOGFIRE_TOKEN")

    if not token:
        return False

    try:
        import logfire

        logfire.configure(token=token)
        logfire.instrument_pydantic_ai()
        _logfire_configured = True
        return True
    except ImportError:
        # logfire not installed
        return False
    except (TypeError, ValueError, RuntimeError, OSError):
        # Configuration failed, continue without Logfire
        return False
