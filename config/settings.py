"""Configuration and settings for the multi-agent research system."""

import os
import yaml
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

def load_prompts() -> dict[str, str]:
    """
    Load agent prompts from prompts.yaml.

    Returns:
        Dictionary mapping agent names to their system prompts

    Raises:
        FileNotFoundError: If prompts.yaml is not found
        yaml.YAMLError: If YAML parsing fails
    """
    prompts_path = Path(__file__).parent / "prompts.yaml"

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path, 'r') as f:
        prompts = yaml.safe_load(f)

    return prompts


def get_api_key() -> str:
    """
    Get Anthropic API key from environment.

    Returns:
        The API key

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )

    return api_key


def get_model() -> str:
    """
    Get the Claude model to use.

    Returns:
        Model identifier (can be overridden with CLAUDE_MODEL env var)
    """
    return os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
