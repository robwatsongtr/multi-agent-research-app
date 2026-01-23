"""Researcher agent that executes research subtasks."""

import logging
import json
import re
from typing import Any, Optional, cast
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Researcher agent that executes individual research subtasks.

    Takes a research subtask and returns structured findings with sources.
    Can use web_search tool to find relevant information.
    """

    def __init__(self, client: Anthropic, system_prompt: str) -> None:
        """
        Initialize the Researcher agent.

        Args:
            client: Anthropic API client instance
            system_prompt: System prompt defining researcher behavior
        """
        super().__init__(client, system_prompt)

    def research( self,
        subtask: str,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_executor: Optional[Any] = None
    ) -> dict[str, Any]:
        """
        Research a specific subtask and return findings.

        Args:
            subtask: The research subtask to investigate
            tools: Optional list of tools (e.g., web_search) the agent can use
            tool_executor: Optional function to execute tools

        Returns:
            Dictionary with structure:
            {
                "subtask": str,
                "findings": [
                    {
                        "claim": str,
                        "source": str,
                        "details": str
                    }
                ]
            }

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.debug(f"Researcher processing subtask: {subtask[:50]}...")
            # Call Claude with the subtask
            response = self.call_claude(
                user_message=subtask,
                max_tokens=4096,
                temperature=1.0,
                tools=tools,
                tool_executor=tool_executor
            )

            # Parse the response to get text content
            response_text = self.parse_response(response)
            logger.debug(f"Researcher response: {response_text[:100]}...")

            # Use the shared JSON parsing utility
            result = parse_json_response(
                response_text,
                expected_fields=["subtask", "findings"]
            )
        except Exception as e:
            raise RuntimeError(f"Research failed: {e}") from e

        # Validate findings is a list
        if not isinstance(result["findings"], list):
            raise ValueError("findings must be a list")

        return cast(dict[str, Any], result)
