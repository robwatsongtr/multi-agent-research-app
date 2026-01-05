"""Researcher agent that executes research subtasks."""

import json
import re
from typing import Any, Optional
from anthropic import Anthropic

from agents.base import BaseAgent


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
        tools: Optional[list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        """
        Research a specific subtask and return findings.

        Args:
            subtask: The research subtask to investigate
            tools: Optional list of tools (e.g., web_search) the agent can use

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
            # Call Claude with the subtask
            response = self.call_claude(
                user_message=subtask,
                max_tokens=4096,
                temperature=1.0,
                tools=tools
            )

            # Parse the response to get text content
            response_text = self.parse_response(response)

            # print(f"\nDEBUG - Researcher raw response:")
            # print(f"{response_text}")
            # print()

            # Strip markdown code blocks - find content between ``` markers
            if '```' in response_text:
                # Extract everything between first ``` and last ```
                parts = response_text.split('```')
                if len(parts) >= 3:
                    # parts[1] is the content between first and second ```
                    response_text = parts[1]
                    # Remove 'json' if it's at the start
                    if response_text.strip().startswith('json'):
                        response_text = response_text.strip()[4:]
                    response_text = response_text.strip()
            else:
                response_text = response_text.strip()

            # print(f"DEBUG - Researcher after stripping:")
            # print(f"{response_text}")
            # print()

            # Parse JSON
            result = json.loads(response_text)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Research failed: {e}") from e

        # Validate required fields
        if "subtask" not in result:
            raise ValueError("Response missing required field: subtask")
        if "findings" not in result:
            raise ValueError("Response missing required field: findings")

        # Validate findings is a list
        if not isinstance(result["findings"], list):
            raise ValueError("findings must be a list")

        return result
