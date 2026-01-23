"""Synthesizer agent that combines research findings into coherent output."""

import logging
import json
from typing import Any, cast
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer agent that combines research findings into a coherent summary.

    Takes multiple research findings and organizes them by theme/topic,
    preserving source citations and creating a logical flow.
    """

    def __init__(self, client: Anthropic, system_prompt: str) -> None:
        """
        Initialize the Synthesizer agent.

        Args:
            client: Anthropic API client instance
            system_prompt: System prompt defining synthesizer behavior
        """
        super().__init__(client, system_prompt)

    def synthesize(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Combine research findings into a coherent summary.

        Args:
            findings: List of research findings from researcher agents
                      Each finding should have structure from ResearcherAgent.research()

        Returns:
            Dictionary with structure:
            {
                "summary": str,
                "sections": [
                    {
                        "title": str,
                        "content": str
                    }
                ],
                "key_insights": [str]
            }

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.info(f"Synthesizer processing {len(findings)} findings...")

            # Format findings into a readable structure for Claude
            findings_text = json.dumps(findings, indent=2)
            user_message = f"Here are the research findings to synthesize:\n\n{findings_text}"

            logger.debug(f"Synthesizer input: {user_message[:100]}...")

            # Call Claude to synthesize the findings
            # Note: We could try using response_format={"type": "json_object"} but that's not
            # supported by all models. Instead, we'll rely on robust parsing below.
            response = self.call_claude(
                user_message=user_message,
                max_tokens=4096,
                temperature=0.7  # Lower temperature for more consistent JSON output
            )

            # Parse the response to get text content
            response_text = self.parse_response(response)

            # Use the shared JSON parsing utility
            result = parse_json_response(
                response_text,
                expected_fields=["summary", "sections", "key_insights"]
            )
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}") from e

        # Validate data types
        if not isinstance(result["sections"], list):
            raise ValueError("sections must be a list")

        if not isinstance(result["key_insights"], list):
            raise ValueError("key_insights must be a list")

        logger.info(f"Generated report with {len(result['sections'])} sections")

        return cast(dict[str, Any], result)
