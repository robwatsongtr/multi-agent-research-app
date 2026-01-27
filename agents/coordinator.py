"""Coordinator agent that breaks queries into research subtasks."""

import logging
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.models import CoordinatorResponse
from agents.parsing import parse_llm_response

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent that breaks down complex research queries into focused subtasks.

    Takes a user's research query and returns 2-4 independent research subtasks
    that can be investigated separately.
    """

    def __init__(self, client: Anthropic, system_prompt: str) -> None:
        """
        Initialize the Coordinator agent.

        Args:
            client: Anthropic API client instance
            system_prompt: System prompt defining coordinator behavior
        """
        super().__init__(client, system_prompt)

    def coordinate(self, query: str) -> list[str]:
        """
        Break down a research query into subtasks.

        Args:
            query: The user's research query

        Returns:
            List of 2-4 research subtasks

        Raises:
            ValueError: If the response cannot be parsed or validated
            RuntimeError: If API call fails
        """
        try:
            logger.info("Coordinator analyzing query...")
            # Call Claude with the user query
            response = self.call_claude(
                user_message=query,
                max_tokens=2048,
                temperature=1.0
            )

            # Parse the response to get text content
            response_text = self.parse_response(response)
            logger.debug(f"Coordinator response: {response_text[:100]}...")

            # Define fallback parser for plain text lists
            def fallback_parser(text: str) -> dict:
                """Extract subtasks from plain text if JSON fails."""
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                subtasks = [line.lstrip('1234567890.-) ') for line in lines if line]
                return {"subtasks": subtasks[:4]}  # Limit to 4

            # Parse using Pydantic model
            result = parse_llm_response(
                response_text,
                CoordinatorResponse,
                fallback_parser=fallback_parser
            )

            logger.info(f"Generated {len(result.subtasks)} subtasks")

            return result.subtasks

        except Exception as e:
            raise RuntimeError(f"Coordination failed: {e}") from e
