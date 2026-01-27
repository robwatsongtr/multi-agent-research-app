"""Coordinator agent that breaks queries into research subtasks."""

import logging
import json
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.models import CoordinatorResponse
from agents.parsing import extract_json_from_text

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

            # Extract and parse JSON (handles markdown code blocks)
            json_text = extract_json_from_text(response_text)
            subtasks_raw = json.loads(json_text)

            # Handle both formats: plain array or object with "subtasks" field
            if isinstance(subtasks_raw, list):
                # Plain array format: ["task1", "task2"]
                result = CoordinatorResponse(subtasks=subtasks_raw)
            elif isinstance(subtasks_raw, dict) and "subtasks" in subtasks_raw:
                # Object format: {"subtasks": ["task1", "task2"]}
                result = CoordinatorResponse(**subtasks_raw)
            else:
                raise ValueError(f"Unexpected JSON format: {type(subtasks_raw)}")

            logger.info(f"Generated {len(result.subtasks)} subtasks")

            return result.subtasks

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Coordination failed: {e}") from e
