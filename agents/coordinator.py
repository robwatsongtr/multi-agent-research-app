"""Coordinator agent that breaks queries into research subtasks."""

import logging
import json
import re
from typing import Any, List
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.json_utils import extract_json_from_response

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
            ValueError: If the response cannot be parsed as JSON
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

            # Use the shared JSON extraction utility
            json_text = extract_json_from_response(response_text)
            subtasks = json.loads(json_text)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Coordination failed: {e}") from e

        # Validate that we got a list
        if not isinstance(subtasks, list):
            raise ValueError(f"Expected list of subtasks, got {type(subtasks)}")

        # Validate we have 2-4 subtasks
        if len(subtasks) < 2 or len(subtasks) > 4:
            raise ValueError(f"Expected 2-4 subtasks, got {len(subtasks)}")

        # Validate all items are strings
        if not all(isinstance(task, str) for task in subtasks):
            raise ValueError("All subtasks must be strings")

        return subtasks
