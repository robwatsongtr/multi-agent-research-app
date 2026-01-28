"""Researcher agent that executes research subtasks."""

import logging
import json
from typing import Any, Optional, Callable
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.models import ResearchResult
from agents.parsing import extract_json_from_text

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

    def research(
        self,
        subtask: str,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, dict[str, Any]], Any]] = None
    ) -> ResearchResult:
        """
        Research a specific subtask and return findings.

        Args:
            subtask: The research subtask to investigate
            tools: Optional list of tools (e.g., web_search) the agent can use
            tool_executor: Optional function to execute tools

        Returns:
            ResearchResult with subtask and list of findings

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.debug(f"Researcher processing subtask: {subtask[:50]}...")
            response = self.call_claude(
                user_message=subtask,
                max_tokens=4096,
                temperature=1.0,
                tools=tools,
                tool_executor=tool_executor
            )

            response_text = self.parse_response(response)
            logger.debug(f"Researcher response: {response_text[:100]}...")

            json_text = extract_json_from_text(response_text)
            result_dict = json.loads(json_text)

            result = ResearchResult(**result_dict)

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Research failed: {e}") from e
