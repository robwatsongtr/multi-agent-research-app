"""Critic agent that validates research reports and identifies gaps."""

import logging
import json
from typing import Any, cast
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.json_utils import parse_json_response

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Critic agent that reviews synthesized research reports.

    Validates claims, identifies contradictions, flags gaps in research,
    and suggests areas for improvement.
    """

    def __init__(self, client: Anthropic, system_prompt: str) -> None:
        """
        Initialize the Critic agent.

        Args:
            client: Anthropic API client instance
            system_prompt: System prompt defining critic behavior
        """
        super().__init__(client, system_prompt)

    def review(self, report: dict[str, Any]) -> dict[str, Any]:
        """
        Review a synthesized research report and identify issues.

        Args:
            report: Research report from SynthesizerAgent.synthesize()
                    Should have structure with summary, sections, key_insights

        Returns:
            Dictionary with structure:
            {
                "overall_quality": str,
                "issues": [
                    {
                        "type": "unsupported_claim|contradiction|gap|other",
                        "description": str,
                        "location": str,
                        "severity": "high|medium|low"
                    }
                ],
                "suggestions": [str],
                "needs_more_research": bool
            }

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.info("Critic analyzing report...")

            # Format report into readable structure for Claude
            report_text = json.dumps(report, indent=2)
            user_message = f"Here is the research report to review:\n\n{report_text}"

            logger.debug(f"Critic input: {user_message[:100]}...")

            # Call Claude to review the report
            response = self.call_claude(
                user_message=user_message,
                max_tokens=4096,
                temperature=0.3  # Low temperature for consistent, analytical output
            )

            # Parse the response to get text content
            response_text = self.parse_response(response)

            # Use the shared JSON parsing utility
            result = parse_json_response(
                response_text,
                expected_fields=["overall_quality", "issues", "suggestions", "needs_more_research"]
            )
        except Exception as e:
            raise RuntimeError(f"Critic review failed: {e}") from e

        # Validate data types
        if not isinstance(result["issues"], list):
            raise ValueError("issues must be a list")

        if not isinstance(result["suggestions"], list):
            raise ValueError("suggestions must be a list")

        if not isinstance(result["needs_more_research"], bool):
            raise ValueError("needs_more_research must be a boolean")

        issue_count = len(result["issues"])
        suggestion_count = len(result["suggestions"])

        logger.info(f"Critic found {issue_count} issues, {suggestion_count} suggestions")

        return cast(dict[str, Any], result)
