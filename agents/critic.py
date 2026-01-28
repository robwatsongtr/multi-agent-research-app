"""Critic agent that validates research reports and identifies gaps."""

import logging
import json
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.models import SynthesizedReport, CriticReview
from agents.parsing import extract_json_from_text

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

    def review(self, report: SynthesizedReport) -> CriticReview:
        """
        Review a synthesized research report and identify issues.

        Args:
            report: SynthesizedReport from SynthesizerAgent

        Returns:
            CriticReview with quality assessment, issues, and suggestions

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.info("Critic analyzing report...")

            report_dict = report.model_dump()
            report_text = json.dumps(report_dict, indent=2)
            user_message = f"Here is the research report to review:\n\n{report_text}"

            logger.debug(f"Critic input: {user_message[:100]}...")

            response = self.call_claude(
                user_message=user_message,
                max_tokens=4096,
                temperature=0.3
            )

            response_text = self.parse_response(response)
            json_text = extract_json_from_text(response_text)
            result_dict = json.loads(json_text)

            result = CriticReview(**result_dict)

            issue_count = len(result.issues)
            suggestion_count = len(result.suggestions)

            logger.info(f"Critic found {issue_count} issues, {suggestion_count} suggestions")

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Critic review failed: {e}") from e
