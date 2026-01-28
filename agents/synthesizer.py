"""Synthesizer agent that combines research findings into coherent output."""

import logging
import json
from anthropic import Anthropic

from agents.base import BaseAgent
from agents.models import ResearchResult, SynthesizedReport
from agents.parsing import extract_json_from_text

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

    def synthesize(self, findings: list[ResearchResult]) -> SynthesizedReport:
        """
        Combine research findings into a coherent summary.

        Args:
            findings: List of ResearchResult from researcher agents

        Returns:
            SynthesizedReport with organized sections and insights

        Raises:
            ValueError: If the response cannot be parsed or is invalid
            RuntimeError: If API call fails
        """
        try:
            logger.info(f"Synthesizer processing {len(findings)} findings...")

            findings_dicts = [finding.model_dump() for finding in findings]
            findings_text = json.dumps(findings_dicts, indent=2)
            user_message = f"Here are the research findings to synthesize:\n\n{findings_text}"

            logger.debug(f"Synthesizer input: {user_message[:100]}...")

            response = self.call_claude(
                user_message=user_message,
                max_tokens=4096,
                temperature=0.7
            )

            response_text = self.parse_response(response)
            json_text = extract_json_from_text(response_text)
            result_dict = json.loads(json_text)

            result = SynthesizedReport(**result_dict)

            logger.info(f"Generated report with {len(result.sections)} sections")

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}") from e
