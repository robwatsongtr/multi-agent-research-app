"""Orchestration workflow for multi-agent research system."""

import logging
from datetime import datetime
from typing import Any
from anthropic import Anthropic

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.critic import CriticAgent
from agents.models import WorkflowResult
from tools import WEB_SEARCH_TOOL, execute_web_search

logger = logging.getLogger(__name__)


def run_research_workflow(
    query: str,
    client: Anthropic,
    coordinator_prompt: str,
    researcher_prompt: str,
    synthesizer_prompt: str,
    critic_prompt: str,
    tavily_api_key: str
) -> WorkflowResult:
    """
    Execute the full research workflow.

    Args:
        query: The user's research query
        client: Anthropic API client
        coordinator_prompt: System prompt for coordinator
        researcher_prompt: System prompt for researcher
        synthesizer_prompt: System prompt for synthesizer
        critic_prompt: System prompt for critic
        tavily_api_key: Tavily API key for web search

    Returns:
        WorkflowResult containing all research outputs
    """
    def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Execute tools requested by the researcher agent.

        Returns results as list of dicts for Anthropic API compatibility.
        """
        if tool_name == "web_search":
            search_query = tool_input["query"]
            logger.info(f"Searching web for: {search_query}")
            search_results = execute_web_search(search_query, tavily_api_key)
            logger.debug(f"Found {len(search_results)} results")

            return [result.model_dump() for result in search_results]
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    previous_year = current_year - 1

    researcher_prompt_with_date = researcher_prompt.format(
        current_date=current_date,
        current_year=current_year,
        previous_year=previous_year
    )

    logger.info("Breaking query into subtasks...")
    coordinator = CoordinatorAgent(client, coordinator_prompt)
    subtasks = coordinator.coordinate(query)
    logger.info(f"Generated {len(subtasks)} subtasks")

    logger.info(f"Starting research on {len(subtasks)} subtasks...")
    researcher = ResearcherAgent(client, researcher_prompt_with_date)
    research_results = []

    for i, subtask in enumerate(subtasks, 1):
        logger.info(f"[{i}/{len(subtasks)}] Researching subtask: {subtask}")
        findings = researcher.research(
            subtask,
            tools=[WEB_SEARCH_TOOL.to_dict()],
            tool_executor=tool_executor
        )
        research_results.append(findings)
        logger.info(f"Completed subtask {i}/{len(subtasks)}")

    logger.info("Synthesizing research findings...")
    synthesizer = SynthesizerAgent(client, synthesizer_prompt)
    synthesis = synthesizer.synthesize(research_results)

    logger.info("Running critic review...")
    critic = CriticAgent(client, critic_prompt)
    critique = critic.review(synthesis)

    return WorkflowResult(
        query=query,
        subtasks=subtasks,
        research_results=research_results,
        synthesis=synthesis,
        critique=critique
    )
