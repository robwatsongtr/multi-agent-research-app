"""Orchestration workflow for multi-agent research system."""

import logging
from datetime import datetime
from typing import Any
from anthropic import Anthropic

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
from tools import WEB_SEARCH_TOOL, execute_web_search

logger = logging.getLogger(__name__)


def run_research_workflow(
    query: str,
    client: Anthropic,
    coordinator_prompt: str,
    researcher_prompt: str,
    synthesizer_prompt: str,
    tavily_api_key: str
) -> dict[str, Any]:
    """
    Execute the full research workflow.

    Args:
        query: The user's research query
        client: Anthropic API client
        coordinator_prompt: System prompt for coordinator
        researcher_prompt: System prompt for researcher
        synthesizer_prompt: System prompt for synthesizer
        tavily_api_key: Tavily API key for web search

    Returns:
        Dictionary with structure:
        {
            "query": str,
            "subtasks": list[str],
            "research_results": list[dict],
            "synthesis": dict
        }
    """
    # Create tool executor function
    def tool_executor(tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute tools requested by the researcher agent."""
        if tool_name == "web_search":
            search_query = tool_input["query"]
            logger.info(f"Searching web for: {search_query}")
            result = execute_web_search(search_query, tavily_api_key)
            logger.debug(f"Found {len(result)} results")

            return result
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # Get current date for context
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    previous_year = current_year - 1

    # Inject current date into researcher prompt using template placeholders
    researcher_prompt_with_date = researcher_prompt.format(
        current_date=current_date,
        current_year=current_year,
        previous_year=previous_year
    )

    # Step 1: Coordinator breaks down query into subtasks
    logger.info("Breaking query into subtasks...")
    coordinator = CoordinatorAgent(client, coordinator_prompt)
    subtasks = coordinator.coordinate(query)
    logger.info(f"Generated {len(subtasks)} subtasks")

    # Step 2: Researcher investigates each subtask
    logger.info(f"Starting research on {len(subtasks)} subtasks...")
    researcher = ResearcherAgent(client, researcher_prompt_with_date)
    research_results = []

    for i, subtask in enumerate(subtasks, 1):
        logger.info(f"[{i}/{len(subtasks)}] Researching subtask: {subtask}")
        findings = researcher.research(
            subtask,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=tool_executor
        )
        research_results.append(findings)
        logger.info(f"Completed subtask {i}/{len(subtasks)}")

    # Step 3: Synthesizer combines findings into coherent report
    logger.info("Synthesizing research findings...")
    synthesizer = SynthesizerAgent(client, synthesizer_prompt)
    synthesis = synthesizer.synthesize(research_results)

    return {
        "query": query,
        "subtasks": subtasks,
        "research_results": research_results,
        "synthesis": synthesis
    }
