"""Orchestration workflow for multi-agent research system."""

import logging
from typing import Any
from anthropic import Anthropic

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from tools import WEB_SEARCH_TOOL, execute_web_search

logger = logging.getLogger(__name__)


def run_research_workflow(
    query: str,
    client: Anthropic,
    coordinator_prompt: str,
    researcher_prompt: str,
    tavily_api_key: str
) -> dict[str, Any]:
    """
    Execute the full research workflow.

    Args:
        query: The user's research query
        client: Anthropic API client
        coordinator_prompt: System prompt for coordinator
        researcher_prompt: System prompt for researcher
        tavily_api_key: Tavily API key for web search

    Returns:
        Dictionary with structure:
        {
            "query": str,
            "subtasks": list[str],
            "research_results": list[dict]
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

    # Step 1: Coordinator breaks down query into subtasks
    logger.info("Breaking query into subtasks...")
    coordinator = CoordinatorAgent(client, coordinator_prompt)
    subtasks = coordinator.coordinate(query)
    logger.info(f"Generated {len(subtasks)} subtasks")

    # Step 2: Researcher investigates each subtask
    logger.info(f"Starting research on {len(subtasks)} subtasks...")
    researcher = ResearcherAgent(client, researcher_prompt)
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

    return {
        "query": query,
        "subtasks": subtasks,
        "research_results": research_results
    }
