"""Orchestration workflow for multi-agent research system."""

from typing import Any
from anthropic import Anthropic

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent


def run_research_workflow(
    query: str,
    client: Anthropic,
    coordinator_prompt: str,
    researcher_prompt: str,
    tools: list[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Execute the full research workflow.

    Args:
        query: The user's research query
        client: Anthropic API client
        coordinator_prompt: System prompt for coordinator
        researcher_prompt: System prompt for researcher
        tools: Optional tools (e.g., web_search) for researcher

    Returns:
        Dictionary with structure:
        {
            "query": str,
            "subtasks": list[str],
            "research_results": list[dict]
        }
    """
    # Step 1: Coordinator breaks down query into subtasks
    coordinator = CoordinatorAgent(client, coordinator_prompt)
    subtasks = coordinator.coordinate(query)

    # Step 2: Researcher investigates each subtask
    researcher = ResearcherAgent(client, researcher_prompt)
    research_results = []

    for subtask in subtasks:
        findings = researcher.research(subtask, tools=tools)
        research_results.append(findings)

    return {
        "query": query,
        "subtasks": subtasks,
        "research_results": research_results
    }
