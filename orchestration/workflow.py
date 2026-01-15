"""Orchestration workflow for multi-agent research system."""

from typing import Any
from anthropic import Anthropic

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from tools import WEB_SEARCH_TOOL, execute_web_search


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
            return execute_web_search(tool_input["query"], tavily_api_key)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # Step 1: Coordinator breaks down query into subtasks
    coordinator = CoordinatorAgent(client, coordinator_prompt)
    subtasks = coordinator.coordinate(query)

    # Step 2: Researcher investigates each subtask
    researcher = ResearcherAgent(client, researcher_prompt)
    research_results = []

    for subtask in subtasks:
        findings = researcher.research(
            subtask,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=tool_executor
        )
        research_results.append(findings)

    return {
        "query": query,
        "subtasks": subtasks,
        "research_results": research_results
    }
