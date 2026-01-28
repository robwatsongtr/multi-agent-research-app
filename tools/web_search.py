"""Web search tool using Tavily API."""

import logging
from typing import Any
from tavily import TavilyClient
from agents.models import SearchResult, ToolSchema

logger = logging.getLogger(__name__)

# Tool schema for Anthropic API
WEB_SEARCH_TOOL = ToolSchema(
    name="web_search",
    description="Search the web for current information on a topic. Returns relevant search results with titles, URLs, and content snippets.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            }
        },
        "required": ["query"]
    }
)


def execute_web_search(query: str, api_key: str) -> list[SearchResult]:
    """
    Execute a web search using Tavily API.

    Args:
        query: The search query
        api_key: Tavily API key

    Returns:
        List of SearchResult objects with title, url, content, and score

    Raises:
        Exception: If the search API call fails
    """
    client = TavilyClient(api_key=api_key)

    logger.debug(f"Executing Tavily search: {query}")
    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        raise

    results = []
    for result in response.get("results", []):
        search_result = SearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0)
        )
        results.append(search_result)

    logger.debug(f"Tavily returned {len(results)} results")

    return results
