"""Web search tool using Tavily API."""

from typing import Any
from tavily import TavilyClient

# Tool schema for Anthropic API
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information on a topic. Returns relevant search results with titles, URLs, and content snippets.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            }
        },
        "required": ["query"]
    }
}


def execute_web_search(query: str, api_key: str) -> list[dict[str, Any]]:
    """
    Execute a web search using Tavily API.

    Args:
        query: The search query
        api_key: Tavily API key

    Returns:
        List of search results with structure:
        [
            {
                "title": str,
                "url": str,
                "content": str,
                "score": float
            }
        ]

    Raises:
        Exception: If the search API call fails
    """
    client = TavilyClient(api_key=api_key)

    # Perform search with Tavily
    response = client.search(
        query=query,
        search_depth="basic",  # or "advanced" for more thorough search
        max_results=5
    )

    # Extract and format results
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0)
        })

    return results
