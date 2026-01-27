#!/usr/bin/env python3
"""
Simple example demonstrating the multi-agent research workflow.

This example shows how to use the research system programmatically
rather than through the CLI.
"""

from anthropic import Anthropic
from config.settings import load_prompts, get_api_key, get_tavily_api_key
from orchestration.workflow import run_research_workflow


def main() -> None:
    """Run a simple research query."""
    # Load configuration
    api_key = get_api_key()
    tavily_api_key = get_tavily_api_key()
    prompts = load_prompts()

    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)

    # Define research query
    query = "What are the latest developments in quantum computing?"

    print(f"Research Query: {query}")
    print("\nRunning multi-agent research workflow...\n")

    # Run the workflow
    result = run_research_workflow(
        query=query,
        client=client,
        coordinator_prompt=prompts['coordinator'],
        researcher_prompt=prompts['researcher'],
        synthesizer_prompt=prompts['synthesizer'],
        critic_prompt=prompts['critic'],
        tavily_api_key=tavily_api_key
    )

    # Display results
    print("="*80)
    print("RESEARCH COMPLETE")
    print("="*80)

    print(f"\nQuery: {result['query']}")
    print(f"Subtasks generated: {len(result['subtasks'])}")
    print(f"Research sections: {len(result['research_results'])}")

    print("\nSubtasks:")
    for i, subtask in enumerate(result['subtasks'], 1):
        print(f"  {i}. {subtask}")

    print("\nSynthesis Summary:")
    print(f"  {result['synthesis']['summary']}")

    print(f"\nSections: {len(result['synthesis']['sections'])}")
    for section in result['synthesis']['sections']:
        print(f"  - {section['title']}")

    print(f"\nKey Insights: {len(result['synthesis']['key_insights'])}")
    for i, insight in enumerate(result['synthesis']['key_insights'], 1):
        print(f"  {i}. {insight}")

    print("\nCritic Assessment:")
    print(f"  Overall Quality: {result['critique']['overall_quality']}")
    print(f"  Issues Found: {len(result['critique']['issues'])}")
    print(f"  Suggestions: {len(result['critique']['suggestions'])}")
    print(f"  Needs More Research: {result['critique']['needs_more_research']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
