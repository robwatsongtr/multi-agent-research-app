#!/usr/bin/env python3
"""
CLI entry point for the multi-agent research system.

Usage:
    python3 main.py "Your research query here"
    python3 main.py "Your research query here" --verbose
"""

import sys
import json
import logging
from anthropic import Anthropic

from config.settings import load_prompts, get_api_key, get_model, get_tavily_api_key
from orchestration.workflow import run_research_workflow


def main() -> None:
    """Main entry point for the CLI."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your research query here\" [--verbose]")
        print("\nExample:")
        print('  python main.py "What are the latest developments in quantum computing?"')
        print('  python main.py "What are the latest developments in quantum computing?" --verbose')
        sys.exit(1)

    query = sys.argv[1]
    verbose = "--verbose" in sys.argv

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        stream=sys.stderr
    )

    try:
        # Load configuration
        print("Loading configuration...")
        api_key = get_api_key()
        tavily_api_key = get_tavily_api_key()
        model = get_model()
        prompts = load_prompts()

        # Initialize Anthropic client
        client = Anthropic(api_key=api_key)

        # Run full research workflow
        print(f"\nResearch Query: {query}")
        print("\nRunning research workflow...")

        result = run_research_workflow(
            query=query,
            client=client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            tavily_api_key=tavily_api_key
        )

        # Display subtasks
        print("\n" + "="*60)
        print("RESEARCH SUBTASKS")
        print("="*60)

        for i, subtask in enumerate(result['subtasks'], 1):
            print(f"\n{i}. {subtask}")

        # Display research results
        print("\n" + "="*60)
        print("RESEARCH FINDINGS")
        print("="*60)

        for i, research in enumerate(result['research_results'], 1):
            print(f"\n[Subtask {i}]: {research['subtask']}")
            print(f"Findings: {len(research['findings'])}")

            for j, finding in enumerate(research['findings'], 1):
                print(f"\n  {j}. {finding['claim']}")
                print(f"     Source: {finding['source']}")
                if finding.get('details'):
                    print(f"     Details: {finding['details']}")

        # Display synthesized report
        print("\n" + "="*60)
        print("SYNTHESIZED RESEARCH REPORT")
        print("="*60)

        synthesis = result['synthesis']
        print(f"\n{synthesis['summary']}\n")

        for i, section in enumerate(synthesis['sections'], 1):
            print(f"\n{'─'*60}")
            print(f"{section['title']}")
            print(f"{'─'*60}")
            print(f"\n{section['content']}\n")

            if section.get('sources'):
                print("Sources:")
                for source in section['sources']:
                    print(f"  • {source}")

        print(f"\n{'─'*60}")
        print("KEY INSIGHTS")
        print(f"{'─'*60}\n")

        for i, insight in enumerate(synthesis['key_insights'], 1):
            print(f"{i}. {insight}")

        print("\n" + "="*60)
        print(f"Research complete: {len(result['subtasks'])} subtasks, {len(synthesis['sections'])} sections")
        print("="*60)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
