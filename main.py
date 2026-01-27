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
import textwrap
from anthropic import Anthropic

from config.settings import load_prompts, get_api_key, get_model, get_tavily_api_key
from orchestration.workflow import run_research_workflow


def wrap_text(text: str, width: int = 100) -> str:
    """
    Wrap text to a specified width, breaking at word boundaries.

    Args:
        text: Text to wrap
        width: Maximum line width (default: 80)

    Returns:
        Wrapped text with line breaks
    """
    # Preserve paragraphs by splitting on double newlines
    paragraphs = text.split('\n\n')
    wrapped_paragraphs = []

    for para in paragraphs:
        # Remove single newlines within paragraphs
        para = para.replace('\n', ' ')
        # Wrap the paragraph
        wrapped = textwrap.fill(para, width=width, break_long_words=False, break_on_hyphens=False)
        wrapped_paragraphs.append(wrapped)

    return '\n\n'.join(wrapped_paragraphs)


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
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_api_key
        )

        # Display subtasks
        print("\n" + "="*80)
        print("RESEARCH SUBTASKS")
        print("="*80)

        for i, subtask in enumerate(result.subtasks, 1):
            wrapped_subtask = wrap_text(subtask, width=77)
            lines = wrapped_subtask.split('\n')
            print(f"\n{i}. {lines[0]}")
            for line in lines[1:]:
                print(f"   {line}")

        # Display research results
        print("\n" + "="*80)
        print("RESEARCH FINDINGS")
        print("="*80)

        for i, research in enumerate(result.research_results, 1):
            wrapped_task = wrap_text(research.subtask, width=68)
            lines = wrapped_task.split('\n')
            print(f"\n[Subtask {i}]: {lines[0]}")
            for line in lines[1:]:
                print(f"            {line}")
            print(f"Findings: {len(research.findings)}")

            for j, finding in enumerate(research.findings, 1):
                wrapped_claim = wrap_text(finding.claim, width=75)
                claim_lines = wrapped_claim.split('\n')
                print(f"\n  {j}. {claim_lines[0]}")
                for line in claim_lines[1:]:
                    print(f"     {line}")

                wrapped_source = wrap_text(finding.source, width=73)
                source_lines = wrapped_source.split('\n')
                print(f"     Source: {source_lines[0]}")
                for line in source_lines[1:]:
                    print(f"             {line}")

                if finding.details:
                    wrapped_details = wrap_text(finding.details, width=72)
                    details_lines = wrapped_details.split('\n')
                    print(f"     Details: {details_lines[0]}")
                    for line in details_lines[1:]:
                        print(f"              {line}")

        # Display synthesized report
        print("\n" + "="*80)
        print("SYNTHESIZED RESEARCH REPORT")
        print("="*80)

        synthesis = result.synthesis
        print(f"\n{wrap_text(synthesis.summary, width=80)}\n")

        for i, section in enumerate(synthesis.sections, 1):
            print(f"\n{'─'*80}")
            print(f"{section.title}")
            print(f"{'─'*80}")
            print(f"\n{wrap_text(section.content, width=80)}\n")

            if section.sources:
                print("Sources:")
                for source in section.sources:
                    print(f"  • {wrap_text(source, width=76)}")

        print(f"\n{'─'*80}")
        print("KEY INSIGHTS")
        print(f"{'─'*80}\n")

        for i, insight in enumerate(synthesis.key_insights, 1):
            wrapped_insight = wrap_text(insight, width=77)
            # Indent wrapped lines after the first
            lines = wrapped_insight.split('\n')
            print(f"{i}. {lines[0]}")
            for line in lines[1:]:
                print(f"   {line}")

        # Display critique
        print("\n" + "="*80)
        print("CRITIC REVIEW")
        print("="*80)

        critique = result.critique
        wrapped_quality = wrap_text(critique.overall_quality, width=63)
        quality_lines = wrapped_quality.split('\n')
        print(f"\nOverall Quality: {quality_lines[0]}")
        for line in quality_lines[1:]:
            print(f"                 {line}")
        print()

        if critique.issues:
            print(f"{'─'*80}")
            print(f"ISSUES IDENTIFIED ({len(critique.issues)})")
            print(f"{'─'*80}\n")

            for i, issue in enumerate(critique.issues, 1):
                print(f"{i}. [{issue.severity.upper()}] {issue.formatted_type}")
                wrapped_desc = wrap_text(issue.description, width=75)
                desc_lines = wrapped_desc.split('\n')
                print(f"   {desc_lines[0]}")
                for line in desc_lines[1:]:
                    print(f"   {line}")
                print(f"   Location: {issue.location}\n")
        else:
            print("No issues identified.\n")

        if critique.suggestions:
            print(f"{'─'*80}")
            print("SUGGESTIONS FOR IMPROVEMENT")
            print(f"{'─'*80}\n")

            for i, suggestion in enumerate(critique.suggestions, 1):
                wrapped_suggestion = wrap_text(suggestion, width=77)
                lines = wrapped_suggestion.split('\n')
                print(f"{i}. {lines[0]}")
                for line in lines[1:]:
                    print(f"   {line}")
        else:
            print("No suggestions for improvement.\n")

        print(f"\n{'─'*80}")
        if critique.needs_more_research:
            print("⚠️  Critic recommends additional research to address identified gaps.")
        else:
            print("✓ Critic assessment: Research is comprehensive.")
        print(f"{'─'*80}")

        print("\n" + "="*80)
        print(f"Research complete: {len(result.subtasks)} subtasks, {len(synthesis.sections)} sections")
        print("="*80)

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
