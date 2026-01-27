"""
LLM evaluation tests for end-to-end workflow quality.

These tests use real API calls to validate the quality of research outputs.
Run with: pytest tests/evals/ -v -m slow
"""

import pytest
from anthropic import Anthropic

from config.settings import load_prompts, get_api_key, get_tavily_api_key
from orchestration.workflow import run_research_workflow


@pytest.fixture
def api_client():
    """Create Anthropic client for evals."""
    api_key = get_api_key()

    return Anthropic(api_key=api_key)


@pytest.fixture
def tavily_key():
    """Get Tavily API key for evals."""
    return get_tavily_api_key()


@pytest.fixture
def prompts():
    """Load system prompts."""
    return load_prompts()


@pytest.mark.slow
class TestWorkflowEvals:
    """Evaluation tests for full research workflow quality."""

    def test_end_to_end_workflow_completion(self, api_client, tavily_key, prompts):
        """
        Eval: End-to-end workflow should complete successfully.

        Tests that the full pipeline (coordinator → researcher → synthesizer → critic)
        completes without errors and produces all expected outputs.
        """
        query = "What are the main benefits of Python for data science?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        # Verify workflow completed
        assert result is not None
        assert "query" in result
        assert "subtasks" in result
        assert "research_results" in result
        assert "synthesis" in result
        assert "critique" in result

        # Verify basic structure
        assert result["query"] == query
        assert len(result["subtasks"]) > 0
        assert len(result["research_results"]) > 0
        assert "summary" in result["synthesis"]
        assert "sections" in result["synthesis"]
        assert "overall_quality" in result["critique"]

    def test_citation_preservation(self, api_client, tavily_key, prompts):
        """
        Eval: Citations should be preserved from research through synthesis.

        Tests that source URLs found during research appear in the final
        synthesized report sections.
        """
        query = "What are recent developments in quantum computing?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        # Collect all research sources
        research_sources = set()
        for research in result["research_results"]:
            for finding in research["findings"]:
                if "source" in finding and finding["source"]:
                    research_sources.add(finding["source"])

        # Check synthesis includes citations
        synthesis_has_sources = False
        synthesis_sources = set()

        for section in result["synthesis"]["sections"]:
            if "sources" in section and section["sources"]:
                synthesis_has_sources = True
                for source in section["sources"]:
                    synthesis_sources.add(source)

        # Verify citations exist
        assert synthesis_has_sources, "Synthesis should include source citations"
        assert len(synthesis_sources) > 0, "Synthesis should have at least one source"

        # Verify at least some research sources appear in synthesis
        # (Not all research findings may make it to final report, but some should)
        overlap = research_sources.intersection(synthesis_sources)
        assert len(overlap) > 0, "At least some research sources should appear in synthesis"

    def test_reasonable_subtask_count(self, api_client, tavily_key, prompts):
        """
        Eval: Coordinator should generate a reasonable number of subtasks.

        Tests that the coordinator breaks queries into 2-4 subtasks as specified
        in the prompt, avoiding both over-simplification (1 task) and
        over-complication (many tasks).
        """
        # Test with a moderately complex query
        query = "How is artificial intelligence transforming healthcare?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        subtask_count = len(result["subtasks"])

        # Verify reasonable range (2-5 tasks is acceptable, 2-4 is ideal)
        assert 2 <= subtask_count <= 5, (
            f"Expected 2-5 subtasks, got {subtask_count}. "
            f"Subtasks: {result['subtasks']}"
        )

        # Verify each subtask is non-empty and substantive
        for i, subtask in enumerate(result["subtasks"], 1):
            assert len(subtask.strip()) > 10, (
                f"Subtask {i} is too short: '{subtask}'"
            )

        # Verify subtasks are distinct (no duplicates)
        unique_subtasks = set(s.lower().strip() for s in result["subtasks"])
        assert len(unique_subtasks) == len(result["subtasks"]), (
            "Subtasks should be distinct, found duplicates"
        )


@pytest.mark.slow
class TestResearchQualityEvals:
    """Evaluation tests for research output quality."""

    def test_research_findings_have_sources(self, api_client, tavily_key, prompts):
        """
        Eval: Research findings should include source URLs.

        Tests that the researcher agent properly extracts and includes
        source citations for claims.
        """
        query = "What is the current state of electric vehicle adoption?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        # Verify at least some findings have sources
        total_findings = 0
        findings_with_sources = 0

        for research in result["research_results"]:
            for finding in research["findings"]:
                total_findings += 1
                if "source" in finding and finding["source"]:
                    # Verify source looks like a URL
                    assert finding["source"].startswith("http"), (
                        f"Source should be a URL: {finding['source']}"
                    )
                    findings_with_sources += 1

        # At least 80% of findings should have sources
        if total_findings > 0:
            source_percentage = findings_with_sources / total_findings
            assert source_percentage >= 0.8, (
                f"Expected at least 80% of findings to have sources, "
                f"got {source_percentage:.0%} ({findings_with_sources}/{total_findings})"
            )

    def test_synthesis_structure(self, api_client, tavily_key, prompts):
        """
        Eval: Synthesis should be well-structured with sections and insights.

        Tests that the synthesizer creates organized output with clear sections
        and key insights.
        """
        query = "What are the environmental impacts of cryptocurrency mining?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        synthesis = result["synthesis"]

        # Verify summary exists and is substantive
        assert "summary" in synthesis
        assert len(synthesis["summary"]) > 50, "Summary should be substantive"

        # Verify sections exist and have required fields
        assert "sections" in synthesis
        assert len(synthesis["sections"]) > 0, "Should have at least one section"

        for i, section in enumerate(synthesis["sections"], 1):
            assert "title" in section, f"Section {i} missing title"
            assert "content" in section, f"Section {i} missing content"
            assert len(section["content"]) > 50, f"Section {i} content too short"

        # Verify key insights exist
        assert "key_insights" in synthesis
        assert len(synthesis["key_insights"]) > 0, "Should have at least one key insight"

        for i, insight in enumerate(synthesis["key_insights"], 1):
            assert len(insight) > 20, f"Key insight {i} too short: '{insight}'"


@pytest.mark.slow
class TestCriticEvals:
    """Evaluation tests for critic agent quality."""

    def test_critic_provides_assessment(self, api_client, tavily_key, prompts):
        """
        Eval: Critic should provide quality assessment.

        Tests that the critic provides an overall quality assessment
        and either identifies issues or confirms quality.
        """
        query = "What are the main programming languages used in web development?"

        result = run_research_workflow(
            query=query,
            client=api_client,
            coordinator_prompt=prompts['coordinator'],
            researcher_prompt=prompts['researcher'],
            synthesizer_prompt=prompts['synthesizer'],
            critic_prompt=prompts['critic'],
            tavily_api_key=tavily_key
        )

        critique = result["critique"]

        # Verify required fields
        assert "overall_quality" in critique
        assert "issues" in critique
        assert "suggestions" in critique
        assert "needs_more_research" in critique

        # Verify overall_quality is substantive
        assert len(critique["overall_quality"]) > 10, (
            "Overall quality assessment should be substantive"
        )

        # Verify issues are well-formed if present
        if critique["issues"]:
            for issue in critique["issues"]:
                assert "severity" in issue
                assert "type" in issue
                assert "description" in issue
                assert "location" in issue
                assert len(issue["description"]) > 10

        # Verify suggestions are substantive if present
        if critique["suggestions"]:
            for suggestion in critique["suggestions"]:
                assert len(suggestion) > 10, "Suggestions should be substantive"

        # Verify needs_more_research is boolean
        assert isinstance(critique["needs_more_research"], bool)
