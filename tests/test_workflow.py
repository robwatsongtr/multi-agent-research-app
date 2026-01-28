"""Integration tests for workflow orchestration with Pydantic models."""

import json
import pytest
from unittest.mock import Mock, patch
from anthropic.types import Message, TextBlock

from orchestration.workflow import run_research_workflow
from agents.models import WorkflowResult, ResearchResult, SynthesizedReport, CriticReview, SearchResult


class TestWorkflow:
    """Tests for research workflow integration with Pydantic models."""

    @patch("orchestration.workflow.execute_web_search")
    def test_run_research_workflow_success(self, mock_web_search):
        """Test complete workflow returns WorkflowResult."""
        client = Mock()

        # Mock web search tool - returns SearchResult Pydantic models
        mock_web_search.return_value = [
            SearchResult(title="Result", url="https://example.com", content="Content", score=0.9)
        ]

        # Mock coordinator response
        mock_coord_text = Mock(spec=TextBlock)
        mock_coord_text.text = json.dumps(
            {"subtasks": ["Subtask 1", "Subtask 2"]}
        )
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_text]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher responses (2 subtasks)
        mock_research_text_1 = Mock(spec=TextBlock)
        mock_research_text_1.text = json.dumps(
            {
                "subtask": "Subtask 1",
                "findings": [
                    {"claim": "Claim 1", "source": "source1.com", "details": "Details 1"}
                ],
            }
        )
        mock_research_message_1 = Mock(spec=Message)
        mock_research_message_1.content = [mock_research_text_1]
        mock_research_message_1.stop_reason = "end_turn"

        mock_research_text_2 = Mock(spec=TextBlock)
        mock_research_text_2.text = json.dumps(
            {
                "subtask": "Subtask 2",
                "findings": [
                    {"claim": "Claim 2", "source": "source2.com", "details": "Details 2"}
                ],
            }
        )
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_text_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer response
        mock_synth_text = Mock(spec=TextBlock)
        mock_synth_text.text = json.dumps(
            {
                "summary": "Research summary",
                "sections": [
                    {
                        "title": "Section 1",
                        "content": "Section content",
                        "sources": ["source1.com"],
                    }
                ],
                "key_insights": ["Insight 1", "Insight 2"],
            }
        )
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_text]
        mock_synth_message.stop_reason = "end_turn"

        # Mock critic response
        mock_critic_text = Mock(spec=TextBlock)
        mock_critic_text.text = json.dumps(
            {
                "overall_quality": "Good",
                "issues": [],
                "suggestions": ["Add more sources"],
                "needs_more_research": False,
            }
        )
        mock_critic_message = Mock(spec=Message)
        mock_critic_message.content = [mock_critic_text]
        mock_critic_message.stop_reason = "end_turn"

        # Set up API call sequence
        client.messages.create.side_effect = [
            mock_coord_message,
            mock_research_message_1,
            mock_research_message_2,
            mock_synth_message,
            mock_critic_message,
        ]

        # Run workflow
        result = run_research_workflow(
            query="Test query",
            client=client,
            coordinator_prompt="Coordinator prompt",
            researcher_prompt="Researcher prompt with {current_date}",
            synthesizer_prompt="Synthesizer prompt",
            critic_prompt="Critic prompt",
            tavily_api_key="test_key",
        )

        # Assert WorkflowResult structure
        assert isinstance(result, WorkflowResult)
        assert result.query == "Test query"
        assert len(result.subtasks) == 2
        assert len(result.research_results) == 2
        assert isinstance(result.research_results[0], ResearchResult)
        assert isinstance(result.synthesis, SynthesizedReport)
        assert isinstance(result.critique, CriticReview)

        # Verify research results
        assert result.research_results[0].subtask == "Subtask 1"
        assert result.research_results[1].subtask == "Subtask 2"

        # Verify synthesis
        assert result.synthesis.summary == "Research summary"
        assert len(result.synthesis.sections) == 1

        # Verify critique
        assert result.critique.overall_quality == "Good"
        assert result.critique.needs_more_research is False

    @patch("orchestration.workflow.execute_web_search")
    def test_run_research_workflow_with_tool_use(self, mock_web_search):
        """Test workflow handles tool use in researcher."""
        client = Mock()

        # Mock web search - returns SearchResult Pydantic models
        mock_web_search.return_value = [
            SearchResult(title="Search Result", url="https://test.com", content="Data", score=0.85)
        ]

        # Mock coordinator
        mock_coord_text = Mock(spec=TextBlock)
        mock_coord_text.text = json.dumps({"subtasks": ["Research task", "Analysis task"]})
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_text]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher with tool use (for first subtask)
        from anthropic.types import ToolUseBlock

        mock_tool_block = Mock(spec=ToolUseBlock)
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "web_search"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_block.id = "tool_123"

        mock_tool_message = Mock(spec=Message)
        mock_tool_message.content = [mock_tool_block]
        mock_tool_message.stop_reason = "tool_use"

        # Researcher final response after tool use
        mock_research_text = Mock(spec=TextBlock)
        mock_research_text.text = json.dumps(
            {
                "subtask": "Research task",
                "findings": [
                    {"claim": "Found via search", "source": "test.com", "details": "Info"}
                ],
            }
        )
        mock_research_message = Mock(spec=Message)
        mock_research_message.content = [mock_research_text]
        mock_research_message.stop_reason = "end_turn"

        # Mock second researcher (no tool use)
        mock_research_text_2 = Mock(spec=TextBlock)
        mock_research_text_2.text = json.dumps(
            {
                "subtask": "Analysis task",
                "findings": [
                    {"claim": "Analysis result", "source": "analysis.com", "details": "Data"}
                ],
            }
        )
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_text_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer
        mock_synth_text = Mock(spec=TextBlock)
        mock_synth_text.text = json.dumps(
            {
                "summary": "Summary",
                "sections": [{"title": "S1", "content": "C1", "sources": []}],
                "key_insights": ["I1"],
            }
        )
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_text]
        mock_synth_message.stop_reason = "end_turn"

        # Mock critic
        mock_critic_text = Mock(spec=TextBlock)
        mock_critic_text.text = json.dumps(
            {
                "overall_quality": "Good",
                "issues": [],
                "suggestions": [],
                "needs_more_research": False,
            }
        )
        mock_critic_message = Mock(spec=Message)
        mock_critic_message.content = [mock_critic_text]
        mock_critic_message.stop_reason = "end_turn"

        client.messages.create.side_effect = [
            mock_coord_message,
            mock_tool_message,
            mock_research_message,  # After tool use
            mock_research_message_2,
            mock_synth_message,
            mock_critic_message,
        ]

        result = run_research_workflow(
            query="Test",
            client=client,
            coordinator_prompt="Coord",
            researcher_prompt="Research {current_date}",
            synthesizer_prompt="Synth",
            critic_prompt="Critic",
            tavily_api_key="key",
        )

        # Verify tool was executed
        mock_web_search.assert_called_once_with("test query", "key")

        # Verify workflow completed
        assert isinstance(result, WorkflowResult)
        assert len(result.research_results) == 2
        assert result.research_results[0].findings[0].claim == "Found via search"

    def test_workflow_result_structure(self):
        """Test WorkflowResult Pydantic model validation."""
        from agents.models import Finding, SynthesisSection, CriticIssue

        # Create a valid WorkflowResult
        result = WorkflowResult(
            query="Test query",
            subtasks=["Task 1", "Task 2"],
            research_results=[
                ResearchResult(
                    subtask="Task 1",
                    findings=[
                        Finding(claim="Claim", source="source.com", details="Details")
                    ],
                )
            ],
            synthesis=SynthesizedReport(
                summary="Summary",
                sections=[
                    SynthesisSection(title="Section", content="Content", sources=[])
                ],
                key_insights=["Insight"],
            ),
            critique=CriticReview(
                overall_quality="Good",
                issues=[],
                suggestions=[],
                needs_more_research=False,
            ),
        )

        # Verify Pydantic model works
        assert result.query == "Test query"
        assert len(result.subtasks) == 2
        assert isinstance(result.research_results[0], ResearchResult)
        assert isinstance(result.synthesis, SynthesizedReport)
        assert isinstance(result.critique, CriticReview)

        # Verify we can convert to dict
        result_dict = result.model_dump()
        assert result_dict["query"] == "Test query"
        assert len(result_dict["subtasks"]) == 2
