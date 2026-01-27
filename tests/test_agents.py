"""Unit tests for agent classes with Pydantic models."""

import json
import pytest
from unittest.mock import Mock
from anthropic.types import Message, TextBlock

from agents.base import BaseAgent
from agents.coordinator import CoordinatorAgent
from agents.synthesizer import SynthesizerAgent
from agents.critic import CriticAgent
from agents.models import (
    Finding,
    ResearchResult,
    SynthesizedReport,
    SynthesisSection,
    CriticReview,
    CriticIssue,
)


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_init(self):
        """Test BaseAgent initialization."""
        client = Mock()
        system_prompt = "You are a test agent"

        agent = BaseAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt
        assert agent.model == "claude-sonnet-4-5-20250929"

    def test_init_with_custom_model(self):
        """Test BaseAgent initialization with custom model."""
        client = Mock()
        system_prompt = "You are a test agent"
        custom_model = "claude-3-opus-20240229"

        agent = BaseAgent(client, system_prompt, model=custom_model)

        assert agent.model == custom_model

    def test_call_claude_without_tools(self):
        """Test call_claude makes correct API call without tools."""
        client = Mock()
        mock_response = Mock(spec=Message)
        mock_response.stop_reason = "end_turn"
        client.messages.create.return_value = mock_response

        agent = BaseAgent(client, "Test prompt")

        response = agent.call_claude("Test message")

        assert response == mock_response
        client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=1.0,
            system="Test prompt",
            messages=[{"role": "user", "content": "Test message"}],
        )

    def test_call_claude_with_tools(self):
        """Test call_claude makes correct API call with tools."""
        client = Mock()
        mock_response = Mock(spec=Message)
        mock_response.stop_reason = "end_turn"
        client.messages.create.return_value = mock_response

        agent = BaseAgent(client, "Test prompt")
        tools = [{"name": "test_tool", "description": "A test tool"}]

        response = agent.call_claude("Test message", tools=tools)

        assert response == mock_response
        client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=1.0,
            system="Test prompt",
            messages=[{"role": "user", "content": "Test message"}],
            tools=tools,
        )

    def test_parse_response_success(self):
        """Test parse_response extracts text correctly."""
        client = Mock()
        agent = BaseAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Test response text"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]

        result = agent.parse_response(mock_message)

        assert result == "Test response text"

    def test_parse_response_no_content(self):
        """Test parse_response raises error when no content."""
        client = Mock()
        agent = BaseAgent(client, "Test prompt")

        mock_message = Mock(spec=Message)
        mock_message.content = []

        with pytest.raises(ValueError, match="Message has no content"):
            agent.parse_response(mock_message)


class TestCoordinatorAgent:
    """Tests for CoordinatorAgent class with Pydantic models."""

    def test_init(self):
        """Test CoordinatorAgent initialization."""
        client = Mock()
        system_prompt = "You are a coordinator"

        agent = CoordinatorAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_coordinate_success(self):
        """Test coordinate successfully returns list of subtasks."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        # Mock response with proper JSON structure for Pydantic
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {"subtasks": ["Subtask 1", "Subtask 2", "Subtask 3"]}
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.coordinate("Test query")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == ["Subtask 1", "Subtask 2", "Subtask 3"]

    def test_coordinate_strips_markdown(self):
        """Test coordinate strips markdown code blocks."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '```json\n{"subtasks": ["Task 1", "Task 2"]}\n```'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.coordinate("Test query")

        assert result == ["Task 1", "Task 2"]

    def test_coordinate_validates_min_subtasks(self):
        """Test coordinate validates minimum subtasks through Pydantic."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({"subtasks": ["Only one"]})
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Coordination failed"):
            agent.coordinate("Test query")

    def test_coordinate_validates_max_subtasks(self):
        """Test coordinate validates maximum subtasks through Pydantic."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {"subtasks": ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]}
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Coordination failed"):
            agent.coordinate("Test query")


class TestSynthesizerAgent:
    """Tests for SynthesizerAgent class with Pydantic models."""

    def test_init(self):
        """Test SynthesizerAgent initialization."""
        client = Mock()
        system_prompt = "You are a synthesizer"

        agent = SynthesizerAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_synthesize_success(self):
        """Test synthesize successfully returns SynthesizedReport."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        # Create Pydantic model input
        findings = [
            ResearchResult(
                subtask="Research AI",
                findings=[
                    Finding(
                        claim="GPT-4 released",
                        source="https://example.com",
                        details="Details",
                    )
                ],
            )
        ]

        # Mock response
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {
                "summary": "AI research summary",
                "sections": [
                    {
                        "title": "AI Section",
                        "content": "Content here",
                        "sources": ["https://example.com"],
                    }
                ],
                "key_insights": ["Insight 1", "Insight 2"],
            }
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.synthesize(findings)

        # Assert Pydantic model attributes
        assert isinstance(result, SynthesizedReport)
        assert result.summary == "AI research summary"
        assert len(result.sections) == 1
        assert result.sections[0].title == "AI Section"
        assert len(result.key_insights) == 2

    def test_synthesize_strips_markdown(self):
        """Test synthesize handles markdown code blocks."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            ResearchResult(
                subtask="Test",
                findings=[Finding(claim="Test", source="test.com", details="test")],
            )
        ]

        mock_text_block = Mock(spec=TextBlock)
        response_json = {
            "summary": "Test summary",
            "sections": [{"title": "Test", "content": "Content", "sources": []}],
            "key_insights": ["Insight 1"],
        }
        mock_text_block.text = f"```json\n{json.dumps(response_json)}\n```"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.synthesize(findings)

        assert result.summary == "Test summary"
        assert len(result.sections) == 1


class TestCriticAgent:
    """Tests for CriticAgent class with Pydantic models."""

    def test_init(self):
        """Test CriticAgent initialization."""
        client = Mock()
        system_prompt = "You are a critic"

        agent = CriticAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_review_success(self):
        """Test review successfully returns CriticReview."""
        client = Mock()
        agent = CriticAgent(client, "Test prompt")

        # Create Pydantic model input
        report = SynthesizedReport(
            summary="Test summary",
            sections=[
                SynthesisSection(title="Section 1", content="Content", sources=[])
            ],
            key_insights=["Insight 1"],
        )

        # Mock response
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {
                "overall_quality": "Good",
                "issues": [
                    {
                        "type": "gap",
                        "description": "Missing info",
                        "location": "Section 1",
                        "severity": "medium",
                    }
                ],
                "suggestions": ["Add more sources"],
                "needs_more_research": True,
            }
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.review(report)

        # Assert Pydantic model attributes
        assert isinstance(result, CriticReview)
        assert result.overall_quality == "Good"
        assert len(result.issues) == 1
        assert result.issues[0].type == "gap"
        assert result.issues[0].severity == "medium"
        assert len(result.suggestions) == 1
        assert result.needs_more_research is True

    def test_review_formatted_type_property(self):
        """Test CriticIssue.formatted_type property."""
        issue = CriticIssue(
            type="unsupported_claim",
            description="Test",
            location="Test",
            severity="high",
        )

        assert issue.formatted_type == "Unsupported Claim"

    def test_review_strips_markdown(self):
        """Test review handles markdown code blocks."""
        client = Mock()
        agent = CriticAgent(client, "Test prompt")

        report = SynthesizedReport(
            summary="Test",
            sections=[SynthesisSection(title="Test", content="Test", sources=[])],
            key_insights=[],
        )

        mock_text_block = Mock(spec=TextBlock)
        response_json = {
            "overall_quality": "Good",
            "issues": [],
            "suggestions": [],
            "needs_more_research": False,
        }
        mock_text_block.text = f"```json\n{json.dumps(response_json)}\n```"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.review(report)

        assert result.overall_quality == "Good"
        assert len(result.issues) == 0
        assert result.needs_more_research is False
