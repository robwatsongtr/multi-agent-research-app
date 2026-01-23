"""Unit tests for agent classes."""

import json
import pytest
from unittest.mock import Mock, MagicMock
from anthropic.types import Message, TextBlock

from agents.base import BaseAgent
from agents.coordinator import CoordinatorAgent
from agents.synthesizer import SynthesizerAgent


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_init(self):
        """Test BaseAgent initialization."""
        # Arrange
        client = Mock()
        system_prompt = "You are a test agent"

        # Act
        agent = BaseAgent(client, system_prompt)

        # Assert
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
            messages=[{"role": "user", "content": "Test message"}]
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
            tools=tools
        )

    def test_parse_response_success(self):
        """Test parse_response extracts text correctly."""
        client = Mock()
        agent = BaseAgent(client, "Test prompt")

        # Create mock message with text content
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Test response text"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]

        result = agent.parse_response(mock_message)

        assert result == "Test response text"

    """
    Testing Exceptions
    """

    def test_parse_response_no_content(self):
        """Test parse_response raises error when no content."""
        client = Mock()
        agent = BaseAgent(client, "Test prompt")

        mock_message = Mock(spec=Message)
        mock_message.content = []

        with pytest.raises(ValueError, match="Message has no content"):
            agent.parse_response(mock_message)

    def test_parse_response_unexpected_type(self):
        """Test parse_response raises error for unexpected content type."""
        client = Mock()
        agent = BaseAgent(client, "Test prompt")

        mock_unknown_block = Mock()
        delattr(mock_unknown_block, 'text')  # Remove text attribute
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_unknown_block]

        with pytest.raises(ValueError, match="Unexpected content block type"):
            agent.parse_response(mock_message)


class TestCoordinatorAgent:
    """Tests for CoordinatorAgent class."""

    def test_init(self):
        """Test CoordinatorAgent initialization."""
        client = Mock()
        system_prompt = "You are a coordinator"

        agent = CoordinatorAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_coordinate_success_with_json_array(self):
        """Test coordinate successfully parses JSON array response."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        # Mock the response
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '["Subtask 1", "Subtask 2", "Subtask 3"]'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.coordinate("Test query")

        assert result == ["Subtask 1", "Subtask 2", "Subtask 3"]

    def test_coordinate_success_strips_markdown(self):
        """Test coordinate strips markdown code blocks."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        # Mock response with markdown
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '```json\n["Subtask 1", "Subtask 2"]\n```'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.coordinate("Test query")

        assert result == ["Subtask 1", "Subtask 2"]

    def test_coordinate_validates_min_subtasks(self):
        """Test coordinate rejects fewer than 2 subtasks."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '["Only one subtask"]'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="Expected 2-4 subtasks, got 1"):
            agent.coordinate("Test query")

    def test_coordinate_validates_max_subtasks(self):
        """Test coordinate rejects more than 4 subtasks."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="Expected 2-4 subtasks, got 5"):
            agent.coordinate("Test query")

    def test_coordinate_validates_string_items(self):
        """Test coordinate rejects non-string items."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = '["Task 1", 123, "Task 3"]'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="All subtasks must be strings"):
            agent.coordinate("Test query")

    def test_coordinate_handles_invalid_json(self):
        """Test coordinate handles invalid JSON gracefully."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = 'Not valid JSON'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Coordination failed"):
            agent.coordinate("Test query")

    def test_coordinate_handles_api_error(self):
        """Test coordinate handles API errors."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Coordination failed"):
            agent.coordinate("Test query")


class TestSynthesizerAgent:
    """Tests for SynthesizerAgent class."""

    def test_init(self):
        """Test SynthesizerAgent initialization."""
        client = Mock()
        system_prompt = "You are a synthesizer"

        agent = SynthesizerAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_synthesize_success_with_valid_json(self):
        """Test synthesize successfully parses valid JSON response."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        # Sample research findings
        findings = [
            {
                "subtask": "Research AI breakthroughs",
                "findings": [
                    {
                        "claim": "GPT-4 released in 2023",
                        "source": "https://example.com/gpt4",
                        "details": "Major language model advancement"
                    }
                ]
            },
            {
                "subtask": "Research quantum computing",
                "findings": [
                    {
                        "claim": "Quantum supremacy achieved",
                        "source": "https://example.com/quantum",
                        "details": "Google's quantum processor"
                    }
                ]
            }
        ]

        # Mock the response
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({
            "summary": "Research on AI and quantum computing advancements",
            "sections": [
                {
                    "title": "AI Developments",
                    "content": "GPT-4 was released in 2023, marking a major advancement.",
                    "sources": ["https://example.com/gpt4"]
                },
                {
                    "title": "Quantum Computing",
                    "content": "Google achieved quantum supremacy.",
                    "sources": ["https://example.com/quantum"]
                }
            ],
            "key_insights": [
                "AI models are rapidly advancing",
                "Quantum computing is becoming practical"
            ]
        })
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.synthesize(findings)

        assert result["summary"] == "Research on AI and quantum computing advancements"
        assert len(result["sections"]) == 2
        assert result["sections"][0]["title"] == "AI Developments"
        assert len(result["key_insights"]) == 2

    def test_synthesize_success_strips_markdown(self):
        """Test synthesize strips markdown code blocks."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test task",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        # Mock response with markdown
        mock_text_block = Mock(spec=TextBlock)
        response_json = {
            "summary": "Test summary",
            "sections": [{"title": "Test", "content": "Content", "sources": []}],
            "key_insights": ["Insight 1"]
        }
        mock_text_block.text = f'```json\n{json.dumps(response_json)}\n```'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.synthesize(findings)

        assert result["summary"] == "Test summary"
        assert len(result["sections"]) == 1

    def test_synthesize_validates_required_fields(self):
        """Test synthesize validates all required fields are present."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        # Test missing summary
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({
            "sections": [],
            "key_insights": []
        })
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Synthesis failed"):
            agent.synthesize(findings)

    def test_synthesize_validates_sections_is_list(self):
        """Test synthesize validates sections is a list."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({
            "summary": "Test",
            "sections": "not a list",
            "key_insights": []
        })
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="sections must be a list"):
            agent.synthesize(findings)

    def test_synthesize_validates_key_insights_is_list(self):
        """Test synthesize validates key_insights is a list."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({
            "summary": "Test",
            "sections": [],
            "key_insights": "not a list"
        })
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="key_insights must be a list"):
            agent.synthesize(findings)

    def test_synthesize_handles_invalid_json(self):
        """Test synthesize handles invalid JSON gracefully."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = 'Not valid JSON'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Synthesis failed"):
            agent.synthesize(findings)

    def test_synthesize_handles_api_error(self):
        """Test synthesize handles API errors."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Synthesis failed"):
            agent.synthesize(findings)

    def test_synthesize_logs_section_count(self):
        """Test synthesize logs the number of sections generated."""
        client = Mock()
        agent = SynthesizerAgent(client, "Test prompt")

        findings = [
            {
                "subtask": "Test",
                "findings": [{"claim": "Test", "source": "test.com", "details": "test"}]
            }
        ]

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({
            "summary": "Test summary",
            "sections": [
                {"title": "Section 1", "content": "Content 1", "sources": []},
                {"title": "Section 2", "content": "Content 2", "sources": []},
                {"title": "Section 3", "content": "Content 3", "sources": []}
            ],
            "key_insights": ["Insight 1"]
        })
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.synthesize(findings)

        assert len(result["sections"]) == 3
