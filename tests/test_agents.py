"""Unit tests for agent classes."""

import json
import pytest
from unittest.mock import Mock, MagicMock
from anthropic.types import Message, TextBlock

from agents.base import BaseAgent
from agents.coordinator import CoordinatorAgent


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

        with pytest.raises(ValueError, match="Failed to parse response as JSON"):
            agent.coordinate("Test query")

    def test_coordinate_handles_api_error(self):
        """Test coordinate handles API errors."""
        client = Mock()
        agent = CoordinatorAgent(client, "Test prompt")

        client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Coordination failed"):
            agent.coordinate("Test query")
