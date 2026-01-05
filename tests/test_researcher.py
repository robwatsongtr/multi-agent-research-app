"""Unit tests for ResearcherAgent."""

import json
import pytest
from unittest.mock import Mock
from anthropic.types import Message, TextBlock

from agents.researcher import ResearcherAgent


class TestResearcherAgent:
    """Tests for ResearcherAgent class."""

    def test_init(self):
        """Test ResearcherAgent initialization."""
        client = Mock()
        system_prompt = "You are a researcher"

        agent = ResearcherAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_research_success_with_json_object(self):
        """Test research successfully parses JSON object response."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock response with research findings
        response_json = {
            "subtask": "AI coding assistants",
            "findings": [
                {
                    "claim": "GitHub Copilot has 1M+ users",
                    "source": "https://example.com",
                    "details": "As of 2024"
                },
                {
                    "claim": "AI tools increase productivity 30%",
                    "source": "https://study.com",
                    "details": "Developer survey results"
                }
            ]
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        result = agent.research("AI coding assistants")

        assert result["subtask"] == "AI coding assistants"
        assert len(result["findings"]) == 2
        assert result["findings"][0]["claim"] == "GitHub Copilot has 1M+ users"
        assert result["findings"][0]["source"] == "https://example.com"

    def test_research_strips_markdown(self):
        """Test research strips markdown code blocks."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        response_json = {
            "subtask": "Test subtask",
            "findings": [
                {
                    "claim": "Test claim",
                    "source": "https://test.com",
                    "details": "Test details"
                }
            ]
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = f'```json\n{json.dumps(response_json)}\n```'
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        result = agent.research("Test subtask")

        assert result["subtask"] == "Test subtask"
        assert len(result["findings"]) == 1

    def test_research_validates_required_fields(self):
        """Test research validates required fields are present."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Missing "findings" field
        response_json = {
            "subtask": "Test subtask"
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="missing required field"):
            agent.research("Test subtask")

    def test_research_validates_findings_is_list(self):
        """Test research validates findings is a list."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        response_json = {
            "subtask": "Test subtask",
            "findings": "not a list"
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="findings must be a list"):
            agent.research("Test subtask")

    def test_research_handles_invalid_json(self):
        """Test research handles invalid JSON gracefully."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Not valid JSON"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="Failed to parse response as JSON"):
            agent.research("Test subtask")

    def test_research_handles_api_error(self):
        """Test research handles API errors."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("Test subtask")

    def test_research_calls_claude_with_tools(self):
        """Test research passes tools parameter to call_claude."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        response_json = {
            "subtask": "Test",
            "findings": []
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message

        tools = [{"name": "web_search", "description": "Search the web"}]

        result = agent.research("Test subtask", tools=tools)

        # Verify tools were passed to the API call
        call_args = client.messages.create.call_args
        assert call_args[1]["tools"] == tools
