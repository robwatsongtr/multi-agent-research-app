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
        mock_message.stop_reason = "end_turn"
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
        mock_message.stop_reason = "end_turn"
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
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Research failed"):
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
        mock_message.stop_reason = "end_turn"
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
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Research failed"):
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
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        tools = [{"name": "web_search", "description": "Search the web"}]

        result = agent.research("Test subtask", tools=tools)

        # Verify tools were passed to the API call
        call_args = client.messages.create.call_args
        assert call_args[1]["tools"] == tools

    def test_research_with_web_search_tool_executor(self):
        """Test research with web_search tool executor."""
        from tools.web_search import WEB_SEARCH_TOOL

        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Create mock tool executor
        tool_executor = Mock()
        tool_executor.return_value = [
            {
                "title": "AI Development Guide",
                "url": "https://example.com/ai",
                "content": "Guide to AI development...",
                "score": 0.95
            }
        ]

        response_json = {
            "subtask": "AI development tools",
            "findings": [
                {
                    "claim": "AI tools are improving developer productivity",
                    "source": "https://example.com/ai",
                    "details": "Based on recent surveys"
                }
            ]
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.research(
            "AI development tools",
            tools=[WEB_SEARCH_TOOL],
            tool_executor=tool_executor
        )

        assert result["subtask"] == "AI development tools"
        assert len(result["findings"]) == 1

    def test_research_tool_executor_called_on_tool_use(self):
        """Test that tool_executor is called when Claude requests tool use."""
        from tools.web_search import WEB_SEARCH_TOOL
        from anthropic.types import ToolUseBlock

        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Create mock tool executor
        tool_executor = Mock()
        tool_executor.return_value = [
            {
                "title": "Test Result",
                "url": "https://test.com",
                "content": "Test content",
                "score": 0.9
            }
        ]

        # First response: Claude wants to use web_search
        mock_tool_use_block = Mock(spec=ToolUseBlock)
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.name = "web_search"
        mock_tool_use_block.input = {"query": "AI development"}
        mock_tool_use_block.id = "tool_123"

        mock_tool_use_message = Mock(spec=Message)
        mock_tool_use_message.content = [mock_tool_use_block]
        mock_tool_use_message.stop_reason = "tool_use"

        # Second response: Claude returns final answer
        response_json = {
            "subtask": "AI development",
            "findings": [
                {
                    "claim": "Test claim",
                    "source": "https://test.com",
                    "details": "Test details"
                }
            ]
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_final_message = Mock(spec=Message)
        mock_final_message.content = [mock_text_block]
        mock_final_message.stop_reason = "end_turn"

        client.messages.create.side_effect = [
            mock_tool_use_message,
            mock_final_message
        ]

        result = agent.research(
            "AI development",
            tools=[WEB_SEARCH_TOOL],
            tool_executor=tool_executor
        )

        # Verify tool_executor was called with correct parameters
        tool_executor.assert_called_once_with("web_search", {"query": "AI development"})

        # Verify final result
        assert result["subtask"] == "AI development"
        assert len(result["findings"]) == 1

    def test_research_handles_tool_executor_error(self):
        """Test research handles errors from tool_executor."""
        from tools.web_search import WEB_SEARCH_TOOL
        from anthropic.types import ToolUseBlock

        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Tool executor that raises an error
        tool_executor = Mock()
        tool_executor.side_effect = Exception("API rate limit exceeded")

        # First response: Claude wants to use web_search
        mock_tool_use_block = Mock(spec=ToolUseBlock)
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.name = "web_search"
        mock_tool_use_block.input = {"query": "test"}
        mock_tool_use_block.id = "tool_123"

        mock_tool_use_message = Mock(spec=Message)
        mock_tool_use_message.content = [mock_tool_use_block]
        mock_tool_use_message.stop_reason = "tool_use"

        # Second response: Claude handles the error
        response_json = {
            "subtask": "test",
            "findings": []
        }

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(response_json)
        mock_final_message = Mock(spec=Message)
        mock_final_message.content = [mock_text_block]
        mock_final_message.stop_reason = "end_turn"

        client.messages.create.side_effect = [
            mock_tool_use_message,
            mock_final_message
        ]

        result = agent.research(
            "test",
            tools=[WEB_SEARCH_TOOL],
            tool_executor=tool_executor
        )

        # Should still return a result even though tool failed
        assert result["subtask"] == "test"
        assert result["findings"] == []

    def test_research_without_tool_executor_raises_error(self):
        """Test research raises error if tools provided but no executor."""
        from tools.web_search import WEB_SEARCH_TOOL
        from anthropic.types import ToolUseBlock

        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock Claude requesting tool use
        mock_tool_use_block = Mock(spec=ToolUseBlock)
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.name = "web_search"
        mock_tool_use_block.input = {"query": "test"}
        mock_tool_use_block.id = "tool_123"

        mock_tool_use_message = Mock(spec=Message)
        mock_tool_use_message.content = [mock_tool_use_block]
        mock_tool_use_message.stop_reason = "tool_use"

        client.messages.create.return_value = mock_tool_use_message

        # Research with tools but no tool_executor should raise error (wrapped in RuntimeError)
        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("test", tools=[WEB_SEARCH_TOOL])
