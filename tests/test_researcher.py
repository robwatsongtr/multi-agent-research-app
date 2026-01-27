"""Unit tests for ResearcherAgent with Pydantic models."""

import json
import pytest
from unittest.mock import Mock
from anthropic.types import Message, TextBlock, ToolUseBlock

from agents.researcher import ResearcherAgent
from agents.models import ResearchResult, Finding


class TestResearcherAgent:
    """Tests for ResearcherAgent class with Pydantic models."""

    def test_init(self):
        """Test ResearcherAgent initialization."""
        client = Mock()
        system_prompt = "You are a researcher"

        agent = ResearcherAgent(client, system_prompt)

        assert agent.client == client
        assert agent.system_prompt == system_prompt

    def test_research_success(self):
        """Test research successfully returns ResearchResult."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock response
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {
                "subtask": "Research AI",
                "findings": [
                    {
                        "claim": "GPT-4 released in 2023",
                        "source": "https://example.com/gpt4",
                        "details": "Major language model advancement",
                    }
                ],
            }
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.research("Research AI breakthroughs")

        # Assert Pydantic model
        assert isinstance(result, ResearchResult)
        assert result.subtask == "Research AI"
        assert len(result.findings) == 1
        assert isinstance(result.findings[0], Finding)
        assert result.findings[0].claim == "GPT-4 released in 2023"
        assert result.findings[0].source == "https://example.com/gpt4"

    def test_research_strips_markdown(self):
        """Test research handles markdown code blocks."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        response_json = {
            "subtask": "Test task",
            "findings": [
                {"claim": "Test claim", "source": "test.com", "details": "Test details"}
            ],
        }
        mock_text_block.text = f"```json\n{json.dumps(response_json)}\n```"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        result = agent.research("Test task")

        assert result.subtask == "Test task"
        assert len(result.findings) == 1

    def test_research_with_tools(self):
        """Test research can be called with tools."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock response without tool use
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {
                "subtask": "Test",
                "findings": [
                    {"claim": "Test", "source": "test.com", "details": "Test"}
                ],
            }
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        tools = [{"name": "web_search", "description": "Search the web"}]
        tool_executor = Mock(return_value=[])

        result = agent.research("Test", tools=tools, tool_executor=tool_executor)

        assert isinstance(result, ResearchResult)
        # Verify tools were passed to API
        call_args = client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools

    def test_research_with_tool_use(self):
        """Test research handles tool use in response."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # First response: Claude wants to use a tool
        mock_tool_block = Mock(spec=ToolUseBlock)
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "web_search"
        mock_tool_block.input = {"query": "AI developments"}
        mock_tool_block.id = "tool_123"

        mock_message_1 = Mock(spec=Message)
        mock_message_1.content = [mock_tool_block]
        mock_message_1.stop_reason = "tool_use"

        # Second response: Final answer with findings
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {
                "subtask": "Research AI",
                "findings": [
                    {"claim": "Found via tool", "source": "tool.com", "details": "Data"}
                ],
            }
        )
        mock_message_2 = Mock(spec=Message)
        mock_message_2.content = [mock_text_block]
        mock_message_2.stop_reason = "end_turn"

        client.messages.create.side_effect = [mock_message_1, mock_message_2]

        tools = [{"name": "web_search"}]
        tool_executor = Mock(return_value=[{"title": "Result", "url": "url.com"}])

        result = agent.research("Research AI", tools=tools, tool_executor=tool_executor)

        # Verify tool was executed
        tool_executor.assert_called_once_with("web_search", {"query": "AI developments"})
        # Verify final result
        assert result.subtask == "Research AI"
        assert result.findings[0].claim == "Found via tool"

    def test_research_validates_findings_list(self):
        """Test research validates findings is a list through Pydantic."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock response with invalid findings (not a list)
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps(
            {"subtask": "Test", "findings": "not a list"}
        )
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("Test task")

    def test_research_validates_min_findings(self):
        """Test research validates at least one finding through Pydantic."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        # Mock response with empty findings list
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = json.dumps({"subtask": "Test", "findings": []})
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("Test task")

    def test_research_handles_api_error(self):
        """Test research handles API errors gracefully."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("Test task")

    def test_research_handles_invalid_json(self):
        """Test research handles invalid JSON responses."""
        client = Mock()
        agent = ResearcherAgent(client, "Test prompt")

        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Not valid JSON at all"
        mock_message = Mock(spec=Message)
        mock_message.content = [mock_text_block]
        mock_message.stop_reason = "end_turn"
        client.messages.create.return_value = mock_message

        with pytest.raises(RuntimeError, match="Research failed"):
            agent.research("Test task")
