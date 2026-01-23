"""Integration tests for workflow orchestration."""

import json
import pytest
from unittest.mock import Mock
from anthropic.types import Message, TextBlock

from orchestration.workflow import run_research_workflow


class TestWorkflow:
    """Tests for research workflow orchestration."""

    def test_run_research_workflow_success(self):
        """Test full workflow: query → subtasks → research findings."""
        client = Mock()

        # Mock coordinator response (returns 2 subtasks)
        coordinator_response = '["Subtask 1", "Subtask 2"]'
        mock_coord_block = Mock(spec=TextBlock)
        mock_coord_block.text = coordinator_response
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_block]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher responses (one for each subtask)
        researcher_response_1 = json.dumps({
            "subtask": "Subtask 1",
            "findings": [
                {
                    "claim": "Finding 1",
                    "source": "https://source1.com",
                    "details": "Details 1"
                }
            ]
        })

        researcher_response_2 = json.dumps({
            "subtask": "Subtask 2",
            "findings": [
                {
                    "claim": "Finding 2",
                    "source": "https://source2.com",
                    "details": "Details 2"
                }
            ]
        })

        mock_research_block_1 = Mock(spec=TextBlock)
        mock_research_block_1.text = researcher_response_1
        mock_research_message_1 = Mock(spec=Message)
        mock_research_message_1.content = [mock_research_block_1]
        mock_research_message_1.stop_reason = "end_turn"

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer response
        synthesizer_response = json.dumps({
            "summary": "Test synthesis summary",
            "sections": [
                {
                    "title": "Section 1",
                    "content": "Content from findings",
                    "sources": ["https://source1.com", "https://source2.com"]
                }
            ],
            "key_insights": ["Insight 1", "Insight 2"]
        })

        mock_synth_block = Mock(spec=TextBlock)
        mock_synth_block.text = synthesizer_response
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_block]
        mock_synth_message.stop_reason = "end_turn"

        # Return coordinator response first, then researcher responses, then synthesizer
        client.messages.create.side_effect = [
            mock_coord_message,
            mock_research_message_1,
            mock_research_message_2,
            mock_synth_message
        ]

        # Run workflow
        result = run_research_workflow(
            query="Test query",
            client=client,
            coordinator_prompt="Coordinator prompt",
            researcher_prompt="Researcher prompt",
            synthesizer_prompt="Synthesizer prompt",
            tavily_api_key="test_api_key"
        )

        # Verify structure
        assert result["query"] == "Test query"
        assert result["subtasks"] == ["Subtask 1", "Subtask 2"]
        assert len(result["research_results"]) == 2

        # Verify first research result
        assert result["research_results"][0]["subtask"] == "Subtask 1"
        assert len(result["research_results"][0]["findings"]) == 1
        assert result["research_results"][0]["findings"][0]["claim"] == "Finding 1"

        # Verify second research result
        assert result["research_results"][1]["subtask"] == "Subtask 2"
        assert len(result["research_results"][1]["findings"]) == 1
        assert result["research_results"][1]["findings"][0]["claim"] == "Finding 2"

        # Verify synthesis
        assert result["synthesis"]["summary"] == "Test synthesis summary"
        assert len(result["synthesis"]["sections"]) == 1
        assert len(result["synthesis"]["key_insights"]) == 2

        # Verify API was called 4 times (1 coordinator + 2 researcher + 1 synthesizer)
        assert client.messages.create.call_count == 4

    def test_run_research_workflow_with_tools(self):
        """Test workflow passes tools to researcher."""
        from tools.web_search import WEB_SEARCH_TOOL

        client = Mock()

        # Mock coordinator response (2 subtasks minimum)
        coordinator_response = '["Subtask 1", "Subtask 2"]'
        mock_coord_block = Mock(spec=TextBlock)
        mock_coord_block.text = coordinator_response
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_block]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher responses (one for each subtask)
        researcher_response_1 = json.dumps({
            "subtask": "Subtask 1",
            "findings": []
        })
        researcher_response_2 = json.dumps({
            "subtask": "Subtask 2",
            "findings": []
        })

        mock_research_block_1 = Mock(spec=TextBlock)
        mock_research_block_1.text = researcher_response_1
        mock_research_message_1 = Mock(spec=Message)
        mock_research_message_1.content = [mock_research_block_1]
        mock_research_message_1.stop_reason = "end_turn"

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer response
        synthesizer_response = json.dumps({
            "summary": "Test",
            "sections": [],
            "key_insights": []
        })
        mock_synth_block = Mock(spec=TextBlock)
        mock_synth_block.text = synthesizer_response
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_block]
        mock_synth_message.stop_reason = "end_turn"

        client.messages.create.side_effect = [
            mock_coord_message,
            mock_research_message_1,
            mock_research_message_2,
            mock_synth_message
        ]

        result = run_research_workflow(
            query="Test query",
            client=client,
            coordinator_prompt="Coordinator prompt",
            researcher_prompt="Researcher prompt",
            synthesizer_prompt="Synthesizer prompt",
            tavily_api_key="test_api_key"
        )

        # Verify WEB_SEARCH_TOOL was passed to researcher calls
        researcher_call_1 = client.messages.create.call_args_list[1]
        researcher_call_2 = client.messages.create.call_args_list[2]
        assert researcher_call_1[1]["tools"] == [WEB_SEARCH_TOOL]
        assert researcher_call_2[1]["tools"] == [WEB_SEARCH_TOOL]

    def test_run_research_workflow_with_web_search_tool(self):
        """Test workflow with web_search tool integration."""
        from unittest.mock import patch
        from anthropic.types import ToolUseBlock

        client = Mock()

        # Mock coordinator response (2 subtasks)
        coordinator_response = '["AI coding tools", "Developer productivity"]'
        mock_coord_block = Mock(spec=TextBlock)
        mock_coord_block.text = coordinator_response
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_block]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher 1 - requests web_search tool
        mock_tool_use_block_1 = Mock(spec=ToolUseBlock)
        mock_tool_use_block_1.type = "tool_use"
        mock_tool_use_block_1.name = "web_search"
        mock_tool_use_block_1.input = {"query": "AI coding assistants 2024"}
        mock_tool_use_block_1.id = "tool_1"

        mock_tool_use_message_1 = Mock(spec=Message)
        mock_tool_use_message_1.content = [mock_tool_use_block_1]
        mock_tool_use_message_1.stop_reason = "tool_use"

        # Mock researcher 1 final response
        researcher_response_1 = json.dumps({
            "subtask": "AI coding tools",
            "findings": [
                {
                    "claim": "GitHub Copilot has millions of users",
                    "source": "https://github.com/blog",
                    "details": "As of 2024"
                }
            ]
        })

        mock_research_block_1 = Mock(spec=TextBlock)
        mock_research_block_1.text = researcher_response_1
        mock_research_message_1 = Mock(spec=Message)
        mock_research_message_1.content = [mock_research_block_1]
        mock_research_message_1.stop_reason = "end_turn"

        # Mock researcher 2 - requests web_search tool
        mock_tool_use_block_2 = Mock(spec=ToolUseBlock)
        mock_tool_use_block_2.type = "tool_use"
        mock_tool_use_block_2.name = "web_search"
        mock_tool_use_block_2.input = {"query": "developer productivity metrics"}
        mock_tool_use_block_2.id = "tool_2"

        mock_tool_use_message_2 = Mock(spec=Message)
        mock_tool_use_message_2.content = [mock_tool_use_block_2]
        mock_tool_use_message_2.stop_reason = "tool_use"

        # Mock researcher 2 final response
        researcher_response_2 = json.dumps({
            "subtask": "Developer productivity",
            "findings": [
                {
                    "claim": "AI tools increase productivity by 30%",
                    "source": "https://research.com/study",
                    "details": "Survey of 1000 developers"
                }
            ]
        })

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer response
        synthesizer_response = json.dumps({
            "summary": "AI tools are transforming software development",
            "sections": [
                {
                    "title": "AI Coding Tools",
                    "content": "GitHub Copilot has millions of users",
                    "sources": ["https://github.com/blog"]
                },
                {
                    "title": "Developer Productivity",
                    "content": "AI tools increase productivity by 30%",
                    "sources": ["https://research.com/study"]
                }
            ],
            "key_insights": ["AI adoption is growing", "Productivity gains are significant"]
        })
        mock_synth_block = Mock(spec=TextBlock)
        mock_synth_block.text = synthesizer_response
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_block]
        mock_synth_message.stop_reason = "end_turn"

        # Set up API call sequence
        client.messages.create.side_effect = [
            mock_coord_message,         # Coordinator call
            mock_tool_use_message_1,    # Researcher 1 requests tool
            mock_research_message_1,    # Researcher 1 final response
            mock_tool_use_message_2,    # Researcher 2 requests tool
            mock_research_message_2,    # Researcher 2 final response
            mock_synth_message          # Synthesizer
        ]

        # Mock execute_web_search function
        def mock_execute_web_search(query, api_key):
            if "coding" in query.lower():
                return [
                    {
                        "title": "GitHub Copilot Stats",
                        "url": "https://github.com/blog",
                        "content": "Copilot usage statistics...",
                        "score": 0.95
                    }
                ]
            elif "productivity" in query.lower():
                return [
                    {
                        "title": "Developer Productivity Study",
                        "url": "https://research.com/study",
                        "content": "Productivity metrics...",
                        "score": 0.90
                    }
                ]

            return []

        with patch('orchestration.workflow.execute_web_search', side_effect=mock_execute_web_search):
            result = run_research_workflow(
                query="Impact of AI on software development",
                client=client,
                coordinator_prompt="Coordinator prompt",
                researcher_prompt="Researcher prompt",
                synthesizer_prompt="Synthesizer prompt",
                tavily_api_key="test_api_key"
            )

        # Verify result structure
        assert result["query"] == "Impact of AI on software development"
        assert result["subtasks"] == ["AI coding tools", "Developer productivity"]
        assert len(result["research_results"]) == 2

        # Verify findings
        assert result["research_results"][0]["subtask"] == "AI coding tools"
        assert result["research_results"][0]["findings"][0]["claim"] == "GitHub Copilot has millions of users"

        assert result["research_results"][1]["subtask"] == "Developer productivity"
        assert result["research_results"][1]["findings"][0]["claim"] == "AI tools increase productivity by 30%"

    def test_run_research_workflow_web_search_error_handling(self):
        """Test workflow handles web_search errors gracefully."""
        from unittest.mock import patch
        from anthropic.types import ToolUseBlock

        client = Mock()

        # Mock coordinator response
        coordinator_response = '["Test subtask 1", "Test subtask 2"]'
        mock_coord_block = Mock(spec=TextBlock)
        mock_coord_block.text = coordinator_response
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_block]
        mock_coord_message.stop_reason = "end_turn"

        # Mock researcher requesting tool that will fail
        mock_tool_use_block = Mock(spec=ToolUseBlock)
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.name = "web_search"
        mock_tool_use_block.input = {"query": "test"}
        mock_tool_use_block.id = "tool_1"

        mock_tool_use_message = Mock(spec=Message)
        mock_tool_use_message.content = [mock_tool_use_block]
        mock_tool_use_message.stop_reason = "tool_use"

        # Mock researcher handling error and returning empty findings
        researcher_response = json.dumps({
            "subtask": "Test subtask 1",
            "findings": []
        })

        mock_research_block = Mock(spec=TextBlock)
        mock_research_block.text = researcher_response
        mock_research_message = Mock(spec=Message)
        mock_research_message.content = [mock_research_block]
        mock_research_message.stop_reason = "end_turn"

        # Second researcher response (no tool use)
        researcher_response_2 = json.dumps({
            "subtask": "Test subtask 2",
            "findings": []
        })

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]
        mock_research_message_2.stop_reason = "end_turn"

        # Mock synthesizer response
        synthesizer_response = json.dumps({
            "summary": "Test",
            "sections": [],
            "key_insights": []
        })
        mock_synth_block = Mock(spec=TextBlock)
        mock_synth_block.text = synthesizer_response
        mock_synth_message = Mock(spec=Message)
        mock_synth_message.content = [mock_synth_block]
        mock_synth_message.stop_reason = "end_turn"

        client.messages.create.side_effect = [
            mock_coord_message,
            mock_tool_use_message,
            mock_research_message,
            mock_research_message_2,
            mock_synth_message
        ]

        # Mock execute_web_search to raise an error
        def failing_execute_web_search(query, api_key):
            raise Exception("Tavily API rate limit exceeded")

        with patch('orchestration.workflow.execute_web_search', side_effect=failing_execute_web_search):
            result = run_research_workflow(
                query="Test query",
                client=client,
                coordinator_prompt="Coordinator prompt",
                researcher_prompt="Researcher prompt",
                synthesizer_prompt="Synthesizer prompt",
                tavily_api_key="test_api_key"
            )

        # Workflow should complete despite tool errors
        assert result["query"] == "Test query"
        assert len(result["research_results"]) == 2
        assert result["research_results"][0]["findings"] == []
