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

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]

        # Return coordinator response first, then researcher responses
        client.messages.create.side_effect = [
            mock_coord_message,
            mock_research_message_1,
            mock_research_message_2
        ]

        # Run workflow
        result = run_research_workflow(
            query="Test query",
            client=client,
            coordinator_prompt="Coordinator prompt",
            researcher_prompt="Researcher prompt"
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

        # Verify API was called 3 times (1 coordinator + 2 researcher)
        assert client.messages.create.call_count == 3

    def test_run_research_workflow_with_tools(self):
        """Test workflow passes tools to researcher."""
        client = Mock()

        # Mock coordinator response (2 subtasks minimum)
        coordinator_response = '["Subtask 1", "Subtask 2"]'
        mock_coord_block = Mock(spec=TextBlock)
        mock_coord_block.text = coordinator_response
        mock_coord_message = Mock(spec=Message)
        mock_coord_message.content = [mock_coord_block]

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

        mock_research_block_2 = Mock(spec=TextBlock)
        mock_research_block_2.text = researcher_response_2
        mock_research_message_2 = Mock(spec=Message)
        mock_research_message_2.content = [mock_research_block_2]

        client.messages.create.side_effect = [
            mock_coord_message,
            mock_research_message_1,
            mock_research_message_2
        ]

        tools = [{"name": "web_search", "description": "Search the web"}]

        result = run_research_workflow(
            query="Test query",
            client=client,
            coordinator_prompt="Coordinator prompt",
            researcher_prompt="Researcher prompt",
            tools=tools
        )

        # Verify tools were passed to researcher calls
        researcher_call_1 = client.messages.create.call_args_list[1]
        researcher_call_2 = client.messages.create.call_args_list[2]
        assert researcher_call_1[1]["tools"] == tools
        assert researcher_call_2[1]["tools"] == tools
