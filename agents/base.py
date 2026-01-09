"""
Base agent class for multi-agent research system.
"""

from typing import Any, Optional, Callable
from anthropic import Anthropic
from anthropic.types import Message
import json 

class BaseAgent:
    """
    Base class for all agents in the multi-agent research system.

    Provides common functionality for making API calls to Claude with
    specialized system prompts and extracting responses.

    Attributes:
        client: Anthropic API client instance
        system_prompt: System prompt that defines the agent's role and behavior
        model: Claude model to use for API calls
    """

    def __init__( self, 
        client: Anthropic, 
        system_prompt: str,
        model: str = "claude-sonnet-4-5-20250929"
    ) -> None:
        """
        Initialize the base agent.

        Args:
            client: Anthropic API client instance
            system_prompt: System prompt that defines agent behavior
            model: Claude model to use (defaults to latest Sonnet)
        """
        self.client = client
        self.system_prompt = system_prompt
        self.model = model

    def call_claude( self,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, dict[str, Any]], Any]] = None
    ) -> Message:
        """
        Make an API call to Claude with the agent's system prompt.
        Handles tool use loop automatically if tools are provided.

        Args:
            user_message: The user message to send to Claude
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            tools: Optional list of tools the agent can use
            tool_executor: Optional function to execute tools. Takes (tool_name, tool_input) and returns results.

        Returns:
            Message object from the Anthropic API (after all tool use is complete)

        Raises:
            anthropic.APIError: If the API call fails
            ValueError: If tool use is requested but no executor provided
        """
        # Build conversation history
        messages = [{"role": "user", "content": user_message}]

        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": self.system_prompt,
        }

        if tools is not None:
            params["tools"] = tools

        # Tool use loop
        while True:
            params["messages"] = messages
            response = self.client.messages.create(**params)

            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                if tool_executor is None:
                    raise ValueError("Tool use requested but no tool_executor provided")

                # Add Claude's response to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Execute all tool uses in this response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id

                        # Execute the tool
                        try:
                            result = tool_executor(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result)
                            })
                        except Exception as e:
                            # Return error to Claude
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True
                            })

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})

                # Continue the loop - Claude will process tool results
                continue

            # No tool use, return the final response
            return response 

    def parse_response(self, message: Message) -> str:
        """
        Extract text content from a Claude API response.

        Args:
            message: Message object from the Anthropic API

        Returns:
            The text content from the first content block

        Raises:
            ValueError: If the message has no text content
        """
        if not message.content:
            raise ValueError("Message has no content")

        # Get the first content block
        first_block = message.content[0]

        # Extract text based on block type
        if hasattr(first_block, 'text'):
            return first_block.text

        raise ValueError(f"Unexpected content block type: {type(first_block)}")
