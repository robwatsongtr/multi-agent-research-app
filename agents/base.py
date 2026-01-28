import logging
from typing import Any, Optional, Callable, cast
from anthropic import Anthropic
from anthropic.types import Message, MessageParam
import json

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Provides common functionality for making API calls to Claude with
    specialized system prompts and extracting responses.

    Attributes:
        client: Anthropic API client instance
        system_prompt: System prompt that defines the agent's role and behavior
        model: Claude model to use for API calls
    """

    def __init__(self,
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

    def call_claude(self,
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
        messages: list[MessageParam] = [{"role": "user", "content": user_message}]

        while True:
            api_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": self.system_prompt,
                "messages": messages,
            }

            if tools is not None:
                api_params["tools"] = tools

            logger.debug("Calling Claude API...")
            response = self.client.messages.create(**api_params)

            if response.stop_reason == "tool_use":
                if tool_executor is None:
                    raise ValueError("Tool use requested but no tool_executor provided")

                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id

                        logger.info(f"Claude requested tool: {tool_name}")

                        try:
                            result = tool_executor(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result)
                            })
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True
                            })

                messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))
                continue

            return cast(Message, response)

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

        first_block = message.content[0]

        if hasattr(first_block, 'text'):
            return first_block.text

        raise ValueError(f"Unexpected content block type: {type(first_block)}")
