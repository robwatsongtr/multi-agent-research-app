"""Base agent class for multi-agent research system."""

from typing import Any, Optional
from anthropic import Anthropic
from anthropic.types import Message 

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
        tools: Optional[list[dict[str, Any]]] = None
    ) -> Message:
        """
        Make an API call to Claude with the agent's system prompt.

        Args:
            user_message: The user message to send to Claude
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            tools: Optional list of tools the agent can use

        Returns:
            Message object from the Anthropic API

        Raises:
            anthropic.APIError: If the API call fails
        """
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }

        if tools is not None:
            params["tools"] = tools

        return self.client.messages.create(**params)

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
