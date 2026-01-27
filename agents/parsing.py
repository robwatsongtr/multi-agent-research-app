"""
Utility functions for parsing JSON from LLM responses.

Handles common issues with LLM-generated JSON including:
- Markdown code blocks before/after JSON
- Language identifiers (```json)
- Text before/after the JSON payload
"""

import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Handles responses like:
    - Plain JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - With preamble: Here's the data:\n```json\n{"key": "value"}\n```

    Args:
        text: Raw text from LLM that may contain JSON

    Returns:
        Extracted JSON string ready for parsing

    Raises:
        ValueError: If no valid JSON structure is found
    """
    if not text:
        raise ValueError("Empty response text")

    # Check for markdown code blocks
    if '```' in text:
        parts = text.split('```')

        # Find the JSON code block (could be parts[1] or later if there's text before)
        json_block = None
        for i in range(1, len(parts), 2):  # Check odd indices (code blocks)
            block = parts[i].strip()

            # Remove language identifier if present
            if block.startswith('json'):
                block = block[4:].strip()
            elif block.startswith('JSON'):
                block = block[4:].strip()

            # Check if this looks like JSON
            if block.startswith('{') or block.startswith('['):
                json_block = block
                break

        if json_block:
            return json_block
        else:
            # No JSON found in code blocks, fall through to try the whole thing
            logger.warning("Found code blocks but no valid JSON inside them")

    # No code blocks or no JSON in code blocks - try the raw text
    text = text.strip()

    # Check if it looks like JSON
    if text.startswith('{') or text.startswith('['):
        return text

    raise ValueError("Could not find valid JSON structure in response")
