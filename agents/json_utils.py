"""
Utility functions for parsing JSON from LLM responses.

Handles common issues with LLM-generated JSON including:
- Markdown code blocks before/after JSON
- Language identifiers (```json)
- Text before/after the JSON payload
- Detailed error reporting for debugging
"""

import json
import logging
from typing import Any, Dict, List, Union, cast

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from an LLM response that may contain markdown or extra text.

    Handles responses like:
    - Plain JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - With preamble: Here's the data:\n```json\n{"key": "value"}\n```

    Args:
        response_text: Raw text response from LLM

    Returns:
        Cleaned JSON string ready for parsing

    Raises:
        ValueError: If no valid JSON structure is found
    """
    if not response_text:
        raise ValueError("Empty response text")

    # Check for markdown code blocks
    if '```' in response_text:
        parts = response_text.split('```')

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
    response_text = response_text.strip()

    # Check if it looks like JSON
    if response_text.startswith('{') or response_text.startswith('['):
        return response_text

    raise ValueError("Could not find valid JSON structure in response")


def parse_json_response(
    response_text: str,
    expected_fields: Union[List[str], None] = None
) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response with robust error handling.

    Args:
        response_text: Raw text response from LLM
        expected_fields: Optional list of required top-level fields to validate

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If JSON cannot be parsed or is missing required fields
    """
    # Log the raw response for debugging
    logger.debug(f"Raw response length: {len(response_text)}")
    logger.debug(f"Response preview: {response_text[:200]}...")

    # Extract JSON from markdown/text
    try:
        json_text = extract_json_from_response(response_text)
        logger.debug(f"Extracted JSON length: {len(json_text)}")
        logger.debug(f"JSON starts with: {json_text[:500]}")
    except ValueError as e:
        logger.error(f"Failed to extract JSON: {e}")
        raise

    # Parse the JSON
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Log detailed error context
        error_pos = e.pos if hasattr(e, 'pos') else 0
        start = max(0, error_pos - 100)
        end = min(len(json_text), error_pos + 100)
        context = json_text[start:end]

        logger.error(f"JSON parse error at position {error_pos}")
        logger.error(f"Context around error: ...{context}...")
        logger.error(f"Full JSON text: {json_text}")

        raise ValueError(f"Failed to parse response as JSON: {e}") from e

    # Validate expected fields if provided
    if expected_fields:
        missing_fields = [field for field in expected_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Response missing required fields: {missing_fields}")

    logger.debug(f"Successfully parsed JSON with keys: {list(result.keys())}")

    return cast(Dict[str, Any], result)
