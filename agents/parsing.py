"""
Robust parsing utilities for LLM responses into Pydantic models.

Handles common LLM output variations like markdown code blocks, trailing commas,
and comments in JSON.
"""

import json
import re
from typing import TypeVar, Type, Callable, Optional
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Tries in order:
    1. JSON code block (```json...```)
    2. Generic code block (```...```)
    3. Raw JSON object/array in text
    4. Entire text (assumes it's all JSON)

    Args:
        text: Raw text from LLM that may contain JSON

    Returns:
        Extracted JSON string
    """
    # Try to find JSON in code blocks with 'json' language marker
    json_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    code_block_pattern = r"```\s*(.*?)\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object or array
    json_pattern = r"(\{.*\}|\[.*\])"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Assume entire text is JSON
    return text.strip()


def clean_json_string(json_str: str) -> str:
    """
    Fix common JSON formatting issues that LLMs make.

    - Removes trailing commas before } or ]
    - Removes // and /* */ style comments
    - Preserves valid JSON structure

    Args:
        json_str: JSON string that may have formatting issues

    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # Remove single-line comments (// ...)
    json_str = re.sub(r"//.*?\n", "\n", json_str)

    # Remove multi-line comments (/* ... */)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    return json_str


def parse_llm_response(
    response_text: str,
    model_class: Type[T],
    fallback_parser: Optional[Callable[[str], dict]] = None,
) -> T:
    """
    Robust parsing of LLM response into Pydantic model.

    Attempts multiple parsing strategies:
    1. Extract JSON from code blocks and validate
    2. Try parsing as dict first, then validate with Pydantic
    3. Use custom fallback parser if provided

    Args:
        response_text: Raw text from LLM
        model_class: Pydantic model class to parse into
        fallback_parser: Optional function that takes text and returns dict

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If all parsing strategies fail, with details about each failure
    """
    errors = []

    # Strategy 1: Extract JSON, clean it, and parse directly
    try:
        json_str = extract_json_from_text(response_text)
        json_str = clean_json_string(json_str)
        return model_class.model_validate_json(json_str)
    except (json.JSONDecodeError, ValidationError) as e:
        errors.append(f"JSON parsing failed: {e}")

    # Strategy 2: Parse to dict first, then validate
    try:
        json_str = extract_json_from_text(response_text)
        json_str = clean_json_string(json_str)
        data = json.loads(json_str)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        errors.append(f"Dict validation failed: {e}")

    # Strategy 3: Use custom fallback parser
    if fallback_parser:
        try:
            data = fallback_parser(response_text)
            return model_class.model_validate(data)
        except Exception as e:
            errors.append(f"Fallback parser failed: {e}")

    # All strategies failed - raise detailed error
    raise ValueError(
        f"Failed to parse LLM response into {model_class.__name__}. "
        f"Errors: {'; '.join(errors)}. "
        f"Response preview: {response_text[:200]}..."
    )
