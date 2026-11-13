"""Strict JSON decoding shared by independent LLM editing operations."""
import json

from .errors import InvalidLLMResponse


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def load_json_response(content, error_message):
    """Reject duplicate keys and NaN/Infinity before operation-specific validation."""
    try:
        return json.loads(content, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        raise InvalidLLMResponse(error_message) from None
