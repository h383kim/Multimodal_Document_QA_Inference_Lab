"""JSON extraction + schema validation with retry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel, ValidationError

from app.backends.base import BackendResponse, ModelBackend

_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(raw: str) -> str | None:
    """Pull a JSON object/array out of arbitrary model text.

    Handles: bare JSON, ```json ... ``` fences, and JSON inside narrative prose.
    Returns None if nothing parses.
    """
    if not raw:
        return None

    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    fenced = _FENCED_BLOCK.search(raw)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Greedy scan for the largest balanced {...} substring.
    candidate = _find_balanced_object(raw)
    if candidate is not None:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None
    return None


def _find_balanced_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


@dataclass
class ParseResult:
    parsed: BaseModel | None
    raw_answer: str
    retry_count: int
    schema_valid: bool
    parse_failure_reason: str | None
    last_backend_response: BackendResponse


def _retry_prompt(original: str, error: str) -> str:
    return (
        f"{original}\n\n"
        f"Your previous response could not be parsed: {error}\n"
        "Reply with ONLY a single valid JSON object that matches the requested schema. "
        "Do not include prose, code fences, or commentary."
    )


def parse_json_with_retry(
    backend: ModelBackend,
    image: Image,
    prompt: str,
    schema: type[BaseModel],
    max_retries: int = 2,
    generation_config: dict[str, Any] | None = None,
) -> ParseResult:
    """Call the backend, parse JSON, validate schema; retry on failure."""
    current_prompt = prompt
    last_response: BackendResponse | None = None
    last_error: str | None = None
    raw_answer = ""

    for attempt in range(max_retries + 1):
        last_response = backend.generate(image, current_prompt, generation_config)
        raw_answer = last_response["answer"]
        candidate = extract_json(raw_answer)
        if candidate is None:
            last_error = "no JSON object found in model output"
            current_prompt = _retry_prompt(prompt, last_error)
            continue
        try:
            parsed_dict = json.loads(candidate)
            parsed_dict = _coerce_schema_keys(parsed_dict, schema)
            parsed = schema.model_validate(parsed_dict)
            return ParseResult(
                parsed=parsed,
                raw_answer=raw_answer,
                retry_count=attempt,
                schema_valid=True,
                parse_failure_reason=None,
                last_backend_response=last_response,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            current_prompt = _retry_prompt(prompt, last_error)

    assert last_response is not None
    return ParseResult(
        parsed=None,
        raw_answer=raw_answer,
        retry_count=max_retries,
        schema_valid=False,
        parse_failure_reason=last_error,
        last_backend_response=last_response,
    )


def _coerce_schema_keys(value: Any, schema: type[BaseModel]) -> Any:
    """Map common model key variants onto the requested Pydantic field names."""
    if not isinstance(value, dict):
        return value

    normalized_fields: dict[str, str] = {}
    for field in schema.model_fields:
        for alias in _field_key_aliases(field):
            normalized_fields[alias] = field
    coerced: dict[str, Any] = {}
    for key, item in value.items():
        key_str = str(key)
        schema_key = normalized_fields.get(_normalize_key(key_str), key_str)
        coerced[schema_key] = item
    return coerced


def _field_key_aliases(field: str) -> set[str]:
    normalized = _normalize_key(field)
    aliases = {normalized}
    for suffix in ("name", "date", "amount", "number", "method"):
        if normalized.endswith(suffix):
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())
