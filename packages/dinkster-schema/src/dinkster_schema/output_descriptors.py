"""Reading the stored description of a ``Described`` row."""

from __future__ import annotations

import json
from typing import Never, cast

MAX_DESCRIPTOR_BYTES = 1_048_576


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-JSON numeric constant: {value}")


def output_descriptor_entries(value: object) -> tuple[dict[str, object], ...]:
    """Read the entry list; additional document/entry fields belong to the node."""
    if not isinstance(value, str):
        raise ValueError("output descriptors must be a stored string literal, not a link")
    if len(value.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
        raise ValueError("output descriptor document exceeds the byte budget")
    try:
        document: object = json.loads(value, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as error:
        raise ValueError("output descriptor document must be valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("output descriptor document must be an object with entries")
    entries = cast("dict[str, object]", document).get("entries")
    if not isinstance(entries, list) or len(cast("list[object]", entries)) > 512:
        raise ValueError("output descriptor entries must be a list of at most 512 entries")
    if any(not isinstance(entry, dict) for entry in cast("list[object]", entries)):
        raise ValueError("output descriptor entries must be objects")
    return tuple(cast("list[dict[str, object]]", entries))
