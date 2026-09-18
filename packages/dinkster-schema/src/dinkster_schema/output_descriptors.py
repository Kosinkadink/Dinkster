"""Pure projection of persisted output declarations into a concrete interface."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Never, cast

from .model import STRUCTURAL_ID_PATTERN, OutputDescriptorsSpec, OutputSpec

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


def elaborate_output_descriptors(
    spec: OutputDescriptorsSpec, value: object
) -> tuple[OutputSpec, ...]:
    entries = output_descriptor_entries(value)
    if not spec.min_entries <= len(entries) <= spec.max_entries:
        raise ValueError("output descriptor count is outside its declared bounds")
    choices = {choice.id: choice for choice in spec.choices}
    ids: set[str] = set()
    names: set[str] = set()
    outputs: list[OutputSpec] = []
    for entry in entries:
        member_id, name, choice_id = entry.get("id"), entry.get("name"), entry.get("type")
        if not isinstance(member_id, str) or not STRUCTURAL_ID_PATTERN.fullmatch(member_id):
            raise ValueError("output descriptor id must match [A-Za-z0-9_-]+")
        if not isinstance(name, str) or not name.strip() or len(name) > 256 or "\0" in name:
            raise ValueError("output descriptor name must be non-empty and at most 256 characters")
        if member_id in ids or name in names:
            raise ValueError("output descriptor ids and names must be unique")
        if not isinstance(choice_id, str) or choice_id not in choices:
            raise ValueError(f"unknown output descriptor type choice: {choice_id!r}")
        if spec.fixed_ids and member_id != choice_id:
            raise ValueError("fixed output descriptor ids must equal their semantic choice ids")
        ids.add(member_id)
        names.add(name)
        outputs.append(replace(choices[choice_id], id=member_id, display_name=name))
    return tuple(outputs)
