"""Guarded import of ComfyUI SavedResult UI records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_workers import current_execution_context


def _ui_mapping(ui: object) -> Mapping[str, object] | None:
    if isinstance(ui, Mapping):
        return cast("Mapping[str, object]", ui)
    as_dict = getattr(ui, "as_dict", None)
    if not callable(as_dict):
        return None
    rendered = cast(Any, as_dict)()
    return cast("Mapping[str, object]", rendered) if isinstance(rendered, Mapping) else None


def _saved_records(ui: object) -> list[Mapping[str, object]]:
    mapping = _ui_mapping(ui)
    if mapping is None:
        return []
    records: list[Mapping[str, object]] = []
    for value in mapping.values():
        if not isinstance(value, (list, tuple)):
            continue
        for item in cast("Sequence[object]", value):
            if isinstance(item, Mapping):
                record = cast("Mapping[str, object]", item)
                if set(record) == {"filename", "subfolder", "type"}:
                    records.append(record)
    return records


def capture_saved_results(ui: object) -> None:
    """Report recognized SavedResult records for host-side validation."""
    context = current_execution_context()
    if context is None or context.artifact_sink is None or context.node_id is None:
        return
    records = _saved_records(ui)
    if not records:
        return
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        filename = record["filename"]
        subfolder = record["subfolder"]
        folder_type = record["type"]
        if not all(type(value) is str for value in (filename, subfolder, folder_type)):
            continue
        parsed = cast("tuple[str, str, str]", (filename, subfolder, folder_type))
        identity = parsed
        if identity not in seen:
            context.artifact_sink(context.node_id, *parsed)
            seen.add(identity)


__all__ = ["capture_saved_results"]
