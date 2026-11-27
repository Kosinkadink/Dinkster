"""Validation of the persistent Board document carried by a session."""

from __future__ import annotations

import math
import re
from typing import cast

_ITEM_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
_DIGEST = re.compile(r"blake3:[0-9a-f]{64}")
_BLUEPRINT_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MEDIA = frozenset({"picture", "video", "sound"})
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 4_000_000
_MAX_JSON_CHARS = 64_000_000


def validate_board_json(value: object) -> str | None:
    """Match the frontend ownership budget for a Board snapshot or patch."""
    nodes = 0
    chars = 0

    def visit(value: object, depth: int) -> str | None:
        nonlocal nodes, chars
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            return "Board exceeds the JSON node budget"
        if isinstance(value, str):
            chars += len(value.encode("utf-16-le", "surrogatepass")) // 2
        elif isinstance(value, (dict, list)):
            if depth >= _MAX_JSON_DEPTH:
                return "Board exceeds the JSON depth budget"
            if nodes + len(cast("dict[object, object] | list[object]", value)) > _MAX_JSON_NODES:
                return "Board exceeds the JSON node budget"
            if isinstance(value, dict):
                for key, child in cast("dict[object, object]", value).items():
                    if key == "__proto__":
                        return "Board contains a forbidden '__proto__' key"
                    if isinstance(key, str):
                        chars += len(key.encode("utf-16-le", "surrogatepass")) // 2
                    if chars > _MAX_JSON_CHARS:
                        return "Board exceeds the JSON string budget"
                    if problem := visit(child, depth + 1):
                        return problem
            else:
                for child in cast("list[object]", value):
                    if problem := visit(child, depth + 1):
                        return problem
        if chars > _MAX_JSON_CHARS:
            return "Board exceeds the JSON string budget"
        return None

    return visit(value, 0)


def _number(value: object, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and minimum <= value <= maximum
        and math.isfinite(value)
    )


def _text(value: object, maximum: int) -> bool:
    # The frontend's bounds count UTF-16 code units, not Unicode code points.
    return isinstance(value, str) and len(value.encode("utf-16-le", "surrogatepass")) <= maximum * 2


def validate_board(document_id: str, value: object) -> str | None:
    """Return the reason a Board cannot be admitted, or None for a valid document."""
    if not isinstance(value, dict):
        return "Board must be an object"
    board = cast("dict[str, object]", value)
    if board.keys() - {"format", "formatVersion", "lineage", "items", "order", "runs"}:
        return "Board contains unsupported fields"
    if board.get("format") != "dinkster-board" or not _number(board.get("formatVersion"), 1, 1):
        return "Board format must be dinkster-board version 1"
    if not document_id or board.get("lineage") != document_id:
        return "Board lineage must match documentId"
    if "runs" in board:
        raw_runs = board["runs"]
        if not isinstance(raw_runs, dict):
            return "Board runs must be an object"
        # References outlive removed cards/results, so replay cannot restore dismissed media.
        for job, raw_run in cast("dict[str, object]", raw_runs).items():
            if not _text(job, 2048) or not job or not isinstance(raw_run, dict):
                return "Board runs need nonempty job references and records"
            run = cast("dict[str, object]", raw_run)
            card, delivered = run.get("card"), run.get("items")
            if (
                run.keys() != {"card", "items"}
                or not isinstance(card, str)
                or not _ITEM_ID.fullmatch(card)
                or not isinstance(delivered, list)
            ):
                return "Board run needs a card id and delivered item ids"
            item_ids = cast("list[object]", delivered)
            if (
                len(item_ids) > 2000
                or not all(isinstance(item, str) and _ITEM_ID.fullmatch(item) for item in item_ids)
                or len(set(item_ids)) != len(item_ids)
            ):
                return "Board run delivered item ids must be unique Board ids"
    raw_items, order = board.get("items"), board.get("order")
    if not isinstance(raw_items, dict) or not isinstance(order, list):
        return "Board needs an items object and an order list"
    items = cast("dict[str, object]", raw_items)
    order = cast("list[object]", order)
    if len(items) > 2000:
        return "Board holds at most 2000 items"
    if not all(_ITEM_ID.fullmatch(key) for key in items):
        return "Board item ids must be lowercase letters, digits and '-', starting with a letter"
    if (
        not all(isinstance(key, str) for key in order)
        or len(order) != len(items)
        or set(order) != set(items)
    ):
        return "Board order must list every item exactly once"
    parents: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for key, raw in items.items():
        if not isinstance(raw, dict):
            return f"Board item '{key}' must be an object"
        item = cast("dict[str, object]", raw)
        if item.get("id") != key:
            return f"Board item '{key}' id must equal its key"
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in (
            *_MEDIA,
            "note",
            "section",
            "workflow",
            "blueprint",
        ):
            return f"Board item '{key}' has an unknown kind"
        content = (
            {"name", "source", "values", "controllers", "autoSize"}
            if kind in {"workflow", "blueprint"}
            else {"text" if kind == "note" else "name" if kind == "section" else "asset"}
        )
        if item.keys() - {"id", "kind", "rect", "sectionId", *content}:
            return f"Board item '{key}' contains unsupported fields"
        kinds[key] = kind
        raw_rect = item.get("rect")
        if not isinstance(raw_rect, dict):
            return f"Board item '{key}' needs a rect"
        rect = cast("dict[str, object]", raw_rect)
        if rect.keys() - {"x", "y", "width", "height"}:
            return f"Board item '{key}' rect contains unsupported fields"
        if any(not _number(rect.get(axis), -1_000_000, 1_000_000) for axis in ("x", "y")) or any(
            not _number(rect.get(axis), 16, 1_000_000) for axis in ("width", "height")
        ):
            return f"Board item '{key}' has an invalid rect"
        if "sectionId" in item:
            parent = item["sectionId"]
            if not isinstance(parent, str) or not _ITEM_ID.fullmatch(parent):
                return f"Board item '{key}' has an invalid sectionId"
            parents[key] = parent
        if kind == "note" and not _text(item.get("text"), 10_000):
            return f"Board note '{key}' text exceeds 10000 characters or is not text"
        if kind == "section" and not _text(item.get("name"), 200):
            return f"Board section '{key}' name exceeds 200 characters or is not text"
        if kind in {"workflow", "blueprint"}:
            if not _text(item.get("name"), 200) or not isinstance(item.get("values"), dict):
                return f"Board card '{key}' needs a name and values object"
            if "autoSize" in item and not isinstance(item["autoSize"], bool):
                return f"Board card '{key}' autoSize must be a boolean"
            if "controllers" in item:
                controllers = item["controllers"]
                if not isinstance(controllers, dict) or any(
                    mode not in ("fixed", "increment", "decrement", "randomize")
                    for mode in cast("dict[str, object]", controllers).values()
                ):
                    return f"Board card '{key}' has invalid controller modes"
            raw_source = item.get("source")
            if not isinstance(raw_source, dict):
                return f"Board card '{key}' needs a source"
            source = cast("dict[str, object]", raw_source)
            if source.get("kind") == "library":
                fields = {"recordId"}
            elif kind == "blueprint" and source.get("kind") == "pack":
                fields = {"packId", "blueprintId", "digest"}
                digest = source.get("digest")
                if not isinstance(digest, str) or not _BLUEPRINT_DIGEST.fullmatch(digest):
                    return f"Board card '{key}' needs a sha256 blueprint digest"
            else:
                return f"Board card '{key}' has an invalid source kind"
            if source.keys() - {"kind", *fields} or any(
                not _text(source.get(field), 200) or not source.get(field) for field in fields
            ):
                return f"Board card '{key}' has an invalid source"
        if kind not in _MEDIA:
            continue
        raw_asset = item.get("asset")
        if not isinstance(raw_asset, dict):
            return f"Board item '{key}' needs an asset"
        asset = cast("dict[str, object]", raw_asset)
        if asset.keys() - {"digest", "mediaType", "name", "width", "height", "duration", "origin"}:
            return f"Board item '{key}' asset contains unsupported fields"
        digest = asset.get("digest")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            return f"Board item '{key}' needs a blake3 asset digest"
        media_type = asset.get("mediaType")
        if not _text(media_type, 200) or not isinstance(media_type, str) or not media_type:
            return f"Board item '{key}' needs a media type"
        if not _text(asset.get("name"), 200):
            return f"Board item '{key}' has an invalid asset name"
        for axis in ("width", "height"):
            if axis in asset and (
                not _number(asset[axis], 1, 2**53 - 1)
                or not float(cast("int | float", asset[axis])).is_integer()
            ):
                return f"Board item '{key}' asset {axis} must be a positive safe integer"
        if "duration" in asset and not _number(asset["duration"], 0, 2**53 - 1):
            return f"Board item '{key}' asset duration must be nonnegative"
        if "origin" in asset:
            origin = asset["origin"]
            if not isinstance(origin, dict) or not all(
                _text(field, 2048) for field in cast("dict[str, object]", origin).values()
            ):
                return f"Board item '{key}' asset origin must contain strings"
    for key in items:
        seen = {key}
        parent = parents.get(key)
        while parent is not None:
            if parent in seen or kinds.get(parent) != "section":
                return f"Board item '{key}' membership must reference sections without cycles"
            seen.add(parent)
            parent = parents.get(parent)
    return None
