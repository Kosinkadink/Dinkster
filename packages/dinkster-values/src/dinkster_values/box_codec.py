"""Closed JSON contracts for pixel boxes and normalized placement elements.

Pixel coordinates retain their numeric kinds and frame nesting. Normalized
elements use thousandths in ymin/xmin/ymax/xmax order; only consumers with
canvas dimensions convert them to pixels. Neither payload carries resources.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import partial
from typing import cast

from .builtin_types import BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE
from .registry import TypeRegistry, TypeSpec


def _number(value: object) -> int | float:
    if type(value) not in (int, float):
        raise ValueError("box coordinates and scores must be finite JSON numbers")
    number = cast("int | float", value)
    try:
        finite = math.isfinite(number)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("box coordinates and scores must be finite JSON numbers")
    return number


def _strings(value: object) -> None:
    if type(value) is not list:
        raise ValueError("box palettes must be lists of strings")
    if any(type(item) is not str for item in cast("list[object]", value)):
        raise ValueError("box palettes must be lists of strings")


def _metadata(value: object) -> None:
    if type(value) is not dict:
        raise ValueError("box metadata must be a declared JSON object")
    data = cast("dict[str, object]", value)
    if set(data) - {"name", "desc", "z_index", "content_rect", "type", "text", "palette"}:
        raise ValueError("unknown box metadata field")
    for key in ("name", "desc", "text"):
        if key in data and type(data[key]) is not str:
            raise ValueError(f"box metadata {key} must be a string")
    if "type" in data and data["type"] not in ("obj", "text"):
        raise ValueError("box metadata type must be obj or text")
    if "z_index" in data and type(data["z_index"]) is not int:
        raise ValueError("box metadata z_index must be an integer")
    if "palette" in data:
        _strings(data["palette"])
    if "content_rect" in data:
        rect = _quad(data["content_rect"])
        if _number(rect[2]) < 0 or _number(rect[3]) < 0:
            raise ValueError("content_rect dimensions must be non-negative")


def _quad(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError("box coordinate arrays must have exactly four numbers")
    result = cast("list[object]", value)
    if len(result) != 4:
        raise ValueError("box coordinate arrays must have exactly four numbers")
    for item in result:
        _number(item)
    return result


def _box(value: object, *, normalized: bool) -> None:
    if type(value) is not dict:
        raise ValueError("a box must be a declared JSON object")
    box = cast("dict[str, object]", value)
    if normalized and "bbox" in box:
        if set(box) - {"bbox", "desc", "type", "text", "color_palette"}:
            raise ValueError("unknown normalized element field")
        _quad(box["bbox"])
        for key in ("desc", "text"):
            if key in box and type(box[key]) is not str:
                raise ValueError(f"element {key} must be a string")
        if "type" in box and box["type"] not in ("obj", "text"):
            raise ValueError("element type must be obj or text")
        if "color_palette" in box:
            _strings(box["color_palette"])
        return
    required = {"x", "y", "width", "height"}
    if not required <= set(box) or set(box) - required - {"label", "score", "metadata"}:
        raise ValueError("pixel boxes require x, y, width, height and only declared fields")
    for key in required:
        number = _number(box[key])
        if key in ("width", "height") and number < 0:
            raise ValueError("pixel box dimensions must be non-negative")
    if "label" in box and type(box["label"]) is not str:
        raise ValueError("box label must be a string")
    if "score" in box and not 0 <= _number(box["score"]) <= 1:
        raise ValueError("box score must be between zero and one")
    if "metadata" in box:
        _metadata(box["metadata"])


def _coerce_boxes(value: object, *, normalized: bool) -> object:
    if not normalized and type(value) is dict:
        box = cast("dict[str, object]", value)
        _box(box, normalized=False)
        return box
    if type(value) is not list:
        raise ValueError("boxes must be a flat or per-frame JSON list")
    items = cast("list[object]", value)
    nested = bool(items) and type(items[0]) is list
    for item in items:
        if nested:
            if type(item) is not list:
                raise ValueError("box frames must not mix nesting levels")
            for box in cast("list[object]", item):
                _box(box, normalized=normalized)
        else:
            _box(item, normalized=normalized)
    return items


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate box JSON field")
        result[key] = value
    return result


def _encode_boxes(value: object, *, normalized: bool) -> bytes:
    _coerce_boxes(value, normalized=normalized)
    return b"json:" + json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _decode_boxes(data: bytes, *, normalized: bool) -> object:
    if not data.startswith(b"json:"):
        raise ValueError("boxes require JSON framing, never pickle")
    try:
        value = json.loads(data[5:].decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid box JSON payload") from exc
    return _coerce_boxes(value, normalized=normalized)


def validate_bounding_box_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    """Pure admission for single, flat or per-frame pixel boxes."""
    _decode_boxes(data, normalized=False)


def validate_placement_array_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    """Pure admission for flat or per-frame pixel/normalized placements."""
    _decode_boxes(data, normalized=True)


def register_box_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Register JSON-only boxes and placements without replacing first authority."""
    specs: list[TypeSpec] = []
    for type_id, normalized, validator in (
        (BOUNDING_BOX_TYPE, False, validate_bounding_box_encoded),
        (COMFY_ARRAY_TYPE, True, validate_placement_array_encoded),
    ):
        specs.append(
            registry.spec(type_id)
            if type_id in registry
            else registry.register(
                type_id,
                encode=partial(_encode_boxes, normalized=normalized),
                decode=partial(_decode_boxes, normalized=normalized),
                coerce=partial(_coerce_boxes, normalized=normalized),
                validate_encoded=validator,
            )
        )
    return tuple(specs)
