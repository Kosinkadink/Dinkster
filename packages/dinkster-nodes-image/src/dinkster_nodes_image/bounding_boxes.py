"""Bounding-box canvas input and element conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import (
    ARRAY_TYPE,
    BOUNDING_BOX_TYPE,
    BOUNDING_BOXES_TYPE,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    MappingSource,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
)
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = value.lstrip("#")
    if len(color) != 6:
        return (255, 255, 255)
    try:
        return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def _readable_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance >= 130:
        return rgb
    factor = (130 - luminance) / (255 - luminance)
    return (
        round(r + (255 - r) * factor),
        round(g + (255 - g) * factor),
        round(b + (255 - b) * factor),
    )


def _wrap(draw: Any, text: str, font: Any, max_width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = word if not line else line + " " + word
            if line and draw.textlength(candidate, font=font) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def render_preview(
    regions: list[Any], width: int, height: int, background: Image.Image | None = None
) -> Any:
    if background is not None:
        iw, ih = background.size
        scale = min(1.0, 1024 / (max(iw, ih) or 1))
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = background.convert("RGB").resize((rw, rh), Image.Resampling.LANCZOS)
        image = ImageEnhance.Brightness(base).enhance(0.25).convert("RGBA")
    else:
        scale = min(1.0, 1024 / (max(width, height) or 1))
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        image = Image.new("RGBA", (rw, rh), (32, 32, 32, 255))
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(10, round(rh / 64))
    font = ImageFont.load_default(font_size)
    tag_font = ImageFont.load_default(max(9, font_size - 2))
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        region = cast("dict[str, Any]", region)
        palette = [color for color in cast("list[Any]", region.get("palette") or []) if color]
        r, g, b = _hex_to_rgb(palette[0]) if palette else (140, 140, 140)
        x1 = max(0, min(rw, round(region.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(region.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((region.get("x", 0) + region.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((region.get("y", 0) + region.get("h", 0)) * rh)))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)
        swatches = palette[:5]
        if swatches and x2 - x1 > 2:
            swatch_height = max(5, font_size // 2)
            segment = (x2 - x1) / len(swatches)
            for position, color in enumerate(swatches):
                draw.rectangle(
                    [
                        x1 + round(position * segment),
                        y1,
                        x1 + round((position + 1) * segment),
                        y1 + swatch_height,
                    ],
                    fill=_hex_to_rgb(color),
                )
        tag = str(index + 1).zfill(2)
        tag_width = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tag_width + 6, y1 + font_size + 2], fill=(r, g, b, 255))
        fill = (0, 0, 0, 255) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255,) * 4
        draw.text((x1 + 3, y1 + 1), tag, fill=fill, font=tag_font)
        body = region.get("desc", "") or ""
        if region.get("type") == "text" and region.get("text"):
            body = '"{}"{}'.format(region["text"], " \u2014 " + body if body else "")
        if body and x2 - x1 > 8:
            ty = y1 + font_size + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=_readable_color((r, g, b)) + (255,), font=font)
                ty += font_size + 2
    composed = Image.alpha_composite(image, overlay).convert("RGB")
    return (np.asarray(composed, dtype=np.float32) / 255.0)[None]


def pixels_to_fractions(box: dict[str, Any], width: int, height: int) -> dict[str, float]:
    return {
        "x": box.get("x", 0) / (width or 1),
        "y": box.get("y", 0) / (height or 1),
        "w": box.get("width", 0) / (width or 1),
        "h": box.get("height", 0) / (height or 1),
    }


def fractions_to_pixels(box: dict[str, Any], width: int, height: int) -> dict[str, int]:
    x, y = box.get("x", 0.0), box.get("y", 0.0)
    w, h = box.get("w", 0.0), box.get("h", 0.0)
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return {
        "x": round(x * width),
        "y": round(y * height),
        "width": round(w * width),
        "height": round(h * height),
    }


def fractions_to_bbox_frame(boxes: list[Any], width: int, height: int) -> list[Any]:
    pixels = [
        fractions_to_pixels(cast("dict[str, Any]", box), width, height)
        for box in boxes
        if isinstance(box, dict)
    ]
    return [pixels] if pixels else []


def boxes_to_regions(boxes: object, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(boxes, list):
        return []
    regions: list[dict[str, Any]] = []
    for box in cast("list[Any]", boxes):
        if not isinstance(box, dict):
            continue
        box = cast("dict[str, Any]", box)
        meta = box.get("metadata")
        meta = cast("dict[str, Any]", meta) if isinstance(meta, dict) else {}
        regions.append(
            {
                **pixels_to_fractions(box, width, height),
                "type": meta.get("type", "obj"),
                "text": meta.get("text", ""),
                "desc": meta.get("desc", ""),
                "palette": meta.get("palette", []),
            }
        )
    return regions


def normalize_incoming_boxes(bboxes: object) -> list[dict[str, Any]]:
    frame: list[Any]
    if isinstance(bboxes, dict):
        frame = [cast("dict[str, Any]", bboxes)]
    elif not isinstance(bboxes, list) or not bboxes:
        frame = []
    elif isinstance(bboxes[0], dict):
        frame = cast("list[Any]", bboxes)
    else:
        frame = cast("list[Any]", bboxes[0]) if isinstance(bboxes[0], list) else []
    boxes: list[dict[str, Any]] = []
    for box in frame:
        if not isinstance(box, dict):
            continue
        box = cast("dict[str, Any]", box)
        normalized = {key: box.get(key, 0) for key in ("x", "y", "width", "height")}
        meta = box.get("metadata")
        if isinstance(meta, dict):
            normalized["metadata"] = cast("dict[str, Any]", meta)
        boxes.append(normalized)
    return boxes


def elements_to_boxes(elements: list[Any], width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        element = cast("dict[str, Any]", element)
        bbox = element.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(cast("list[Any]", bbox)) == 4):
            raise ValueError("bboxes element is missing a valid 'bbox' [ymin, xmin, ymax, xmax]")
        try:
            ymin, xmin, ymax, xmax = (float(value) / 1000.0 for value in cast("list[Any]", bbox))
        except (TypeError, ValueError):
            raise ValueError("bboxes element 'bbox' must contain four numbers") from None
        kind = "text" if element.get("type") == "text" else "obj"
        boxes.append(
            {
                "x": round(min(xmin, xmax) * width),
                "y": round(min(ymin, ymax) * height),
                "width": round(abs(xmax - xmin) * width),
                "height": round(abs(ymax - ymin) * height),
                "metadata": {
                    "type": kind,
                    "text": element.get("text", "") if kind == "text" else "",
                    "desc": element.get("desc", ""),
                    "palette": element.get("color_palette", []) or [],
                },
            }
        )
    return boxes


def boxes_from_input(data: Any, width: int, height: int) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"bboxes string input is not valid JSON: {error}") from error
    if isinstance(data, dict):
        data = cast("dict[str, Any]", data)
        bbox = data.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(cast("list[Any]", bbox)) == 4:
            return elements_to_boxes([data], width, height)
        if all(key in data for key in ("x", "y", "width", "height")):
            return normalize_incoming_boxes(data)
        raise ValueError(
            "bboxes dict must be a bounding box (x, y, width, height) or an element (with a 'bbox')"
        )
    if not isinstance(data, list):
        raise ValueError(
            "bboxes input must be bounding boxes, elements, or a JSON string, "
            f"got {type(data).__name__}"
        )
    if not data:
        return []
    data = cast("list[Any]", data)
    first: Any = data[0]
    if isinstance(first, list):
        return normalize_incoming_boxes(data)
    if isinstance(first, dict):
        first = cast("dict[str, Any]", first)
        bbox = first.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(cast("list[Any]", bbox)) == 4:
            return elements_to_boxes(data, width, height)
        if all(key in first for key in ("x", "y", "width", "height")):
            return normalize_incoming_boxes(data)
        raise ValueError(
            "bboxes items must be bounding boxes (x, y, width, height) or elements (with a 'bbox')"
        )
    raise ValueError(
        f"bboxes list must contain bounding boxes or elements, got {type(first).__name__}"
    )


def build_elements(regions: list[Any]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region = cast("dict[str, Any]", region)
        kind = "text" if region.get("type") == "text" else "obj"
        x, y = region.get("x", 0.0), region.get("y", 0.0)
        w, h = region.get("w", 0.0), region.get("h", 0.0)
        ymin, xmin, ymax, xmax = (
            max(0, min(1000, round(value * 1000))) for value in (y, x, y + h, x + w)
        )
        element: dict[str, Any] = {
            "type": kind,
            "bbox": [min(ymin, ymax), min(xmin, xmax), max(ymin, ymax), max(xmin, xmax)],
        }
        if kind == "text":
            element["text"] = region.get("text", "")
        element["desc"] = region.get("desc", "")
        colors = region.get("palette", [])
        if isinstance(colors, dict):
            colors = cast("dict[str, Any]", colors).values()
        palette = [color.upper() for color in colors if isinstance(color, str) and color]
        if palette:
            element["color_palette"] = palette[:5]
        elements.append(element)
    return elements


class CreateBoundingBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.image.create_bounding_boxes",
            display_name="Create Bounding Boxes",
            category="utilities",
            description=(
                "Draw bounding boxes in a canvas. Outputs Ideogram prompt elements, "
                "pixel-space bounding boxes, and a preview image."
            ),
            search_terms=("CreateBoundingBoxes",),
            output_node=True,
            inputs=(
                InputSpec(
                    "background",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc="Optional image used as background in the canvas and preview.",
                ),
                InputSpec(
                    "bboxes",
                    TypeExpr.union(BOUNDING_BOX_TYPE, ARRAY_TYPE, CORE_STRING),
                    required=False,
                    doc=(
                        "Bounding boxes, elements, or a JSON string to initialize the canvas. "
                        "A new upstream value initializes the canvas; edits made on the canvas "
                        "take priority and are kept until the upstream value changes again."
                    ),
                ),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=16_384, step=16),
                    doc="Width of the canvas and the pixel grid for the bounding boxes.",
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=16_384, step=16),
                    doc="Height of the canvas and the pixel grid for the bounding boxes.",
                ),
                InputSpec(
                    "editor_state",
                    TypeExpr.concrete(BOUNDING_BOXES_TYPE),
                    default=[],
                    doc=(
                        "Draw bounding boxes and set each box type, text, description, color "
                        "palette. Start with background element first and foreground last."
                    ),
                ),
                InputSpec(
                    "last_incoming",
                    TypeExpr.concrete(BOUNDING_BOXES_TYPE),
                    required=False,
                    default=[],
                    socketless=True,
                    doc=(
                        "Internal state managed by the canvas: the upstream bboxes value that "
                        "last initialized it. Leave empty to re-initialize the canvas from "
                        "the bboxes input on the next run."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("preview", TypeExpr.concrete(IMAGE_TYPE), display_name="preview"),
                OutputSpec("bboxes", TypeExpr.concrete(BOUNDING_BOX_TYPE), display_name="bboxes"),
                OutputSpec("elements", TypeExpr.concrete(ARRAY_TYPE), display_name="elements"),
            ),
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="CreateBoundingBoxes",
                    cases=(
                        ReplacementCase.build(
                            schema.node_type,
                            inputs={spec.id: MappingSource.copy(spec.id) for spec in schema.inputs},
                            outputs={spec.id: spec.id for spec in schema.outputs},
                        ),
                    ),
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        editor_state: Any = None,
        last_incoming: Any = None,
        background: Any = None,
        bboxes: Any = None,
    ) -> Mapping[str, object]:
        incoming = boxes_from_input(bboxes, width, height)
        applied = cast("list[Any]", last_incoming) if isinstance(last_incoming, list) else []
        upstream_changed = bool(incoming) and incoming != applied
        source = incoming if upstream_changed else (editor_state or [])
        regions = boxes_to_regions(source, width, height)
        bg = None
        if background is not None:
            if hasattr(background, "detach"):
                array = background[0].detach().cpu().numpy()
            else:
                array = np.asarray(background)[0]
            bg = Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))
        return {
            "preview": render_preview(regions, width, height, bg),
            "bboxes": fractions_to_bbox_frame(regions, width, height),
            "elements": build_elements(regions),
        }
