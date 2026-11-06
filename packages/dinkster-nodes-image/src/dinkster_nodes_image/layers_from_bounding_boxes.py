"""Place individual image frames using pixel boxes or normalized elements."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    BOUNDING_BOX_TYPE,
    CORE_BOOLEAN,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LAYERS_TYPE,
    MASK_TYPE,
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

from .bounding_boxes import boxes_from_input


def _integer(value: Any, default: int = 0) -> int:
    return (
        int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def _canvas(document: Any) -> tuple[int, int] | None:
    if not isinstance(document, dict):
        return None
    canvas: Any = cast("dict[str, Any]", document).get("canvas")
    if not isinstance(canvas, (list, tuple)) or len(cast("list[Any]", canvas)) != 2:
        return None
    width, height = (_integer(value) for value in cast("list[Any]", canvas))
    return (width, height) if width > 0 and height > 0 else None


def _boxes(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"bboxes string input is not valid JSON: {error}") from error
    probe: list[Any] = cast("list[Any]", value) if isinstance(value, list) else [value]
    if probe and isinstance(probe[0], list):
        probe = cast("list[Any]", probe[0])
    has_elements = any(
        isinstance(box, dict) and isinstance(cast("dict[str, Any]", box).get("bbox"), (list, tuple))
        for box in probe
    )
    if has_elements and (width <= 0 or height <= 0):
        raise ValueError(
            "normalized element boxes need canvas_width and canvas_height to resolve to pixels"
        )
    return boxes_from_input(value, width, height)


class LayersFromBoundingBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.image.layers_from_bounding_boxes",
            display_name="Layers From Bounding Boxes",
            category="image",
            search_terms=("LayersFromBoundingBoxes",),
            description=(
                "Turn an image batch plus its bounding boxes into a layer stack, one layer per "
                "frame, each placed by its own box. Use this when a node emits layers as a batch "
                "- a batch carries a single placement for every frame, so the individual "
                "positions are otherwise lost."
            ),
            inputs=(
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Image batch; each frame becomes one layer.",
                ),
                InputSpec(
                    "bboxes",
                    TypeExpr.union(BOUNDING_BOX_TYPE, ARRAY_TYPE, CORE_STRING),
                    doc=(
                        "Placement boxes, index-aligned with the image batch. Accepts bounding "
                        "boxes (x, y, width, height), normalized elements (with a 'bbox' - these "
                        "need canvas_width/canvas_height to resolve to pixels), or a JSON string "
                        "of either. Frames without a matching box are placed at the origin. "
                        "A box's width/height scales the layer to fit it. metadata.name (or desc) "
                        "and metadata.z_index are used when present, and metadata.content_rect "
                        "(frame-relative) crops the frame to its real content."
                    ),
                ),
                InputSpec(
                    "mask",
                    TypeExpr.concrete(MASK_TYPE),
                    required=False,
                    doc=(
                        "Per-frame transparency, index-aligned with the image batch "
                        "(1 = transparent, LoadImage convention)."
                    ),
                ),
                InputSpec(
                    "layers",
                    TypeExpr.concrete(LAYERS_TYPE),
                    required=False,
                    doc="Layer stack to append to. Leave unconnected to start a new stack.",
                ),
                InputSpec(
                    "crop_to_content",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=True,
                    doc=(
                        "Crop each frame to metadata.content_rect where present and place the "
                        "content at the box position plus the rect offset. Leave on for batches "
                        "whose frames are padded - it keeps only the real content at its true spot."
                    ),
                ),
                InputSpec(
                    "canvas_width",
                    TypeExpr.concrete(CORE_INT),
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384),
                    doc="Document canvas width. 0 derives it from the placed layers.",
                ),
                InputSpec(
                    "canvas_height",
                    TypeExpr.concrete(CORE_INT),
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16_384),
                    doc="Document canvas height. 0 derives it from the placed layers.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "LAYERS",
                    TypeExpr.concrete(LAYERS_TYPE),
                    doc="The layer stack, ready for Create Layered Image.",
                ),
            ),
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="LayersFromBoundingBoxes",
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
        image: Any,
        bboxes: Any,
        mask: Any = None,
        layers: Any = None,
        crop_to_content: bool = True,
        canvas_width: int = 0,
        canvas_height: int = 0,
    ) -> Mapping[str, object]:
        boxes = _boxes(bboxes, canvas_width, canvas_height)
        previous = cast("dict[str, Any]", layers) if isinstance(layers, dict) else None
        items: list[dict[str, Any]] = list((previous.get("layers") or []) if previous else [])
        base_z = max((_integer(item.get("z_index")) for item in items), default=-1) + 1
        for index in range(image.shape[0]):
            box = boxes[index] if index < len(boxes) else {}
            meta = box.get("metadata")
            meta = cast("dict[str, Any]", meta) if isinstance(meta, dict) else {}
            frame = image[index : index + 1]
            frame_mask = None
            if mask is not None:
                if mask.shape[0] == 1:
                    frame_mask = mask[:1]
                elif index < mask.shape[0]:
                    frame_mask = mask[index : index + 1]
            x, y = _integer(box.get("x")), _integer(box.get("y"))
            box_width, box_height = _integer(box.get("width")), _integer(box.get("height"))
            cropped = False
            rect = meta.get("content_rect")
            if (
                crop_to_content
                and isinstance(rect, (list, tuple))
                and len(cast("list[Any]", rect)) == 4
            ):
                left, top, width, height = (_integer(value) for value in cast("list[Any]", rect))
                left = min(max(left, 0), int(frame.shape[2]))
                top = min(max(top, 0), int(frame.shape[1]))
                width = min(max(width, 0), int(frame.shape[2]) - left)
                height = min(max(height, 0), int(frame.shape[1]) - top)
                if width > 0 and height > 0:
                    frame = frame[:, top : top + height, left : left + width]
                    if frame_mask is not None:
                        frame_mask = frame_mask[:, top : top + height, left : left + width]
                    x, y = x + left, y + top
                    cropped = True
            item: dict[str, Any] = {
                "image": frame,
                "type": "raster",
                "x": x,
                "y": y,
                "z_index": _integer(meta.get("z_index"), base_z + index),
            }
            if not cropped:
                if box_width > 0:
                    item["w"] = box_width
                if box_height > 0:
                    item["h"] = box_height
            if frame_mask is not None:
                item["mask"] = frame_mask
            name = meta.get("name")
            if not (isinstance(name, str) and name):
                name = meta.get("desc")
            if isinstance(name, str) and name:
                item["name"] = name
            items.append(item)
        document: dict[str, Any] = {"version": 1, "layers": items}
        if canvas_width > 0 and canvas_height > 0:
            document["canvas"] = (canvas_width, canvas_height)
        elif inherited := _canvas(previous):
            document["canvas"] = inherited
        return {"LAYERS": document}
