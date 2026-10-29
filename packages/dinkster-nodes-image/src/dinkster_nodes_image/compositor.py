"""Node builders and editor replay over the authoritative image document."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import (
    CompositorWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    copy_media_semantics,
    media_semantics,
    report_event,
    report_preview,
)
from dinkster_image_document.document import (
    ImageDocument,
    append_raster,
    empty_document,
    flatten,
    raster_png,
)
from dinkster_image_document.format import BLEND_MODES

from .geometry import BOOLEAN, FLOAT, IMAGE, INT, MASK
from .layer_document import migrate_layers
from .support import combo_input, image_array, mask_array
from .types import Detection, Region

LAYERS = TypeExpr.concrete("dinkster.layers")
COMPOSITOR = TypeExpr.concrete("dinkster.compositor")


class AddLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.add",
            display_name="Add Layer",
            category="image/compositor",
            inputs=(
                InputSpec("layers", LAYERS, required=False, default=None),
                InputSpec("image", IMAGE),
                InputSpec(
                    "mask",
                    MASK,
                    required=False,
                    default=None,
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
                InputSpec(
                    "name",
                    TypeExpr.concrete("core.string"),
                    required=False,
                    default="Layer",
                    widget=StringWidget(),
                ),
                *(
                    InputSpec(key, FLOAT, required=False, default=0.0)
                    for key in ("x", "y", "rotation")
                ),
                *(InputSpec(key, INT, required=False, default=0) for key in ("width", "height")),
                InputSpec("opacity", FLOAT, required=False, default=1.0),
                InputSpec("z_index", INT, required=False, default=0),
                combo_input("blend_mode", BLEND_MODES, "normal"),
                InputSpec("visible", BOOLEAN, required=False, default=True),
                InputSpec("flip_horizontal", BOOLEAN, required=False, default=False),
                InputSpec("flip_vertical", BOOLEAN, required=False, default=False),
            ),
            outputs=(OutputSpec("layers", LAYERS),),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        layers: object | None = None,
        mask: object | None = None,
        name: str = "Layer",
        x: float = 0,
        y: float = 0,
        width: int = 0,
        height: int = 0,
        rotation: float = 0,
        opacity: float = 1,
        z_index: int = 0,
        blend_mode: str = "normal",
        visible: bool = True,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> Mapping[str, object]:
        source = image_array(image)
        masks = None if mask is None else mask_array(mask)
        if masks is not None and (
            masks.shape[0] not in (1, source.shape[0]) or masks.shape[1:] != source.shape[1:3]
        ):
            raise ValueError("mask batch and dimensions must match image")
        document = (
            migrate_layers(layers)
            if layers is not None
            else empty_document(
                max(1, math.ceil(x + (width or source.shape[2]))),
                max(1, math.ceil(y + (height or source.shape[1]))),
                color=cast("Mapping[str, object] | None", media_semantics(image).get("color")),
            )
        )
        for index, frame in enumerate(source):
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            document = append_raster(
                document,
                raster_png(copy_media_semantics(image, frame)),
                mask_data=None
                if masks is None
                else raster_png(masks[0 if masks.shape[0] == 1 else index][..., None]),
                name=f"{name} {index + 1}" if len(source) > 1 else name,
                x=x,
                y=y,
                width=width,
                height=height,
                rotation=rotation,
                opacity=opacity,
                z_index=z_index,
                blend_mode=blend_mode,
                visible=visible,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
            )
        return cls.outputs(layers=document)


class LayersFromBoundingBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.from_bounding_boxes",
            display_name="Layers From Bounding Boxes",
            category="image/compositor",
            inputs=(
                InputSpec("layers", LAYERS, required=False, default=None),
                InputSpec("image", IMAGE),
                InputSpec(
                    "bounding_boxes",
                    TypeExpr.list_of(TypeExpr.union("dinkster.region", "dinkster.detection")),
                    required=False,
                    default=None,
                ),
                InputSpec(
                    "mask",
                    MASK,
                    required=False,
                    default=None,
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
                InputSpec("canvas_width", INT, required=False, default=0),
                InputSpec("canvas_height", INT, required=False, default=0),
                InputSpec(
                    "bboxes",
                    TypeExpr.union("core.string", "comfy.BOUNDING_BOX", "comfy.ARRAY"),
                    required=False,
                    default=None,
                ),
                InputSpec("crop_to_content", BOOLEAN, required=False, default=True),
            ),
            outputs=(OutputSpec("layers", LAYERS),),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        bounding_boxes: Sequence[object] | None = None,
        layers: object | None = None,
        mask: object | None = None,
        canvas_width: int = 0,
        canvas_height: int = 0,
        bboxes: object = None,
        crop_to_content: bool = True,
    ) -> Mapping[str, object]:
        source = image_array(image)
        metadata: list[dict[str, Any]] = [{} for _ in source]
        if bounding_boxes is None:
            raw: Any = json.loads(bboxes) if isinstance(bboxes, str) else bboxes
            items: list[Any] = [] if raw is None else [raw] if isinstance(raw, Mapping) else raw
            if items and isinstance(items[0], list):
                items = cast(list[Any], items[0])
            boxes = cast(list[dict[str, Any]], items)
            parsed_regions: list[Region] = []
            for index, frame in enumerate(source):
                box = boxes[index] if index < len(boxes) else {}
                metadata[index] = box.get("metadata", {})
                if "bbox" in box:
                    ymin, xmin, ymax, xmax = (float(value) / 1000 for value in box["bbox"])
                    if not canvas_width or not canvas_height:
                        raise ValueError("normalized boxes require canvas dimensions")
                    x, y = (
                        round(min(xmin, xmax) * canvas_width),
                        round(min(ymin, ymax) * canvas_height),
                    )
                    width, height = (
                        round(abs(xmax - xmin) * canvas_width),
                        round(abs(ymax - ymin) * canvas_height),
                    )
                    metadata[index] = {"desc": box.get("desc", "")}
                else:
                    x, y = box.get("x", 0), box.get("y", 0)
                    width, height = (
                        box.get("width", 0) or frame.shape[1],
                        box.get("height", 0) or frame.shape[0],
                    )
                parsed_regions.append(Region(x, y, width, height))
            bounding_boxes = parsed_regions
        if len(bounding_boxes) != source.shape[0]:
            raise ValueError("bounding box count must equal the image batch")
        masks = None if mask is None else mask_array(mask)
        if masks is not None and masks.shape[0] not in (1, source.shape[0]):
            raise ValueError("mask batch must be one or equal the image batch")
        raw_regions = [box.region if isinstance(box, Detection) else box for box in bounding_boxes]
        if any(not isinstance(region, Region) for region in raw_regions):
            raise TypeError("bounding boxes must contain Region or Detection values")
        regions = cast(list[Region], raw_regions)
        document = (
            migrate_layers(layers)
            if layers is not None
            else empty_document(
                canvas_width or max(1, math.ceil(max(r.x + r.width for r in regions))),
                canvas_height or max(1, math.ceil(max(r.y + r.height for r in regions))),
                color=cast("Mapping[str, object] | None", media_semantics(image).get("color")),
            )
        )
        base_z = (
            max(
                (layer.get("z_index", 0) for layer in document.to_record()["layers"].values()),
                default=-1,
            )
            + 1
        )
        for index, (box, region) in enumerate(zip(bounding_boxes, regions, strict=True)):
            frame = source[index : index + 1]
            selected = (
                masks[0 if masks.shape[0] == 1 else index][None]
                if masks is not None
                else (
                    box.mask[None] if isinstance(box, Detection) and box.mask is not None else None
                )
            )
            rect = metadata[index].get("content_rect")
            if crop_to_content and rect is not None:
                left, top, width, height = (int(value) for value in rect)
                left, top = min(max(0, left), frame.shape[2]), min(max(0, top), frame.shape[1])
                width, height = (
                    min(max(0, width), frame.shape[2] - left),
                    min(max(0, height), frame.shape[1] - top),
                )
                if width and height:
                    frame = frame[:, top : top + height, left : left + width]
                    selected = (
                        None
                        if selected is None
                        else selected[:, top : top + height, left : left + width]
                    )
                    region = Region(region.x + left, region.y + top, width, height)
            document = cast(
                ImageDocument,
                AddLayer.execute(
                    image=copy_media_semantics(image, frame),
                    layers=document,
                    mask=selected,
                    name=box.label
                    if isinstance(box, Detection)
                    else metadata[index].get("name")
                    or metadata[index].get("desc")
                    or f"Layer {index + 1}",
                    z_index=metadata[index].get("z_index", base_z + index),
                    x=region.x,
                    y=region.y,
                    width=round(region.width),
                    height=round(region.height),
                )["layers"],
            )
        if canvas_width or canvas_height:
            record = document.to_record()
            record["canvas"]["width"] = canvas_width or record["canvas"]["width"]
            record["canvas"]["height"] = canvas_height or record["canvas"]["height"]
            document = ImageDocument.from_record(record, document.bind)
        return cls.outputs(layers=document)


class CreateLayeredImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.create_layered",
            editor_role="compositor",
            display_name="Create Layered Image",
            category="image/compositor",
            inputs=(
                InputSpec("layers", LAYERS),
                InputSpec(
                    "compositor",
                    COMPOSITOR,
                    required=False,
                    default={"version": 2, "documentDigest": None, "commands": []},
                    widget=CompositorWidget(),
                ),
                combo_input("color_space", ("perceptual", "linear"), "perceptual", advanced=True),
            ),
            outputs=(
                OutputSpec("image", IMAGE, preview=True),
                OutputSpec(
                    "transparency_mask",
                    MASK,
                    preview=True,
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
            ),
            output_node=True,
            emits_previews=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, layers: object, compositor: object | None = None, color_space: str = "perceptual"
    ) -> Mapping[str, object]:
        from .layer_recipe import prepare_compositor

        document, input_digest, stale = prepare_compositor(layers, compositor)
        if color_space == "linear":
            record = document.to_record()
            record["canvas"]["compositing"] = "linear-premultiplied-alpha"
            document = ImageDocument.from_record(record, document.bind)
        elif color_space != "perceptual":
            report_event(
                "dinkster.diagnostic",
                {"message": "Unknown compositor color space; using the document default."},
            )
        image, mask = flatten(document)
        record = document.to_record()
        streams: list[str] = []
        for index, identifier in enumerate(record["rootLayerIds"]):
            stream = f"compositor.layer.{index}"
            preview = document.render(f"layer:{identifier}")
            report_preview(
                preview.png,
                mime="image/png",
                width=preview.width,
                height=preview.height,
                stream=stream,
            )
            streams.append(stream)
        report_event(
            "dinkster.compositor.state",
            {
                "version": 2,
                "document": record,
                "documentDigest": input_digest,
                "stale": stale,
                "layerStreams": streams,
            },
        )
        return cls.outputs(image=image, transparency_mask=mask)


COMPOSITOR_NODES: tuple[type[Node], ...] = (AddLayer, LayersFromBoundingBoxes, CreateLayeredImage)
