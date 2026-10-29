"""Layer value migration and node expressions over the shared image document."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, StringWidget, TypeExpr
from dinkster_image_document.compat import from_comfy_layers
from dinkster_image_document.document import (
    ImageDocument,
    apply_commands,
    document_meta,
    flatten,
)
from dinkster_image_document.format import ordered_layer_ids

from .compositor_types import LayerStack, decode_layer_stack, source_layer_fingerprint
from .types import Region

LAYERS = TypeExpr.concrete("dinkster.layers")
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
STRING = TypeExpr.concrete("core.string")


def migrate_layers(obj: object) -> ImageDocument:
    if isinstance(obj, ImageDocument):
        return obj
    if isinstance(obj, Mapping):
        return from_comfy_layers(cast(Mapping[str, Any], obj))
    elif isinstance(obj, LayerStack):
        sources = obj.expanded()
        width = obj.canvas_width or max(
            1, math.ceil(max(source.x + source.width for source in sources))
        )
        height = obj.canvas_height or max(
            1, math.ceil(max(source.y + source.height for source in sources))
        )
        layers: list[dict[str, Any]] = [
            dict(
                image=s.image,
                mask=s.mask,
                name=s.name,
                x=s.x,
                y=s.y,
                w=s.width,
                h=s.height,
                rotation=s.rotation,
                opacity=s.opacity,
                blend_mode=s.blend_mode,
                visible=s.visible,
                flip_h=s.flip_horizontal,
                flip_v=s.flip_vertical,
            )
            for s in sources
        ]
    else:
        raise TypeError("LAYERS expects an ImageDocument or legacy layer document")
    migrated = from_comfy_layers({"version": 1, "canvas": (width, height), "layers": layers})
    record = migrated.to_record()
    record["canvas"]["compositing"] = "premultiplied-alpha"
    record["extensions"]["legacyLayerInputs"] = [source_layer_fingerprint(s) for s in sources]
    return ImageDocument.from_record(record, migrated.bind)


def decode_layers(data: bytes) -> ImageDocument:
    if data.startswith(b"DINKSTER-LAYERS\x00\x01"):
        return migrate_layers(decode_layer_stack(data))
    return ImageDocument(data)


def layer_meta(obj: object) -> Mapping[str, object]:
    return document_meta(migrate_layers(obj))


class FlattenLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.flatten",
            editor_role="layers-flatten",
            display_name="Flatten Layers",
            category="image/compositor",
            inputs=(
                InputSpec("layers", LAYERS),
                InputSpec(
                    "selector", STRING, required=False, default="composite", widget=StringWidget()
                ),
            ),
            outputs=(
                OutputSpec("image", IMAGE, preview=True),
                OutputSpec(
                    "transparency_mask", MASK, mask_polarity="transparency", mask_semantic="alpha"
                ),
            ),
        )

    @classmethod
    def execute(cls, *, layers: object, selector: str = "composite") -> Mapping[str, object]:
        image, mask = flatten(migrate_layers(layers), selector)
        return cls.outputs(image=image, transparency_mask=mask)


class EditLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.edit",
            editor_role="layers-edit",
            display_name="Edit Layers",
            category="image/compositor",
            description=(
                "Apply remove, reorder, group, layer, mask, transform, "
                "add_mask or remove_mask commands."
            ),
            inputs=(
                InputSpec("layers", LAYERS),
                InputSpec("commands", STRING, widget=StringWidget(multiline=True)),
            ),
            outputs=(OutputSpec("layers", LAYERS),),
        )

    @classmethod
    def execute(cls, *, layers: object, commands: str) -> Mapping[str, object]:
        edits = json.loads(commands)
        if not isinstance(edits, list):
            raise ValueError("commands must be an array")
        return cls.outputs(
            layers=apply_commands(migrate_layers(layers), cast(list[dict[str, Any]], edits))
        )


class SplitLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.split",
            display_name="Split Layers",
            category="image/compositor",
            inputs=(InputSpec("layers", LAYERS),),
            outputs=(
                OutputSpec("images", TypeExpr.list_of(IMAGE)),
                OutputSpec(
                    "masks",
                    TypeExpr.list_of(MASK),
                    mask_polarity="transparency",
                    mask_semantic="alpha",
                ),
                OutputSpec("regions", TypeExpr.list_of(TypeExpr.concrete("dinkster.region"))),
            ),
        )

    @classmethod
    def execute(cls, *, layers: object) -> Mapping[str, object]:
        document = migrate_layers(layers)
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        regions: list[Region] = []
        record = document.to_record()

        def raster_ids(ids: list[str]) -> Iterator[str]:
            for identifier in ordered_layer_ids(record["layers"], ids):
                layer = record["layers"][identifier]
                if layer["kind"] == "group":
                    yield from raster_ids(layer["childLayerIds"])
                else:
                    yield identifier

        for identifier in raster_ids(record["rootLayerIds"]):
            image, mask = flatten(document, f"layer:{identifier}")
            images.append(image)
            masks.append(mask)
            regions.append(Region(0, 0, record["canvas"]["width"], record["canvas"]["height"]))
        return cls.outputs(images=images, masks=masks, regions=regions)


LAYER_DOCUMENT_NODES: tuple[type[Node], ...] = (FlattenLayers, EditLayers, SplitLayers)
