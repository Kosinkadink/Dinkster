"""Native image layer document construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LAYERS_TYPE,
    MASK_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

_BLEND_MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "linear-dodge",
    "linear-burn",
    "vivid-light",
    "pin-light",
    "linear-light",
    "hard-mix",
    "subtract",
    "divide",
    "grain-extract",
    "grain-merge",
    "hue",
    "saturation",
    "color",
    "luminosity",
)
_INT = TypeExpr.concrete(CORE_INT)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_LAYERS = TypeExpr.concrete(LAYERS_TYPE)
_MAX_RESOLUTION = 16_384


class AddLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "AddLayer",
            display_name="Add Layer",
            category="image",
            inputs=(
                InputSpec(
                    "layers",
                    _LAYERS,
                    required=False,
                    doc="Layer stack to append to. Leave unconnected to start a new stack.",
                ),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Layer content at its native size. A batch expands to consecutive layers.",
                ),
                InputSpec(
                    "mask",
                    TypeExpr.concrete(MASK_TYPE),
                    required=False,
                    doc=(
                        "Transparency mask for this layer. Masked areas (value 1) become "
                        "transparent, multiplying with any alpha channel the image already carries."
                    ),
                ),
                InputSpec(
                    "name",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default="",
                    widget=StringWidget(multiline=False),
                    doc="Layer name shown in the compositor editor.",
                ),
                InputSpec(
                    "x",
                    _INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-_MAX_RESOLUTION, max=_MAX_RESOLUTION),
                    doc="Initial horizontal placement on the canvas.",
                ),
                InputSpec(
                    "y",
                    _INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-_MAX_RESOLUTION, max=_MAX_RESOLUTION),
                    doc="Initial vertical placement on the canvas.",
                ),
                InputSpec(
                    "opacity",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Initial layer opacity.",
                ),
                InputSpec(
                    "blend_mode",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="normal",
                    widget=ComboWidget(options=_BLEND_MODES),
                    doc=(
                        "Initial blend mode, applied against the layers below. On the bottom layer "
                        "over the default transparent background, non-normal modes produce "
                        "transparency."
                    ),
                ),
                InputSpec(
                    "rotation",
                    _FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=-360.0, max=360.0, step=1.0),
                    doc="Initial rotation in degrees, clockwise.",
                ),
                InputSpec(
                    "width",
                    _INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=_MAX_RESOLUTION),
                    doc="Initial display width. 0 keeps the image's native width.",
                ),
                InputSpec(
                    "height",
                    _INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=_MAX_RESOLUTION),
                    doc="Initial display height. 0 keeps the image's native height.",
                ),
                InputSpec(
                    "z_index",
                    _INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=-1000, max=1000),
                    doc=(
                        "Stacking override. Layers are stable-sorted by z_index; "
                        "equal values keep their list order."
                    ),
                ),
                InputSpec(
                    "flip_h",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=False,
                    doc="Flip the layer horizontally.",
                ),
                InputSpec(
                    "flip_v",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=False,
                    doc="Flip the layer vertically.",
                ),
            ),
            outputs=(
                OutputSpec("LAYERS", _LAYERS, doc="The layer stack with this layer appended."),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        layers: object = None,
        mask: object = None,
        name: str = "",
        x: int = 0,
        y: int = 0,
        opacity: float = 1.0,
        blend_mode: str = "normal",
        rotation: float = 0.0,
        width: int = 0,
        height: int = 0,
        z_index: int = 0,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> Mapping[str, object]:
        item: dict[str, Any] = {
            "image": image,
            "type": "raster",
            "x": int(x),
            "y": int(y),
            "z_index": int(z_index),
        }
        if mask is not None:
            item["mask"] = mask
        if name:
            item["name"] = name
        if opacity != 1.0:
            item["opacity"] = float(opacity)
        if blend_mode != "normal":
            item["blend_mode"] = blend_mode
        if rotation != 0.0:
            item["rotation"] = math.radians(rotation)
        if width > 0:
            item["w"] = int(width)
        if height > 0:
            item["h"] = int(height)
        if flip_h:
            item["flip_h"] = True
        if flip_v:
            item["flip_v"] = True
        previous = cast(dict[str, Any], layers) if isinstance(layers, dict) else None
        document: dict[str, Any] = {
            "version": 1,
            "layers": [*(previous.get("layers") or []), item] if previous else [item],
        }
        canvas = previous.get("canvas") if previous else None
        if isinstance(canvas, (tuple, list)):
            dimensions = cast(Sequence[object], canvas)
            if len(dimensions) == 2:
                # The reference accepts numeric dimensions, not booleans or numeric strings.
                w, h = (
                    int(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else 0
                    for value in dimensions
                )
                if w > 0 and h > 0:
                    document["canvas"] = (w, h)
        return cls.outputs(LAYERS=document)
