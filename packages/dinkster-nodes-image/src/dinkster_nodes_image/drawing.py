"""Deterministic CPU image generation and ordinary drawing overlays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ColorWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    image_math,
)

from .migration import with_v1_migration
from .support import (
    MAX_DIMENSION,
    materialized_inputs,
)
from .support import (
    color_from_int as _color_from_int,
)
from .support import (
    combo_input as _combo,
)
from .support import (
    image_array as _image_array,
)
from .support import (
    linear_ramp as _linear_ramp,
)
from .support import (
    mask_array as _mask_array,
)
from .support import (
    number_input as _number,
)
from .support import (
    parse_color as _parse_color,
)
from .support import (
    radial_ramp as _radial_ramp,
)
from .support import (
    validate_canvas as _validate_canvas,
)
from .types import REGION_TYPE, Region

overlay_rgba = image_math.overlay_rgba
text_coverage = image_math.text_coverage

IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
REGION = TypeExpr.concrete(REGION_TYPE)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)


def _color_input(id: str, default: str, *, advanced: bool = False) -> InputSpec:
    return InputSpec(
        id,
        STRING,
        required=False,
        default=default,
        widget=ColorWidget(),
        advanced=advanced,
    )


def _broadcast_image_coverage(
    image: np.ndarray, coverage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    image_batch, coverage_batch = int(image.shape[0]), int(coverage.shape[0])
    if image_batch == coverage_batch:
        return image, coverage
    if image_batch == 1:
        return np.broadcast_to(image, (coverage_batch, *image.shape[1:])), coverage
    if coverage_batch == 1:
        return image, np.broadcast_to(coverage, (image_batch, *coverage.shape[1:]))
    raise ValueError(
        "image and overlay batches must be equal or singleton, got "
        f"{image_batch} and {coverage_batch}"
    )


def _overlay_color(
    image: np.ndarray,
    coverage: np.ndarray,
    color: str,
    opacity: float,
    alpha_mode: str = "source_over",
) -> np.ndarray:
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be finite and between 0 and 1")
    if alpha_mode not in ("source_over", "max"):
        raise ValueError(f"unknown alpha composition mode: {alpha_mode}")
    image, coverage = _broadcast_image_coverage(image, coverage)
    rgba = _parse_color(color)
    alpha = np.clip(coverage, 0.0, 1.0) * opacity * rgba[3]
    return overlay_rgba(image, alpha, rgba, alpha_mode)


def _resize_mask_nearest_exact(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = int(mask.shape[1]), int(mask.shape[2])
    rows = np.minimum(
        np.floor((np.arange(height) + 0.5) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    columns = np.minimum(
        np.floor((np.arange(width) + 0.5) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    return np.ascontiguousarray(mask[:, rows[:, None], columns[None, :]])


class MakeImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        specs = (
            _color_input("color_a", "#000000"),
            _color_input("color_b", "#ffffff"),
            _number("color_value", INT, 0, minimum=0, maximum=0xFFFFFF, step=1),
            _number("angle", FLOAT, 0.0, step=0.1),
            _number("center_x", FLOAT, 0.5, step=0.01),
            _number("center_y", FLOAT, 0.5, step=0.01),
            _number("radius", FLOAT, 0.5, minimum=0.001, step=0.01),
            _number("tile_size", INT, 32, minimum=1, maximum=MAX_DIMENSION, step=1),
        )
        by_id = {item.id: item for item in specs}
        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.generate",
                version=2,
                display_name="Create Image",
                category="image/create",
                inputs=(
                    _number("width", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                    _number("height", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                    _number("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
                    _combo("channels", ("rgb", "rgba"), "rgb"),
                ),
                combos=(
                    DynamicComboSpec(
                        "color_source",
                        (
                            DynamicComboOption("hex", (by_id["color_a"],)),
                            DynamicComboOption("integer", (by_id["color_value"],)),
                        ),
                        default="hex",
                    ),
                    DynamicComboSpec(
                        "operation",
                        (
                            DynamicComboOption("solid"),
                            DynamicComboOption(
                                "linear_gradient", (by_id["color_b"], by_id["angle"])
                            ),
                            DynamicComboOption(
                                "radial_gradient",
                                (
                                    by_id["color_b"],
                                    by_id["center_x"],
                                    by_id["center_y"],
                                    by_id["radius"],
                                ),
                            ),
                            DynamicComboOption(
                                "checkerboard", (by_id["color_b"], by_id["tile_size"])
                            ),
                        ),
                        default="solid",
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "EmptyImage",
                    "Image Blank",
                    "Image Generate Gradient",
                    "solid color image",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        operation: str = "solid",
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        channels: str = "rgb",
        color_a: str = "#000000",
        color_b: str = "#ffffff",
        color_source: str = "hex",
        color_value: int = 0,
        angle: float = 0.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius: float = 0.5,
        tile_size: int = 32,
    ) -> Mapping[str, object]:
        channel_count = 3 if channels == "rgb" else 4 if channels == "rgba" else 0
        if channel_count == 0:
            raise ValueError(f"unknown image channel layout: {channels}")
        _validate_canvas(width, height, batch_size, channel_count)
        if color_source == "hex":
            first = _parse_color(color_a)[:channel_count]
        elif color_source == "integer":
            first = _color_from_int(color_value)[:channel_count]
        else:
            raise ValueError(f"unknown color source: {color_source}")
        second = _parse_color(color_b)[:channel_count]
        if operation == "solid":
            plane = np.broadcast_to(first, (height, width, channel_count)).copy()
        elif operation == "linear_gradient":
            ramp = _linear_ramp(width, height, angle)
            plane = first + ramp[:, :, None] * (second - first)
        elif operation == "radial_gradient":
            ramp = _radial_ramp(width, height, center_x, center_y, radius)
            plane = first + ramp[:, :, None] * (second - first)
        elif operation == "checkerboard":
            if tile_size < 1:
                raise ValueError("tile_size must be positive")
            rows, columns = np.ogrid[:height, :width]
            selected = ((columns // tile_size) + (rows // tile_size)) % 2 == 0
            plane = np.where(selected[:, :, None], first, second)
        else:
            raise ValueError(f"unknown image creation operation: {operation}")
        output = np.broadcast_to(
            plane[None, :, :, :], (batch_size, height, width, channel_count)
        ).copy()
        return cls.outputs(image=np.ascontiguousarray(output, dtype=np.float32))


class DrawText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.draw_text",
            display_name="Draw Text on Image",
            category="image/draw",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                _number("x", INT, 0, step=1),
                _number("y", INT, 0, step=1),
                _number("font_size", INT, 32, minimum=1, maximum=1024, step=1),
                _color_input("color", "#ffffff"),
                _number("opacity", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
                _number("line_spacing", INT, 4, minimum=0, maximum=1024, step=1, advanced=True),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("DrawText+", "text overlay", "AddText"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        text: str,
        x: int = 0,
        y: int = 0,
        font_size: int = 32,
        color: str = "#ffffff",
        opacity: float = 1.0,
        line_spacing: int = 4,
    ) -> Mapping[str, object]:
        source = _image_array(image)
        coverage = text_coverage(
            int(source.shape[2]), int(source.shape[1]), text, x, y, font_size, line_spacing
        )
        return cls.outputs(image=_overlay_color(source, coverage, color, opacity))


def _region_coverage(
    width: int,
    height: int,
    region: Region,
    shape: str,
    mode: str,
    line_width: int,
) -> np.ndarray:
    if line_width < 1:
        raise ValueError("line_width must be positive")
    rows, columns = np.ogrid[:height, :width]

    def inside(inset: float) -> np.ndarray:
        left, top = region.x + inset, region.y + inset
        region_width, region_height = region.width - inset * 2, region.height - inset * 2
        if region_width <= 0 or region_height <= 0:
            return np.zeros((height, width), dtype=np.bool_)
        if shape == "rectangle":
            return (
                (columns + 0.5 >= left)
                & (columns + 0.5 < left + region_width)
                & (rows + 0.5 >= top)
                & (rows + 0.5 < top + region_height)
            )
        if shape == "ellipse":
            center_x, center_y = left + region_width / 2, top + region_height / 2
            return (
                np.square((columns + 0.5 - center_x) / (region_width / 2))
                + np.square((rows + 0.5 - center_y) / (region_height / 2))
                <= 1.0
            )
        raise ValueError(f"unknown region shape: {shape}")

    outer = inside(0)
    if mode == "fill":
        return outer.astype(np.float32)
    if mode == "outline":
        return np.logical_and(outer, np.logical_not(inside(line_width))).astype(np.float32)
    raise ValueError(f"unknown region draw mode: {mode}")


def _region_number(value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError("legacy region coordinates must be numbers")
    return float(cast("int | float", value))


def _draw_regions(value: object, coordinate_format: str) -> tuple[Region, ...]:
    if coordinate_format not in ("xywh", "xyxy"):
        raise ValueError(f"unknown region coordinate format: {coordinate_format}")
    if isinstance(value, Region):
        return (value,)
    if isinstance(value, np.ndarray):
        value = cast("object", value.tolist())
    elif callable(to_list := getattr(value, "tolist", None)):
        value = to_list()
    if isinstance(value, Mapping):
        record = cast("Mapping[str, object]", value)
        if not all(name in record for name in ("x", "y", "width", "height")):
            raise ValueError("legacy region records require x, y, width, and height")
        return (
            Region(
                x=_region_number(record["x"]),
                y=_region_number(record["y"]),
                width=_region_number(record["width"]),
                height=_region_number(record["height"]),
            ),
        )
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        coordinates = items[:4]
        if len(coordinates) == 4 and all(type(item) in (int, float) for item in coordinates):
            x, y, third, fourth = (_region_number(item) for item in coordinates)
            width = third - x if coordinate_format == "xyxy" else third
            height = fourth - y if coordinate_format == "xyxy" else fourth
            return (Region(x=x, y=y, width=width, height=height),)
        regions = tuple(
            region for item in items for region in _draw_regions(item, coordinate_format)
        )
        if regions:
            return regions
    raise TypeError("region must be a Region or legacy bounding-box value")


class DrawRegion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.draw_region",
                version=2,
                display_name="Draw Region on Image",
                category="image/draw",
                inputs=(
                    InputSpec("image", IMAGE),
                    InputSpec("region", REGION),
                    _combo("shape", ("rectangle", "ellipse"), "rectangle"),
                    _color_input("color", "#ff0000"),
                    _number("opacity", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
                    _combo("coordinate_format", ("xywh", "xyxy"), "xywh", hidden=True),
                    _combo(
                        "batch_policy",
                        ("broadcast", "pairwise_truncate"),
                        "broadcast",
                        advanced=True,
                    ),
                ),
                combos=(
                    DynamicComboSpec(
                        "mode",
                        (
                            DynamicComboOption("fill"),
                            DynamicComboOption(
                                "outline",
                                (
                                    _number(
                                        "line_width",
                                        INT,
                                        1,
                                        minimum=1,
                                        maximum=MAX_DIMENSION,
                                        step=1,
                                    ),
                                ),
                            ),
                        ),
                        default="outline",
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "BboxVisualize",
                    "draw bounding box",
                    "rectangle overlay",
                    "ellipse overlay",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        region: object,
        shape: str = "rectangle",
        mode: str = "outline",
        line_width: int = 1,
        color: str = "#ff0000",
        opacity: float = 1.0,
        coordinate_format: str = "xywh",
        batch_policy: str = "broadcast",
    ) -> Mapping[str, object]:
        source = _image_array(image)
        regions = _draw_regions(region, coordinate_format)
        if batch_policy == "broadcast":
            if len(regions) != 1:
                raise ValueError("broadcast region drawing requires exactly one region")
            coverage = _region_coverage(
                int(source.shape[2]),
                int(source.shape[1]),
                regions[0],
                shape,
                mode,
                line_width,
            )[None, :, :]
        elif batch_policy == "pairwise_truncate":
            count = min(int(source.shape[0]), len(regions))
            source = source[:count]
            coverage = np.stack(
                [
                    _region_coverage(
                        int(source.shape[2]),
                        int(source.shape[1]),
                        regions[index],
                        shape,
                        mode,
                        line_width,
                    )
                    for index in range(count)
                ]
            )
        else:
            raise ValueError(f"unknown region drawing batch policy: {batch_policy}")
        return cls.outputs(image=_overlay_color(source, coverage, color, opacity))


class DrawMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.draw_mask",
            display_name="Draw Mask on Image",
            category="image/draw",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("mask", MASK),
                _number("x", INT, 0, step=1),
                _number("y", INT, 0, step=1),
                _color_input("color", "#ff0000"),
                _number("opacity", FLOAT, 0.5, minimum=0.0, maximum=1.0, step=0.01),
                InputSpec("invert", BOOLEAN, required=False, default=False),
                _combo("mask_size", ("preserve", "resize_to_image"), "preserve", advanced=True),
                _combo(
                    "batch_policy",
                    ("singleton_broadcast", "cyclic_repeat"),
                    "singleton_broadcast",
                    advanced=True,
                ),
                _combo("alpha_mode", ("source_over", "max"), "source_over", advanced=True),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("DrawMaskOnImage", "mask color overlay", "Mix Color By Mask"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        mask: object,
        x: int = 0,
        y: int = 0,
        color: str = "#ff0000",
        opacity: float = 0.5,
        invert: bool = False,
        mask_size: str = "preserve",
        batch_policy: str = "singleton_broadcast",
        alpha_mode: str = "source_over",
    ) -> Mapping[str, object]:
        source = _image_array(image)
        mask_array = _mask_array(mask)
        image_height, image_width = int(source.shape[1]), int(source.shape[2])
        if mask_size == "resize_to_image":
            if mask_array.shape[1:3] != source.shape[1:3]:
                mask_array = _resize_mask_nearest_exact(mask_array, image_width, image_height)
        elif mask_size != "preserve":
            raise ValueError(f"unknown mask size policy: {mask_size}")
        image_batch, mask_batch = int(source.shape[0]), int(mask_array.shape[0])
        if batch_policy == "singleton_broadcast":
            if image_batch == 1 and mask_batch > 1:
                source = np.broadcast_to(source, (mask_batch, *source.shape[1:]))
            elif mask_batch == 1 and image_batch > 1:
                mask_array = np.broadcast_to(mask_array, (image_batch, *mask_array.shape[1:]))
            elif image_batch != mask_batch:
                raise ValueError(
                    "image and mask batches must be equal or singleton, got "
                    f"{image_batch} and {mask_batch}"
                )
        elif batch_policy == "cyclic_repeat":
            indexes = np.arange(image_batch) % mask_batch
            mask_array = mask_array[indexes]
        else:
            raise ValueError(f"unknown batch policy: {batch_policy}")
        if type(invert) is not bool:
            raise TypeError("invert must be a boolean")
        if invert:
            mask_array = 1.0 - mask_array
        coverage = np.zeros(source.shape[:3], dtype=np.float32)
        mask_height, mask_width = mask_array.shape[1:3]
        destination_left, destination_top = max(0, x), max(0, y)
        destination_right = min(image_width, x + mask_width)
        destination_bottom = min(image_height, y + mask_height)
        if destination_right > destination_left and destination_bottom > destination_top:
            source_left, source_top = destination_left - x, destination_top - y
            source_right = source_left + destination_right - destination_left
            source_bottom = source_top + destination_bottom - destination_top
            coverage[:, destination_top:destination_bottom, destination_left:destination_right] = (
                mask_array[:, source_top:source_bottom, source_left:source_right]
            )
        return cls.outputs(
            image=_overlay_color(source, coverage, color, opacity, alpha_mode=alpha_mode)
        )


DRAWING_NODES: tuple[type[Node], ...] = (MakeImage, DrawText, DrawRegion, DrawMask)


__all__ = ["DRAWING_NODES", "DrawMask", "DrawRegion", "DrawText", "MakeImage"]
