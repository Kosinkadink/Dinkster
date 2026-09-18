"""Deterministic CPU mask creation, morphology, composition, and conversion."""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Literal, cast

import cv2
import numpy as np
from dinkster_api.v1 import (
    ABSENT,
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
    annotate_mask,
    copy_media_semantics,
    media_semantics,
)
from PIL import Image, ImageDraw, ImageFont

from .migration import with_mask_polarity, with_v1_migration
from .support import (
    MAX_DIMENSION,
    MAX_IMAGE_BYTES,
    materialized_inputs,
    normalize_mask_polarity,
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

IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
REGION = TypeExpr.concrete(REGION_TYPE)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
FLOAT_LIST = TypeExpr.list_of(FLOAT)

EdgePolicy = Literal["constant", "replicate", "reflect", "wrap"]


def _coordinate_values(value: object, subject: str) -> tuple[float, ...]:
    if value is ABSENT:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{subject} must be a list of numbers")
    try:
        return tuple(float(cast("int | float", item)) for item in cast(Iterable[object], value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{subject} must be a list of numbers") from exc


def _transition_progress(position: int, length: int, timing: str) -> float:
    if length <= 1:
        return 1.0
    progress = position / (length - 1)
    if timing == "linear":
        return progress
    if timing == "in":
        return progress * progress
    if timing == "out":
        return 1.0 - (1.0 - progress) ** 2
    if timing == "in_out":
        if progress < 0.5:
            return 2.0 * progress * progress
        return 1.0 - 2.0 * (1.0 - progress) ** 2
    raise ValueError(f"unknown transition timing function: {timing}")


def _transition_selection(
    width: int, height: int, progress: float, transition_type: str
) -> np.ndarray:
    selected = np.zeros((height, width), dtype=np.float32)
    if transition_type == "horizontal_slide":
        selected[:, : round(width * progress)] = 1.0
    elif transition_type == "vertical_slide":
        selected[: round(height * progress), :] = 1.0
    elif transition_type == "center_box":
        box_width, box_height = round(width * progress), round(height * progress)
        left, top = (width - box_width) // 2, (height - box_height) // 2
        selected[top : top + box_height, left : left + box_width] = 1.0
    elif transition_type == "circle":
        circle_radius = math.ceil(math.hypot(width, height) * progress / 2)
        rows, columns = np.ogrid[:height, :width]
        selected[
            np.square(columns - width // 2) + np.square(rows - height // 2)
            <= circle_radius * circle_radius
        ] = 1.0
    elif transition_type == "horizontal_bar":
        size = round(height * progress)
        top = (height - size) // 2
        selected[top : top + size, :] = 1.0
    elif transition_type == "vertical_bar":
        size = round(width * progress)
        left = (width - size) // 2
        selected[:, left : left + size] = 1.0
    elif transition_type == "horizontal_door":
        size = math.ceil(height * progress / 2)
        if size:
            selected[:size, :] = 1.0
            selected[-size:, :] = 1.0
    elif transition_type == "vertical_door":
        size = math.ceil(width * progress / 2)
        if size:
            selected[:, :size] = 1.0
            selected[:, -size:] = 1.0
    elif transition_type == "fade":
        selected.fill(progress)
    else:
        raise ValueError(f"unknown mask transition type: {transition_type}")
    return selected


class MakeMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        inputs = (
            _combo(
                "operation",
                (
                    "solid",
                    "rectangle",
                    "ellipse",
                    "triangle",
                    "polygon",
                    "region",
                    "linear_gradient",
                    "radial_gradient",
                    "frame_gradient",
                    "transition",
                    "checkerboard",
                    "noise",
                ),
                "solid",
            ),
            _number("width", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("height", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
            _number("foreground", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
            _number("background", FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.01),
            _number("x", INT, 0, step=1),
            _number("y", INT, 0, step=1),
            _combo("shape_origin", ("top_left", "center"), "top_left"),
            _number("shape_width", INT, 256, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("shape_height", INT, 256, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("grow", INT, 0, minimum=-MAX_DIMENSION, maximum=MAX_DIMENSION, step=1),
            InputSpec("region", REGION, required=False),
            InputSpec("points_x", FLOAT_LIST, required=False),
            InputSpec("points_y", FLOAT_LIST, required=False),
            _number("angle", FLOAT, 0.0, step=0.1),
            _number("center_x", FLOAT, 0.5, step=0.01),
            _number("center_y", FLOAT, 0.5, step=0.01),
            _number("radius", FLOAT, 0.5, minimum=0.001, step=0.01),
            _number("start_frame", INT, 0, minimum=0, step=1),
            _number("end_frame", INT, 9999, minimum=0, step=1),
            _combo(
                "transition_type",
                (
                    "horizontal_slide",
                    "vertical_slide",
                    "horizontal_bar",
                    "vertical_bar",
                    "center_box",
                    "horizontal_door",
                    "vertical_door",
                    "circle",
                    "fade",
                ),
                "horizontal_slide",
            ),
            _combo("timing_function", ("linear", "in", "out", "in_out"), "linear"),
            _number("tile_size", INT, 32, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("seed", INT, 0, minimum=0, maximum=0x1FFFFFFFFFFFFF, step=1),
            _combo("noise_mode", ("uniform", "binary"), "uniform", advanced=True),
            _number("noise_density", FLOAT, 0.5, minimum=0.0, maximum=1.0, step=0.01),
            InputSpec("invert", BOOLEAN, required=False, default=False),
        )
        by_id = {item.id: item for item in inputs}

        def option(key: str, *ids: str) -> DynamicComboOption:
            return DynamicComboOption(key, tuple(by_id[id_] for id_ in ids))

        foreground_background = ("foreground", "background")
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.mask.make",
                display_name="Create Mask",
                category="mask/create",
                inputs=tuple(by_id[id_] for id_ in ("width", "height", "batch_size", "invert")),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        (
                            option("solid", "foreground"),
                            *(
                                option(
                                    key,
                                    "foreground",
                                    "background",
                                    "x",
                                    "y",
                                    "shape_origin",
                                    "shape_width",
                                    "shape_height",
                                    "grow",
                                )
                                for key in ("rectangle", "ellipse", "triangle")
                            ),
                            option("polygon", "foreground", "background", "points_x", "points_y"),
                            option("region", "foreground", "background", "region"),
                            option("linear_gradient", *foreground_background, "angle"),
                            option(
                                "radial_gradient",
                                *foreground_background,
                                "center_x",
                                "center_y",
                                "radius",
                            ),
                            option("frame_gradient"),
                            option(
                                "transition",
                                *foreground_background,
                                "start_frame",
                                "end_frame",
                                "transition_type",
                                "timing_function",
                            ),
                            option("checkerboard", *foreground_background, "tile_size"),
                            DynamicComboOption(
                                "noise",
                                (
                                    by_id["foreground"],
                                    by_id["background"],
                                    by_id["seed"],
                                    DynamicComboSpec(
                                        "noise_mode",
                                        (option("uniform"), option("binary", "noise_density")),
                                        default="uniform",
                                    ),
                                ),
                            ),
                        ),
                        default="solid",
                    ),
                ),
                outputs=(
                    OutputSpec("mask", MASK, preview=True),
                    OutputSpec("inverse_mask", MASK, preview=True),
                ),
                search_terms=(
                    "SolidMask",
                    "Constant Mask",
                    "Create Rect Mask",
                    "CreateShapeMask",
                    "gradient mask",
                    "TransitionMask+",
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
        foreground: float = 1.0,
        background: float = 0.0,
        x: int = 0,
        y: int = 0,
        shape_origin: str = "top_left",
        shape_width: int = 256,
        shape_height: int = 256,
        grow: int = 0,
        region: object = ABSENT,
        points_x: object = ABSENT,
        points_y: object = ABSENT,
        angle: float = 0.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius: float = 0.5,
        start_frame: int = 0,
        end_frame: int = 9999,
        transition_type: str = "horizontal_slide",
        timing_function: str = "linear",
        tile_size: int = 32,
        seed: int = 0,
        noise_mode: str = "uniform",
        noise_density: float = 0.5,
        invert: bool = False,
    ) -> Mapping[str, object]:
        _validate_canvas(width, height, batch_size)
        if not all(math.isfinite(value) for value in (foreground, background)):
            raise ValueError("mask values must be finite")
        if shape_origin not in ("top_left", "center"):
            raise ValueError(f"unknown shape origin: {shape_origin}")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(noise_density) or not 0.0 <= noise_density <= 1.0:
            raise ValueError("noise_density must be finite and between 0 and 1")
        output = np.empty((batch_size, height, width), dtype=np.float32)
        rows, columns = np.ogrid[:height, :width]
        polygon_x = _coordinate_values(points_x, "points_x")
        polygon_y = _coordinate_values(points_y, "points_y")
        if len(polygon_x) != len(polygon_y):
            raise ValueError("points_x and points_y must contain the same number of coordinates")
        if not all(math.isfinite(value) for value in (*polygon_x, *polygon_y)):
            raise ValueError("polygon coordinates must be finite")
        for batch_index in range(batch_size):
            current_width = max(0, shape_width + batch_index * grow)
            current_height = max(0, shape_height + batch_index * grow)
            if operation == "solid":
                plane = np.full((height, width), foreground, dtype=np.float32)
            elif operation in ("rectangle", "ellipse", "triangle"):
                plane = np.full((height, width), background, dtype=np.float32)
                left = x - current_width / 2 if shape_origin == "center" else x
                top = y - current_height / 2 if shape_origin == "center" else y
                if current_width > 0 and current_height > 0:
                    if operation == "rectangle":
                        selected = (
                            (columns + 0.5 >= left)
                            & (columns + 0.5 < left + current_width)
                            & (rows + 0.5 >= top)
                            & (rows + 0.5 < top + current_height)
                        )
                        plane[selected] = foreground
                    elif operation == "ellipse":
                        center_column = left + current_width / 2
                        center_row = top + current_height / 2
                        selected = (
                            np.square((columns + 0.5 - center_column) / (current_width / 2))
                            + np.square((rows + 0.5 - center_row) / (current_height / 2))
                            <= 1.0
                        )
                        plane[selected] = foreground
                    else:
                        raster = Image.new("F", (width, height), background)
                        ImageDraw.Draw(raster).polygon(
                            (
                                (left + current_width / 2, top),
                                (left, top + current_height),
                                (left + current_width, top + current_height),
                            ),
                            fill=foreground,
                        )
                        plane = np.asarray(raster, dtype=np.float32)
            elif operation == "polygon":
                if len(polygon_x) < 3:
                    raise ValueError("polygon masks require at least three points")
                raster = Image.new("F", (width, height), background)
                ImageDraw.Draw(raster).polygon(
                    tuple(zip(polygon_x, polygon_y, strict=True)), fill=foreground
                )
                plane = np.asarray(raster, dtype=np.float32)
            elif operation == "region":
                if not isinstance(region, Region):
                    raise TypeError("region operation requires a Region")
                plane = np.full((height, width), background, dtype=np.float32)
                left = max(0, math.floor(region.x))
                top = max(0, math.floor(region.y))
                right = min(width, math.ceil(region.right))
                bottom = min(height, math.ceil(region.bottom))
                if right > left and bottom > top:
                    plane[top:bottom, left:right] = foreground
            elif operation == "linear_gradient":
                ramp = _linear_ramp(width, height, angle)
                plane = background + ramp * (foreground - background)
            elif operation == "radial_gradient":
                ramp = _radial_ramp(width, height, center_x, center_y, radius)
                plane = foreground + ramp * (background - foreground)
            elif operation == "frame_gradient":
                ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)
                plane = np.broadcast_to(ramp - batch_index / batch_size, (height, width))
            elif operation == "transition":
                transition_end = min(batch_size, end_frame)
                if start_frame < 0 or end_frame < 0:
                    raise ValueError("transition frame bounds must be non-negative")
                if start_frame >= transition_end:
                    plane = np.full(
                        (height, width),
                        background if batch_index < start_frame else foreground,
                        dtype=np.float32,
                    )
                elif batch_index < start_frame:
                    plane = np.full((height, width), background, dtype=np.float32)
                elif batch_index >= transition_end:
                    plane = np.full((height, width), foreground, dtype=np.float32)
                else:
                    transition_length = transition_end - start_frame
                    position = batch_index - start_frame
                    progress = _transition_progress(position, transition_length, timing_function)
                    selected = _transition_selection(
                        width, height, progress, transition_type
                    ).astype(np.float32)
                    plane = background + selected * (foreground - background)
            elif operation == "checkerboard":
                if tile_size < 1:
                    raise ValueError("tile_size must be positive")
                selected = ((columns // tile_size) + (rows // tile_size)) % 2 == 0
                plane = np.where(selected, foreground, background).astype(np.float32)
            elif operation == "noise":
                if noise_mode not in ("uniform", "binary"):
                    raise ValueError(f"unknown noise mode: {noise_mode}")
                indices = np.arange(height * width, dtype=np.uint64).reshape(height, width)
                values = indices + np.uint64(seed) + np.uint64(batch_index * 0x9E3779B1)
                values ^= values >> np.uint64(16)
                values *= np.uint64(0x7FEB352D)
                values ^= values >> np.uint64(15)
                values *= np.uint64(0x846CA68B)
                values ^= values >> np.uint64(16)
                unit = np.asarray(values & np.uint64(0xFFFFFF), dtype=np.float32) / 0xFFFFFF
                if noise_mode == "binary":
                    unit = np.asarray(unit < noise_density, dtype=np.float32)
                plane = background + unit * (foreground - background)
            else:
                raise ValueError(f"unknown mask creation operation: {operation}")
            output[batch_index] = plane
        output = np.ascontiguousarray(output, dtype=np.float32)
        if type(invert) is not bool:
            raise TypeError("invert must be a boolean")
        if invert:
            output = np.ascontiguousarray(1.0 - output)
        return cls.outputs(mask=output, inverse_mask=np.ascontiguousarray(1.0 - output))


class TextMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        foreground = _number("foreground", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01)
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.mask.text",
                display_name="Create Text Mask",
                category="mask/create",
                inputs=(
                    InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                    _number("width", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                    _number("height", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                    _number("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
                    _number("x", INT, 0, step=1),
                    _number("y", INT, 0, step=1),
                    _number("font_size", INT, 32, minimum=1, maximum=1024, step=1),
                    _number("background", FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.01),
                    InputSpec(
                        "color",
                        STRING,
                        required=False,
                        default="#ffffff",
                        widget=ColorWidget(),
                    ),
                    _number("line_spacing", INT, 4, minimum=0, maximum=1024, step=1, advanced=True),
                    _number("start_rotation", FLOAT, 0.0, step=0.1),
                    _number("end_rotation", FLOAT, 0.0, step=0.1),
                    InputSpec("invert", BOOLEAN, required=False, default=False),
                ),
                combos=(
                    DynamicComboSpec(
                        "mask_value",
                        (
                            DynamicComboOption("foreground", (foreground,)),
                            DynamicComboOption("color_red"),
                        ),
                        default="foreground",
                    ),
                ),
                outputs=(
                    OutputSpec("mask", MASK, preview=True),
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("inverse_mask", MASK, preview=True),
                ),
                search_terms=("CreateTextMask", "Mask By Text", "text raster"),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        text: str,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        x: int = 0,
        y: int = 0,
        font_size: int = 32,
        foreground: float = 1.0,
        background: float = 0.0,
        color: str = "#ffffff",
        mask_value: str = "foreground",
        line_spacing: int = 4,
        start_rotation: float = 0.0,
        end_rotation: float = 0.0,
        invert: bool = False,
    ) -> Mapping[str, object]:
        _validate_canvas(width, height, batch_size, 3)
        if font_size < 1 or line_spacing < 0:
            raise ValueError("font_size must be positive and line_spacing must be non-negative")
        if not all(
            math.isfinite(value) for value in (foreground, background, start_rotation, end_rotation)
        ):
            raise ValueError("mask values must be finite")
        rgba = _parse_color(color)
        if mask_value == "color_red":
            foreground = float(rgba[0])
        elif mask_value != "foreground":
            raise ValueError(f"unknown text mask value policy: {mask_value}")
        raster = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(raster)
        font = ImageFont.load_default(size=font_size)
        draw.multiline_text((x, y), text, fill=255, font=font, spacing=line_spacing)
        masks = np.empty((batch_size, height, width), dtype=np.float32)
        images = np.empty((batch_size, height, width, 3), dtype=np.float32)
        for batch_index in range(batch_size):
            progress = batch_index / (batch_size - 1) if batch_size > 1 else 0.0
            rotation = start_rotation + progress * (end_rotation - start_rotation)
            frame = (
                raster
                if rotation == 0.0
                else raster.rotate(rotation, resample=Image.Resampling.BICUBIC)
            )
            coverage = np.asarray(frame, dtype=np.float32) / 255.0
            masks[batch_index] = background + coverage * (foreground - background)
            images[batch_index] = coverage[:, :, None] * rgba[:3]
        if type(invert) is not bool:
            raise TypeError("invert must be a boolean")
        if invert:
            masks = 1.0 - masks
            images = 1.0 - images
        masks = np.ascontiguousarray(masks, dtype=np.float32)
        return cls.outputs(
            mask=masks,
            image=np.ascontiguousarray(images),
            inverse_mask=np.ascontiguousarray(1.0 - masks),
        )


def _validate_edge_policy(edge_policy: str) -> EdgePolicy:
    if edge_policy not in ("constant", "replicate", "reflect", "wrap"):
        raise ValueError(f"unknown edge policy: {edge_policy}")
    return edge_policy


def _pad_spatial(
    mask: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    pad_width = ((0, 0), (top, bottom), (left, right))
    if edge_policy == "constant":
        return np.pad(mask, pad_width, mode="constant", constant_values=edge_value)
    if edge_policy == "replicate" or (
        edge_policy == "reflect" and (mask.shape[1] == 1 or mask.shape[2] == 1)
    ):
        return np.pad(mask, pad_width, mode="edge")
    if edge_policy == "reflect":
        return np.pad(mask, pad_width, mode="symmetric")
    if edge_policy == "wrap":
        return np.pad(mask, pad_width, mode="wrap")
    raise ValueError(f"unknown edge policy: {edge_policy}")


def _footprint(radius: int, tapered_corners: bool) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be positive")
    coordinates = np.arange(-radius, radius + 1)
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    if tapered_corners:
        return np.abs(rows) + np.abs(columns) <= radius
    return np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.bool_)


def _morphology_pass(
    mask: np.ndarray,
    footprint: np.ndarray,
    reduction: Literal["min", "max"],
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    radius_y, radius_x = footprint.shape[0] // 2, footprint.shape[1] // 2
    padded = _pad_spatial(
        mask,
        radius_y,
        radius_y,
        radius_x,
        radius_x,
        edge_policy,
        edge_value,
    )
    coordinates = np.argwhere(footprint)
    if coordinates.size == 0:
        raise ValueError("morphology footprint must not be empty")
    row, column = (int(value) for value in coordinates[0])
    output = np.array(
        padded[:, row : row + mask.shape[1], column : column + mask.shape[2]],
        copy=True,
    )
    for coordinate in coordinates[1:]:
        row, column = (int(value) for value in coordinate)
        selected = padded[:, row : row + mask.shape[1], column : column + mask.shape[2]]
        if reduction == "max":
            np.maximum(output, selected, out=output)
        else:
            np.minimum(output, selected, out=output)
    return output


def _morphology(
    mask: np.ndarray,
    footprint: np.ndarray,
    reduction: Literal["min", "max"],
    iterations: int,
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    output = mask
    for _ in range(iterations):
        output = _morphology_pass(output, footprint, reduction, edge_policy, edge_value)
    return output


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    if radius < 0:
        raise ValueError("blur radius must be non-negative")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if radius == 0:
        return np.ones(1, dtype=np.float32)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-np.square(coordinates) / (2 * sigma * sigma))
    return np.asarray(kernel / kernel.sum(), dtype=np.float32)


def _convolve_axis(
    mask: np.ndarray,
    kernel: np.ndarray,
    axis: Literal[1, 2],
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    radius = kernel.size // 2
    if radius == 0:
        return mask
    padded = _pad_spatial(
        mask,
        radius if axis == 1 else 0,
        radius if axis == 1 else 0,
        radius if axis == 2 else 0,
        radius if axis == 2 else 0,
        edge_policy,
        edge_value,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)
    return np.asarray(np.tensordot(windows, kernel, axes=([-1], [0])), dtype=np.float32)


def _blur(
    mask: np.ndarray,
    radius: int,
    sigma: float,
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    kernel = _gaussian_kernel(radius, sigma)
    output = _convolve_axis(mask, kernel, 2, edge_policy, edge_value)
    return _convolve_axis(output, kernel, 1, edge_policy, edge_value)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    output = np.empty_like(mask)
    for batch_index, frame in enumerate(mask):
        background = frame <= 0
        connected = np.zeros_like(background)
        pending: deque[tuple[int, int]] = deque()
        for row in (0, frame.shape[0] - 1):
            for column in range(frame.shape[1]):
                if background[row, column] and not connected[row, column]:
                    connected[row, column] = True
                    pending.append((row, column))
        for column in (0, frame.shape[1] - 1):
            for row in range(frame.shape[0]):
                if background[row, column] and not connected[row, column]:
                    connected[row, column] = True
                    pending.append((row, column))
        while pending:
            row, column = pending.popleft()
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < frame.shape[0]
                    and 0 <= next_column < frame.shape[1]
                    and background[next_row, next_column]
                    and not connected[next_row, next_column]
                ):
                    connected[next_row, next_column] = True
                    pending.append((next_row, next_column))
        output[batch_index] = np.logical_not(connected)
    return output


def _remove_small_components(
    mask: np.ndarray,
    minimum_area: int,
    threshold: float,
    connectivity: int,
) -> np.ndarray:
    output = mask.copy()
    for batch_index, frame in enumerate(mask):
        selected = np.asarray(frame > threshold, dtype=np.uint8)
        _count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
            selected, connectivity=connectivity
        )
        label_indices = np.asarray(labels, dtype=np.int32)
        retained = np.asarray(statistics, dtype=np.int32)[:, cv2.CC_STAT_AREA] >= minimum_area
        retained[0] = False
        removed = np.logical_and(selected != 0, np.logical_not(retained[label_indices]))
        output[batch_index][removed] = 0.0
    return output


def _offset(
    mask: np.ndarray,
    x: int,
    y: int,
    edge_policy: EdgePolicy,
    edge_value: float,
) -> np.ndarray:
    if edge_policy == "wrap":
        return np.ascontiguousarray(np.roll(mask, shift=(y, x), axis=(1, 2)))
    height, width = int(mask.shape[1]), int(mask.shape[2])
    top, bottom = max(y, 0), max(-y, 0)
    left, right = max(x, 0), max(-x, 0)
    padded = _pad_spatial(mask, top, bottom, left, right, edge_policy, edge_value)
    row_start = bottom
    column_start = right
    return np.ascontiguousarray(
        padded[:, row_start : row_start + height, column_start : column_start + width]
    )


def _blockify(mask: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    output = np.empty_like(mask)
    for top in range(0, mask.shape[1], kernel_size):
        for left in range(0, mask.shape[2], kernel_size):
            block = mask[:, top : top + kernel_size, left : left + kernel_size]
            if mode == "mean":
                value = block.mean(axis=(1, 2), keepdims=True, dtype=np.float64)
            elif mode == "min":
                value = block.min(axis=(1, 2), keepdims=True)
            elif mode == "max":
                value = block.max(axis=(1, 2), keepdims=True)
            else:
                raise ValueError(f"unknown block mode: {mode}")
            output[:, top : top + kernel_size, left : left + kernel_size] = value
    return output


def _check_kernel_work(estimated_samples: int) -> None:
    limit = MAX_IMAGE_BYTES // np.dtype(np.float32).itemsize
    if estimated_samples > limit:
        raise ValueError(f"mask kernel work exceeds the {limit}-sample operation limit")


def dilate_mask(mask: np.ndarray, radius: int, tapered_corners: bool) -> np.ndarray:
    """Grow (positive radius) or erode (negative) a BHW mask with zero edges."""
    if radius == 0:
        return mask
    footprint = _footprint(1, tapered_corners)
    passes = abs(radius)
    _check_kernel_work(int(mask.size) * passes * int(footprint.sum()))
    return _morphology(mask, footprint, "max" if radius > 0 else "min", passes, "constant", 0.0)


class MaskMorphology(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        dynamic_inputs = (
            _number("threshold", FLOAT, 0.5, minimum=0.0, maximum=1.0, step=0.01),
            _number(
                "minimum_area",
                INT,
                64,
                minimum=1,
                maximum=MAX_DIMENSION * MAX_DIMENSION,
                step=1,
            ),
            _combo("connectivity", ("4", "8"), "8", advanced=True),
            _number("radius", INT, 1, minimum=-4096, maximum=4096, step=1),
            _number("blur_radius", INT, 0, minimum=0, maximum=4096, step=1),
            _number("blur_amount", FLOAT, 0.0, minimum=0.0, maximum=100.0, step=0.1),
            _number(
                "incremental_expandrate",
                FLOAT,
                0.0,
                minimum=0.0,
                maximum=100.0,
                step=0.1,
                advanced=True,
            ),
            InputSpec("flip_input", BOOLEAN, required=False, default=False, advanced=True),
            _number("lerp_alpha", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01, advanced=True),
            _number(
                "decay_factor",
                FLOAT,
                1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                advanced=True,
            ),
            InputSpec("fill_holes", BOOLEAN, required=False, default=False, advanced=True),
            _number("kernel_size", INT, 3, minimum=1, maximum=4097, step=2),
            _number("iterations", INT, 1, minimum=1, maximum=4096, step=1),
            InputSpec("tapered_corners", BOOLEAN, required=False, default=True, advanced=True),
            _number("sigma", FLOAT, 1.0, minimum=0.001, maximum=4096.0, step=0.01, advanced=True),
            _number("edge_value", FLOAT, 0.0, step=0.01, advanced=True),
            _number("left", INT, 0, minimum=0, maximum=MAX_DIMENSION, step=1),
            _number("top", INT, 0, minimum=0, maximum=MAX_DIMENSION, step=1),
            _number("right", INT, 0, minimum=0, maximum=MAX_DIMENSION, step=1),
            _number("bottom", INT, 0, minimum=0, maximum=MAX_DIMENSION, step=1),
            _number("x", INT, 0, step=1),
            _number("y", INT, 0, step=1),
            _number("width", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("height", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
            _number("input_low", FLOAT, 0.0, step=0.01),
            _number("input_high", FLOAT, 1.0, step=0.01),
            _number("output_low", FLOAT, 0.0, step=0.01),
            _number("output_high", FLOAT, 1.0, step=0.01),
            InputSpec("clamp", BOOLEAN, required=False, default=True, advanced=True),
            _combo("block_mode", ("mean", "min", "max"), "mean", advanced=True),
        )
        by_id = {item.id: item for item in dynamic_inputs}

        def edge_policy() -> DynamicComboSpec:
            return DynamicComboSpec(
                "edge_policy",
                (
                    DynamicComboOption("constant", (by_id["edge_value"],)),
                    DynamicComboOption("replicate"),
                    DynamicComboOption("reflect"),
                    DynamicComboOption("wrap"),
                ),
                default="constant",
            )

        def option(
            key: str,
            *ids: str,
            edge: bool = False,
        ) -> DynamicComboOption:
            inputs: tuple[InputSpec | DynamicComboSpec, ...] = tuple(
                by_id[input_id] for input_id in ids
            )
            if edge:
                inputs = (*inputs, edge_policy())
            return DynamicComboOption(key, inputs)

        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.mask.morphology",
                display_name="Mask Morphology",
                category="mask/operations",
                inputs=(InputSpec("mask", MASK),),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        (
                            option("threshold", "threshold"),
                            option(
                                "remove_small_components",
                                "minimum_area",
                                "threshold",
                                "connectivity",
                            ),
                            option("fill_holes"),
                            option(
                                "grow_erode", "radius", "iterations", "tapered_corners", edge=True
                            ),
                            option(
                                "grow_blur",
                                "radius",
                                "blur_radius",
                                "blur_amount",
                                "incremental_expandrate",
                                "flip_input",
                                "lerp_alpha",
                                "decay_factor",
                                "fill_holes",
                                "iterations",
                                "tapered_corners",
                                "sigma",
                                edge=True,
                            ),
                            option(
                                "open", "kernel_size", "iterations", "tapered_corners", edge=True
                            ),
                            option(
                                "close", "kernel_size", "iterations", "tapered_corners", edge=True
                            ),
                            option("feather_edges", "left", "top", "right", "bottom"),
                            DynamicComboOption(
                                "blur",
                                (
                                    by_id["radius"],
                                    dataclasses.replace(by_id["sigma"], advanced=False),
                                    edge_policy(),
                                ),
                            ),
                            option("offset", "x", "y", edge=True),
                            option(
                                "remap",
                                "input_low",
                                "input_high",
                                "output_low",
                                "output_high",
                                "clamp",
                            ),
                            option("round", "radius", "iterations", edge=True),
                            option("block", "kernel_size", "block_mode"),
                            option("invert"),
                            option("crop", "x", "y", "width", "height"),
                        ),
                        default="threshold",
                    ),
                ),
                outputs=(
                    OutputSpec("mask", MASK, preview=True),
                    OutputSpec("inverse_mask", MASK, preview=True),
                ),
                search_terms=(
                    "ThresholdMask",
                    "GrowMask",
                    "FeatherMask",
                    "ImpactDilateMask",
                    "ImpactGaussianBlurMask",
                    "MaskBlur+",
                    "MaskSmooth+",
                    "GrowMaskWithBlur",
                    "OffsetMask",
                    "RemapMaskRange",
                    "RoundMask",
                    "BlockifyMask",
                    "Remove Small Mask Components",
                    "Fill Mask Holes",
                ),
            ),
            current_only_inputs=frozenset(("minimum_area", "connectivity")),
            current_only_choices={
                "operation": frozenset(("remove_small_components", "fill_holes"))
            },
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        mask: object,
        operation: str = "threshold",
        threshold: float = 0.5,
        minimum_area: int = 64,
        connectivity: str = "8",
        radius: int = 1,
        blur_radius: int = 0,
        blur_amount: float = 0.0,
        incremental_expandrate: float = 0.0,
        flip_input: bool = False,
        lerp_alpha: float = 1.0,
        decay_factor: float = 1.0,
        fill_holes: bool = False,
        kernel_size: int = 3,
        iterations: int = 1,
        tapered_corners: bool = True,
        sigma: float = 1.0,
        edge_policy: str = "constant",
        edge_value: float = 0.0,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
        x: int = 0,
        y: int = 0,
        width: int = 512,
        height: int = 512,
        input_low: float = 0.0,
        input_high: float = 1.0,
        output_low: float = 0.0,
        output_high: float = 1.0,
        clamp: bool = True,
        block_mode: str = "mean",
    ) -> Mapping[str, object]:
        source = _mask_array(mask)
        if iterations < 1:
            raise ValueError("iterations must be positive")
        policy = _validate_edge_policy(edge_policy)
        if not math.isfinite(edge_value):
            raise ValueError("edge_value must be finite")
        if any(
            type(value) is not bool for value in (tapered_corners, clamp, flip_input, fill_holes)
        ):
            raise TypeError("mask morphology flags must be booleans")
        if operation == "threshold":
            if not math.isfinite(threshold):
                raise ValueError("threshold must be finite")
            output = (source > threshold).astype(np.float32)
        elif operation == "remove_small_components":
            if type(minimum_area) is not int or minimum_area < 1:
                raise ValueError("minimum_area must be a positive integer")
            if not math.isfinite(threshold):
                raise ValueError("threshold must be finite")
            if connectivity not in ("4", "8"):
                raise ValueError("connectivity must be 4 or 8")
            output = _remove_small_components(source, minimum_area, threshold, int(connectivity))
        elif operation == "fill_holes":
            output = _fill_holes(source)
        elif operation in ("grow_erode", "grow_blur"):
            if operation == "grow_erode":
                if radius == 0:
                    output = source.copy()
                else:
                    footprint = _footprint(1, tapered_corners)
                    passes = abs(radius) * iterations
                    _check_kernel_work(int(source.size) * passes * int(footprint.sum()))
                    output = _morphology(
                        source,
                        footprint,
                        "max" if radius > 0 else "min",
                        passes,
                        policy,
                        edge_value,
                    )
            else:
                if not all(
                    math.isfinite(value)
                    for value in (blur_amount, incremental_expandrate, lerp_alpha, decay_factor)
                ) or not (0.0 <= lerp_alpha <= 1.0 and 0.0 <= decay_factor <= 1.0):
                    raise ValueError("grow/blur controls must be finite and within their ranges")
                working = 1.0 - source if flip_input else source
                frames: list[np.ndarray] = []
                previous: np.ndarray | None = None
                current_radius = float(radius)
                rounded_radii: list[int] = []
                for _ in working:
                    rounded_radii.append(round(current_radius))
                    current_radius += math.copysign(
                        abs(incremental_expandrate), current_radius if current_radius else 1.0
                    )
                footprint = _footprint(1, tapered_corners)
                morphology_work = (
                    int(source.shape[1])
                    * int(source.shape[2])
                    * sum(abs(value) * iterations for value in rounded_radii)
                    * int(footprint.sum())
                )
                effective_blur_radius = math.ceil(blur_amount * 3) if blur_amount else blur_radius
                if effective_blur_radius < 0:
                    raise ValueError("blur radius must be non-negative")
                blur_work = (
                    int(source.size) * 2 * (effective_blur_radius * 2 + 1)
                    if effective_blur_radius
                    else 0
                )
                _check_kernel_work(morphology_work + blur_work)
                for frame, rounded_radius in zip(working, rounded_radii, strict=True):
                    expanded = frame[None, :, :]
                    if rounded_radius:
                        expanded = _morphology(
                            expanded,
                            footprint,
                            "max" if rounded_radius > 0 else "min",
                            abs(rounded_radius) * iterations,
                            policy,
                            edge_value,
                        )
                    if fill_holes:
                        expanded = _fill_holes(expanded)
                    current = expanded[0]
                    if previous is not None and lerp_alpha < 1.0:
                        current = lerp_alpha * current + (1.0 - lerp_alpha) * previous
                    if previous is not None and decay_factor < 1.0:
                        current = current + decay_factor * previous
                        maximum = float(current.max())
                        if maximum > 0:
                            current = current / maximum
                    previous = current
                    frames.append(current)
                output = np.stack(frames)
                if blur_amount:
                    output = _blur(
                        output,
                        effective_blur_radius,
                        blur_amount,
                        policy,
                        edge_value,
                    )
                elif blur_radius:
                    output = _blur(output, effective_blur_radius, sigma, policy, edge_value)
        elif operation in ("open", "close"):
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be a positive odd integer")
            footprint = (
                _footprint(1, tapered_corners)
                if kernel_size > 1
                else np.ones((1, 1), dtype=np.bool_)
            )
            morphology_iterations = (kernel_size // 2) * iterations if kernel_size > 1 else 1
            _check_kernel_work(int(source.size) * morphology_iterations * int(footprint.sum()) * 2)
            reductions: tuple[Literal["min", "max"], Literal["min", "max"]]
            reductions = ("min", "max") if operation == "open" else ("max", "min")
            output = _morphology(
                source, footprint, reductions[0], morphology_iterations, policy, edge_value
            )
            output = _morphology(
                output, footprint, reductions[1], morphology_iterations, policy, edge_value
            )
        elif operation == "feather_edges":
            if min(left, top, right, bottom) < 0:
                raise ValueError("feather extents must be non-negative")
            output = source.copy()
            height, width = source.shape[1:]
            if left:
                extent = min(left, width)
                output[:, :, :extent] *= (np.arange(extent, dtype=np.float32) + 1) / extent
            if right:
                extent = min(right, width)
                output[:, :, width - extent :] *= (np.arange(extent, dtype=np.float32) + 1)[
                    ::-1
                ] / extent
            if top:
                extent = min(top, height)
                output[:, :extent, :] *= ((np.arange(extent, dtype=np.float32) + 1) / extent)[
                    None, :, None
                ]
            if bottom:
                extent = min(bottom, height)
                output[:, height - extent :, :] *= (
                    (np.arange(extent, dtype=np.float32) + 1)[::-1] / extent
                )[None, :, None]
        elif operation == "blur":
            if radius >= 0:
                _check_kernel_work(int(source.size) * 2 * (radius * 2 + 1))
            output = _blur(source, radius, sigma, policy, edge_value)
        elif operation == "offset":
            output = _offset(source, x, y, policy, edge_value)
        elif operation == "remap":
            values = (input_low, input_high, output_low, output_high)
            if not all(math.isfinite(value) for value in values) or input_high == input_low:
                raise ValueError("remap bounds must be finite and input bounds must differ")
            scale = (source - input_low) / (input_high - input_low)
            if clamp:
                scale = np.clip(scale, 0.0, 1.0)
            output = output_low + scale * (output_high - output_low)
        elif operation == "round":
            if radius < 1:
                raise ValueError("round radius must be positive")
            footprint = _footprint(1, True)
            morphology_iterations = radius * iterations
            _check_kernel_work(int(source.size) * morphology_iterations * int(footprint.sum()) * 4)
            output = _morphology(
                source, footprint, "max", morphology_iterations, policy, edge_value
            )
            output = _morphology(
                output, footprint, "min", morphology_iterations, policy, edge_value
            )
            output = _morphology(
                output, footprint, "min", morphology_iterations, policy, edge_value
            )
            output = _morphology(
                output, footprint, "max", morphology_iterations, policy, edge_value
            )
        elif operation == "block":
            output = _blockify(source, kernel_size, block_mode)
        elif operation == "invert":
            output = 1.0 - source
        elif operation == "crop":
            if x < 0 or y < 0 or width < 1 or height < 1:
                raise ValueError("crop coordinates must be non-negative and dimensions positive")
            output = source[:, y : y + height, x : x + width]
            if output.shape[1] < 1 or output.shape[2] < 1:
                raise ValueError("crop must overlap the source mask")
        else:
            raise ValueError(f"unknown mask morphology operation: {operation}")
        output = np.ascontiguousarray(output, dtype=np.float32)
        return cls.outputs(
            mask=copy_media_semantics(mask, output),
            inverse_mask=copy_media_semantics(mask, np.ascontiguousarray(1.0 - output)),
        )


def _broadcast_pair(
    destination: np.ndarray, source: np.ndarray, batch_policy: str
) -> tuple[np.ndarray, np.ndarray]:
    if batch_policy not in ("singleton_broadcast", "source_singleton"):
        raise ValueError(f"unknown mask combine batch policy: {batch_policy}")
    destination_batch, source_batch = int(destination.shape[0]), int(source.shape[0])
    if destination_batch == source_batch:
        return destination, source
    if batch_policy == "singleton_broadcast" and destination_batch == 1:
        return np.broadcast_to(destination, (source_batch, *destination.shape[1:])), source
    if source_batch == 1:
        return destination, np.broadcast_to(source, (destination_batch, *source.shape[1:]))
    raise ValueError(
        f"mask batches do not satisfy {batch_policy}, got {destination_batch} and {source_batch}"
    )


class MaskCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mask.combine",
            display_name="Combine Masks",
            category="mask/operations",
            inputs=(
                InputSpec("destination", MASK),
                InputSpec("source", MASK),
                _combo(
                    "operation",
                    ("multiply", "add", "subtract", "min", "max", "and", "or", "xor"),
                    "multiply",
                ),
                _number("x", INT, 0, step=1),
                _number("y", INT, 0, step=1),
                _combo(
                    "batch_policy",
                    ("singleton_broadcast", "source_singleton"),
                    "singleton_broadcast",
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("mask", MASK, preview=True),),
            search_terms=(
                "MaskComposite",
                "Combine Masks",
                "AddMask",
                "SubtractMask",
                "BitwiseAndMask",
                "Masks Add",
                "Masks Subtract",
                "Masks Combine Regions",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        destination: object,
        source: object,
        operation: str = "multiply",
        x: int = 0,
        y: int = 0,
        batch_policy: str = "singleton_broadcast",
    ) -> Mapping[str, object]:
        destination_array, source_array = _broadcast_pair(
            _mask_array(destination, subject="destination"),
            _mask_array(source, subject="source"),
            batch_policy,
        )
        output = np.array(destination_array, copy=True)
        destination_height, destination_width = output.shape[1:]
        source_height, source_width = source_array.shape[1:]
        destination_left, destination_top = max(0, x), max(0, y)
        destination_right = min(destination_width, x + source_width)
        destination_bottom = min(destination_height, y + source_height)
        if destination_right > destination_left and destination_bottom > destination_top:
            source_left, source_top = destination_left - x, destination_top - y
            source_right = source_left + destination_right - destination_left
            source_bottom = source_top + destination_bottom - destination_top
            destination_view = output[
                :, destination_top:destination_bottom, destination_left:destination_right
            ]
            source_view = source_array[:, source_top:source_bottom, source_left:source_right]
            if operation == "multiply":
                combined = destination_view * source_view
            elif operation == "add":
                combined = destination_view + source_view
            elif operation == "subtract":
                combined = destination_view - source_view
            elif operation == "min":
                combined = np.minimum(destination_view, source_view)
            elif operation == "max":
                combined = np.maximum(destination_view, source_view)
            elif operation in ("and", "or", "xor"):
                left = np.rint(destination_view).astype(np.bool_)
                right = np.rint(source_view).astype(np.bool_)
                if operation == "and":
                    combined = np.logical_and(left, right)
                elif operation == "or":
                    combined = np.logical_or(left, right)
                else:
                    combined = np.logical_xor(left, right)
            else:
                raise ValueError(f"unknown mask combine operation: {operation}")
            destination_view[...] = np.asarray(combined, dtype=np.float32)
        elif operation not in ("multiply", "add", "subtract", "min", "max", "and", "or", "xor"):
            raise ValueError(f"unknown mask combine operation: {operation}")
        return cls.outputs(
            mask=copy_media_semantics(destination, np.ascontiguousarray(np.clip(output, 0.0, 1.0)))
        )


class ImageToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        channel = _combo(
            "channel",
            ("red", "green", "blue", "alpha", "transparency", "luma"),
            "red",
        )
        color = InputSpec(
            "color",
            STRING,
            required=False,
            default="#000000",
            widget=ColorWidget(),
        )
        color_value = _number("color_value", INT, 0, minimum=0, maximum=0xFFFFFF, step=1)
        red = _number("red", INT, 0, minimum=0, maximum=255, step=1)
        green = _number("green", INT, 0, minimum=0, maximum=255, step=1)
        blue = _number("blue", INT, 0, minimum=0, maximum=255, step=1)
        tolerance = _number("tolerance", FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.001)
        metric = _combo(
            "metric",
            ("max_channel", "euclidean_rgb", "euclidean_rgb_sum", "euclidean_rgba"),
            "max_channel",
            advanced=True,
        )

        def color_source() -> DynamicComboSpec:
            return DynamicComboSpec(
                "color_source",
                (
                    DynamicComboOption("hex", (color,)),
                    DynamicComboOption("integer", (color_value,)),
                    DynamicComboOption("channels", (red, green, blue)),
                ),
                default="hex",
            )

        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.image.to_mask",
                display_name="Convert Image to Mask",
                category="mask/convert",
                inputs=(
                    InputSpec("image", IMAGE),
                    InputSpec("invert", BOOLEAN, required=False, default=False),
                ),
                combos=(
                    DynamicComboSpec(
                        "policy",
                        (
                            DynamicComboOption("channel", (channel,)),
                            DynamicComboOption("exact_color", (color_source(),)),
                            DynamicComboOption(
                                "tolerance_color", (color_source(), tolerance, metric)
                            ),
                        ),
                        default="channel",
                    ),
                ),
                outputs=(OutputSpec("mask", MASK, preview=True),),
                search_terms=(
                    "ImageToMask",
                    "ImageColorToMask",
                    "ColorToMask",
                    "MaskFromColor+",
                    "easy imageToMask",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        policy: str = "channel",
        channel: str = "red",
        color: str = "#000000",
        color_source: str = "hex",
        color_value: int = 0,
        red: int = 0,
        green: int = 0,
        blue: int = 0,
        tolerance: float = 0.0,
        metric: str = "max_channel",
        invert: bool = False,
    ) -> Mapping[str, object]:
        array = _image_array(image)
        channels = int(array.shape[3])
        if policy == "channel":
            indexes = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
            if channel == "transparency":
                if channels != 4:
                    raise ValueError("transparency extraction requires an RGBA image")
                output = 1.0 - array[:, :, :, 3]
            elif channel == "luma":
                if channels < 3:
                    output = array[:, :, :, 0]
                else:
                    output = (
                        array[:, :, :, 0] * 0.2126
                        + array[:, :, :, 1] * 0.7152
                        + array[:, :, :, 2] * 0.0722
                    )
            else:
                index = indexes.get(channel)
                if index is None:
                    raise ValueError(f"unknown image channel: {channel}")
                if index >= channels:
                    raise ValueError(f"{channel} extraction requires that image channel")
                output = array[:, :, :, index]
        elif policy in ("exact_color", "tolerance_color"):
            if color_source == "hex":
                target = _parse_color(color)
            elif color_source == "integer":
                target = _color_from_int(color_value)
            elif color_source == "channels":
                if any(
                    type(value) is not int or not 0 <= value <= 255 for value in (red, green, blue)
                ):
                    raise ValueError("red, green, and blue must be integers between 0 and 255")
                target = np.asarray(
                    [red / 255.0, green / 255.0, blue / 255.0, 1.0], dtype=np.float32
                )
            else:
                raise ValueError(f"unknown color source: {color_source}")
            if policy == "exact_color":
                count = min(3, channels)
                pixels = np.rint(np.clip(array[:, :, :, :count], 0.0, 1.0) * 255).astype(np.uint8)
                wanted = np.rint(target[:count] * 255).astype(np.uint8)
                output = np.all(pixels == wanted, axis=3).astype(np.float32)
            else:
                if not math.isfinite(tolerance) or tolerance < 0:
                    raise ValueError("tolerance must be finite and non-negative")
                if metric == "euclidean_rgba":
                    if channels != 4:
                        raise ValueError("euclidean_rgba requires an RGBA image")
                    difference = array - target
                    distance = np.sqrt(np.mean(np.square(difference), axis=3))
                else:
                    count = min(3, channels)
                    difference = array[:, :, :, :count] - target[:count]
                    if metric == "max_channel":
                        distance = np.max(np.abs(difference), axis=3)
                    elif metric == "euclidean_rgb":
                        distance = np.sqrt(np.mean(np.square(difference), axis=3))
                    elif metric == "euclidean_rgb_sum":
                        distance = np.sqrt(np.sum(np.square(difference), axis=3))
                    else:
                        raise ValueError(f"unknown color distance metric: {metric}")
                output = (distance <= tolerance).astype(np.float32)
        else:
            raise ValueError(f"unknown image-to-mask policy: {policy}")
        if type(invert) is not bool:
            raise TypeError("invert must be a boolean")
        if invert:
            output = 1.0 - output
        alpha = policy == "channel" and channel in ("alpha", "transparency")
        polarity = (
            "transparency" if alpha and ((channel == "transparency") != invert) else "coverage"
        )
        return cls.outputs(
            mask=annotate_mask(
                np.ascontiguousarray(output, dtype=np.float32),
                polarity=polarity,
                semantic="alpha" if alpha else "selection",
            )
        )


class MaskToImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        color = InputSpec(
            "color",
            STRING,
            required=False,
            default="#ffffff",
            widget=ColorWidget(),
        )
        alpha_polarity = _combo(
            "alpha_polarity",
            ("mask_is_opacity", "mask_is_transparency"),
            "mask_is_opacity",
            advanced=True,
        )
        return with_mask_polarity(
            with_v1_migration(
                NodeSchema(
                    version=2,
                    node_type="dinkster.mask.to_image",
                    display_name="Convert Mask to Image",
                    category="mask/convert",
                    inputs=(InputSpec("mask", MASK),),
                    combos=(
                        DynamicComboSpec(
                            "channels",
                            (
                                DynamicComboOption("rgb"),
                                DynamicComboOption("rgba", (color, alpha_polarity)),
                            ),
                            default="rgb",
                        ),
                    ),
                    outputs=(OutputSpec("image", IMAGE, preview=True),),
                    search_terms=("MaskToImage", "Convert Masks to Images", "mask preview"),
                )
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        mask: object,
        channels: str = "rgb",
        mask_polarity: str = "coverage",
        color: str = "#ffffff",
    ) -> Mapping[str, object]:
        mask_polarity = normalize_mask_polarity(mask_polarity)
        array = _mask_array(mask)
        if channels == "rgb":
            output = np.repeat(array[:, :, :, None], 3, axis=3)
        elif channels == "rgba":
            rgb = _parse_color(color)[:3]
            output = np.empty((*array.shape, 4), dtype=np.float32)
            output[:, :, :, :3] = rgb
            if mask_polarity == "coverage":
                output[:, :, :, 3] = array
            elif mask_polarity == "transparency":
                output[:, :, :, 3] = 1.0 - array
            else:
                raise ValueError(f"unknown mask polarity: {mask_polarity}")
        else:
            raise ValueError(f"unknown image channel layout: {channels}")
        return cls.outputs(image=np.ascontiguousarray(output, dtype=np.float32))


def _mask_bounds(mask: np.ndarray, threshold: float) -> tuple[Region, int]:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    selected = mask > threshold
    area = int(selected.sum(dtype=np.int64))
    spatial = np.any(selected, axis=0)
    rows, columns = np.nonzero(spatial)
    if rows.size == 0:
        return Region(0, 0, 0, 0), area
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return Region(left, top, right - left, bottom - top), area


class MaskInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mask.info",
            display_name="Mask Information",
            category="mask/inspect",
            inputs=(
                InputSpec("mask", MASK),
                _number("threshold", FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.01),
                _combo("empty", ("zero_region", "error"), "zero_region", advanced=True),
            ),
            outputs=(
                OutputSpec("mask", MASK, preview=True),
                OutputSpec("width", INT),
                OutputSpec("height", INT),
                OutputSpec("count", INT),
                OutputSpec("area", INT),
                OutputSpec("bounds", REGION),
            ),
            search_terms=("GetMaskSizeAndCount", "mask area", "mask bounds", "isMaskEmpty"),
        )

    @classmethod
    def execute(
        cls, *, mask: object, threshold: float = 0.0, empty: str = "zero_region"
    ) -> Mapping[str, object]:
        array = _mask_array(mask)
        bounds, area = _mask_bounds(array, threshold)
        if area == 0 and empty == "error":
            raise ValueError("mask does not contain any selected pixels")
        if empty not in ("zero_region", "error"):
            raise ValueError(f"unknown empty mask policy: {empty}")
        return cls.outputs(
            mask=copy_media_semantics(mask, array),
            width=int(array.shape[2]),
            height=int(array.shape[1]),
            count=int(array.shape[0]),
            area=area,
            bounds=bounds,
        )


class MaskPolarity(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mask.polarity",
            display_name="Convert Mask Polarity",
            category="mask/convert",
            inputs=(
                InputSpec("mask", MASK),
                _combo("mask_polarity", ("coverage", "transparency"), "transparency"),
            ),
            outputs=(OutputSpec("mask", MASK, preview=True),),
            description="Invert mask values and declare the output polarity; retain its semantic.",
        )

    @classmethod
    def execute(cls, *, mask: object, mask_polarity: str = "transparency") -> Mapping[str, object]:
        if mask_polarity not in ("coverage", "transparency"):
            raise ValueError(f"unknown mask polarity: {mask_polarity}")
        return cls.outputs(
            mask=annotate_mask(
                np.ascontiguousarray(1.0 - _mask_array(mask)),
                polarity=mask_polarity,
                semantic=str(media_semantics(mask).get("semantic", "selection")),
            )
        )


MASK_NODES: tuple[type[Node], ...] = (
    MakeMask,
    TextMask,
    MaskMorphology,
    MaskCombine,
    ImageToMask,
    MaskToImage,
    MaskInfo,
    MaskPolarity,
)


__all__ = [
    "MASK_NODES",
    "ImageToMask",
    "MakeMask",
    "MaskCombine",
    "MaskInfo",
    "MaskMorphology",
    "MaskPolarity",
    "MaskToImage",
    "TextMask",
]
