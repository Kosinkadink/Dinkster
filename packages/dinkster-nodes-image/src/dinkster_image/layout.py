"""Deterministic image grids and stitches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    ConditionalWidgetGroup,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
)

from .geometry import BOOLEAN, IMAGE, INT
from .support import (
    Interpolation,
    check_output_size,
    combo_input,
    image_array,
    number_input,
    resize_array,
)


class ImageGridCompose(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.grid.compose",
            display_name="Compose Image Grid",
            category="image/layout",
            inputs=(number_input("columns", INT, 2, minimum=1, maximum=100, step=1),),
            input_families=(
                InputFamilySpec(
                    "images", IMAGE, min_members=1, max_members=100, member_prefix="image_"
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("ImageGridComposite2x2", "ImageGridComposite3x3", "image grid"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Mapping[str, object],
        columns: int = 2,
    ) -> Mapping[str, object]:
        if columns < 1:
            raise ValueError(f"columns must be positive, got {columns}")
        arrays = [
            image_array(value, subject=f"image {index}")
            for index, value in enumerate(images.values())
        ]
        if not arrays:
            raise ValueError("images must not be empty")
        if len(arrays) % columns:
            raise ValueError("image count must be divisible by columns")
        if any(
            array.shape[0] != arrays[0].shape[0] or array.shape[3] != arrays[0].shape[3]
            for array in arrays[1:]
        ):
            raise ValueError("all grid images must have identical batch and channel counts")
        rows: list[list[np.ndarray]] = []
        for index in range(0, len(arrays), columns):
            row = arrays[index : index + columns]
            if any(array.shape[1] != row[0].shape[1] for array in row[1:]):
                raise ValueError("images in each grid row must have identical heights")
            rows.append(row)
        row_widths = [sum(int(array.shape[2]) for array in row) for row in rows]
        if any(width != row_widths[0] for width in row_widths[1:]):
            raise ValueError("grid rows must have identical widths")
        check_output_size(
            (
                len(arrays[0]),
                sum(int(row[0].shape[1]) for row in rows),
                row_widths[0],
                int(arrays[0].shape[3]),
            )
        )
        row_images = [np.concatenate(row, axis=2) for row in rows]
        return cls.outputs(image=np.ascontiguousarray(np.concatenate(row_images, axis=1)))


class ImageGridDecompose(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.grid.decompose",
            display_name="Decompose Image Grid",
            category="image/layout",
            inputs=(
                InputSpec("image", IMAGE),
                number_input("columns", INT, 3, minimum=1, maximum=100, step=1),
                number_input("rows", INT, 0, minimum=0, maximum=100, step=1),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("ImageGridtoBatch", "grid to batch"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        columns: int = 3,
        rows: int = 0,
    ) -> Mapping[str, object]:
        array = image_array(image)
        if columns < 1 or rows < 0:
            raise ValueError("columns must be positive and rows must be non-negative")
        batch, height, width, channels = (int(value) for value in array.shape)
        cell_width = width // columns
        if cell_width < 1:
            raise ValueError("columns exceed the image width")
        if rows == 0:
            cell_height = height // columns
            if cell_height < 1:
                raise ValueError("automatic grid rows exceed the image height")
            rows = height // cell_height
        else:
            cell_height = height // rows
        if cell_height < 1:
            raise ValueError("rows exceed the image height")
        cropped = array[:, : rows * cell_height, : columns * cell_width, :]
        output = cropped.reshape(batch, rows, cell_height, columns, cell_width, channels)
        output = output.transpose(0, 1, 3, 2, 4, 5)
        return cls.outputs(
            image=np.ascontiguousarray(
                output.reshape(batch * rows * columns, cell_height, cell_width, channels)
            )
        )


_STITCH_COLORS: dict[str, float | tuple[float, float, float]] = {
    "white": 1.0,
    "black": 0.0,
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def _extend_last(array: np.ndarray, count: int) -> np.ndarray:
    if len(array) == count:
        return array
    return np.concatenate((array, np.repeat(array[-1:], count - len(array), axis=0)), axis=0)


def _pad_axis(array: np.ndarray, axis: int, size: int, value: float) -> np.ndarray:
    difference = size - int(array.shape[axis])
    if difference == 0:
        return array
    before = difference // 2
    after = difference - before
    pads = [(0, 0)] * array.ndim
    pads[axis] = (before, after)
    return np.pad(array, pads, mode="constant", constant_values=value)


def _pad_channels(array: np.ndarray, channels: int) -> np.ndarray:
    difference = channels - int(array.shape[3])
    if difference == 0:
        return array
    alpha = np.ones((*array.shape[:3], difference), dtype=np.float32)
    return np.concatenate((array, alpha), axis=3)


def _stitch(
    first: np.ndarray,
    second: np.ndarray,
    *,
    direction: str,
    match_image_size: bool,
    spacing_width: int,
    spacing_color: str,
) -> np.ndarray:
    if direction not in ("right", "down", "left", "up"):
        raise ValueError(f"unknown stitch direction: {direction}")
    if spacing_width < 0:
        raise ValueError(f"spacing_width must be non-negative, got {spacing_width}")
    if spacing_color not in _STITCH_COLORS:
        raise ValueError(f"unknown spacing color: {spacing_color}")
    batch = max(len(first), len(second))
    horizontal = direction in ("left", "right")
    color = _STITCH_COLORS[spacing_color]
    channels = max(int(first.shape[3]), int(second.shape[3]))
    spacing_width += spacing_width % 2
    target_height, target_width = int(second.shape[1]), int(second.shape[2])
    if match_image_size:
        if horizontal:
            target_height = int(first.shape[1])
            target_width = max(1, int(target_height * int(second.shape[2]) / int(second.shape[1])))
            output_height = target_height
            output_width = int(first.shape[2]) + target_width + spacing_width
        else:
            target_width = int(first.shape[2])
            target_height = max(1, int(target_width * int(second.shape[1]) / int(second.shape[2])))
            output_height = int(first.shape[1]) + target_height + spacing_width
            output_width = target_width
    elif horizontal:
        output_height = max(int(first.shape[1]), int(second.shape[1]))
        output_width = int(first.shape[2]) + int(second.shape[2]) + spacing_width
    else:
        output_height = int(first.shape[1]) + int(second.shape[1]) + spacing_width
        output_width = max(int(first.shape[2]), int(second.shape[2]))
    check_output_size((batch, output_height, output_width, channels))
    first, second = _extend_last(first, batch), _extend_last(second, batch)
    if match_image_size:
        second = resize_array(
            second,
            target_width,
            target_height,
            cast("Interpolation", "lanczos"),
        )
    else:
        pad_value = color if isinstance(color, float) else 0.0
        axis = 1 if horizontal else 2
        target = max(int(first.shape[axis]), int(second.shape[axis]))
        first = _pad_axis(first, axis, target, pad_value)
        second = _pad_axis(second, axis, target, pad_value)
    first, second = _pad_channels(first, channels), _pad_channels(second, channels)
    concat_axis = 2 if horizontal else 1
    ordered = [second, first] if direction in ("left", "up") else [first, second]
    if spacing_width:
        spacing_shape = list(first.shape)
        spacing_shape[concat_axis] = spacing_width
        spacing = np.zeros(tuple(spacing_shape), dtype=np.float32)
        if isinstance(color, tuple):
            for index, channel in enumerate(color[:channels]):
                spacing[..., index] = channel
        else:
            spacing[..., : min(3, channels)] = color
        if channels == 4:
            spacing[..., 3] = 1.0
        ordered.insert(1, spacing)
    return np.ascontiguousarray(np.concatenate(ordered, axis=concat_axis))


class ImageStitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.stitch",
            display_name="Stitch Images",
            category="image/layout",
            inputs=(
                InputSpec("first", IMAGE),
                combo_input("direction", ("right", "down", "left", "up"), "right"),
                InputSpec("match_image_size", BOOLEAN, required=False, default=True),
                number_input("spacing_width", INT, 0, minimum=0, maximum=1024, step=2),
                combo_input(
                    "spacing_color",
                    ("white", "black", "red", "green", "blue"),
                    "white",
                ),
                InputSpec("second", IMAGE, required=False, default=None),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            widget_groups=(
                ConditionalWidgetGroup(
                    "spacing_width",
                    tuple(range(1, 1025)),
                    ("spacing_color",),
                ),
            ),
            search_terms=("ImageStitch", "join images", "concatenate images", "side by side"),
        )

    @classmethod
    def execute(
        cls,
        *,
        first: object,
        direction: str = "right",
        match_image_size: bool = True,
        spacing_width: int = 0,
        spacing_color: str = "white",
        second: object | None = None,
    ) -> Mapping[str, object]:
        first_array = image_array(first, subject="first")
        if second is None:
            return cls.outputs(image=first_array)
        return cls.outputs(
            image=_stitch(
                first_array,
                image_array(second, subject="second"),
                direction=direction,
                match_image_size=match_image_size,
                spacing_width=spacing_width,
                spacing_color=spacing_color,
            )
        )


LAYOUT_NODES: tuple[type[Node], ...] = (ImageGridCompose, ImageGridDecompose, ImageStitch)


__all__ = [
    "LAYOUT_NODES",
    "ImageGridCompose",
    "ImageGridDecompose",
    "ImageStitch",
]
