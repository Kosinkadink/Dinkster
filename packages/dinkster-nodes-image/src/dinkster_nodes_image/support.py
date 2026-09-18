"""Shared implementation helpers for first-party image and mask nodes."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Literal, ParamSpec, TypeVar, cast

import numpy as np
from dinkster_api.v1 import (
    CORE_COMBO,
    ComboWidget,
    InputSpec,
    NumberWidget,
    SlotValue,
    TypeExpr,
    image_math,
)
from PIL import Image

parse_color = image_math.parse_color

MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_DIMENSION = 16_384

COMBO = TypeExpr.concrete(CORE_COMBO)
Interpolation = Literal["nearest", "nearest-exact", "bilinear", "bicubic", "lanczos", "area"]
INTERPOLATION_OPTIONS = ("nearest", "nearest-exact", "bilinear", "bicubic", "lanczos", "area")
RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "nearest-exact": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}

P = ParamSpec("P")
R = TypeVar("R")


def normalize_mask_polarity(value: str) -> str:
    """Accept legacy values delivered through migrated connected controls."""
    return {
        "opacity": "coverage",
        "mask_is_opacity": "coverage",
        "mask_is_transparency": "transparency",
    }.get(value, value)


def combo_input(
    id: str,
    options: tuple[str, ...],
    default: str,
    *,
    advanced: bool = False,
    display_name: str = "",
    hidden: bool = False,
) -> InputSpec:
    return InputSpec(
        id,
        COMBO,
        required=False,
        default=default,
        widget=ComboWidget(options=options),
        advanced=advanced,
        display_name=display_name,
        hidden=hidden,
    )


def materialized_inputs(function: Callable[P, R]) -> Callable[P, R]:
    """Let an explicit execute signature consume materialized dynamic paths."""

    @wraps(function)
    def wrapped(*args: object, **inputs: object) -> R:
        projected: dict[str, object] = {}

        def add(input_id: str, value: object) -> None:
            if input_id in projected:
                raise ValueError(f"dynamic inputs project duplicate id {input_id!r}")
            projected[input_id] = value

        for path, value in inputs.items():
            input_id = path.rsplit(".", 1)[-1]
            if isinstance(value, SlotValue):
                add(input_id, value.value)
                for option, option_value in value.options.items():
                    add(option, option_value)
            else:
                add(input_id, value)
        return cast("Callable[..., R]", function)(*args, **projected)

    return cast("Callable[P, R]", wrapped)


def number_input(
    id: str,
    type_expr: TypeExpr,
    default: int | float,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    advanced: bool = False,
    hidden: bool = False,
) -> InputSpec:
    return InputSpec(
        id,
        type_expr,
        required=False,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
        advanced=advanced,
        hidden=hidden,
    )


def check_output_size(shape: tuple[int, ...]) -> None:
    if any(size < 1 for size in shape):
        raise ValueError(f"output dimensions must be positive, got {shape}")
    if any(size > MAX_DIMENSION for size in shape[-3:-1]):
        raise ValueError(f"output dimensions exceed {MAX_DIMENSION}: {shape}")
    if math.prod(shape) * np.dtype(np.float32).itemsize > MAX_IMAGE_BYTES:
        raise ValueError(f"output exceeds the {MAX_IMAGE_BYTES}-byte image operation limit")


def validate_canvas(width: int, height: int, batch_size: int, channels: int = 1) -> None:
    if not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION:
        raise ValueError(f"width and height must be between 1 and {MAX_DIMENSION}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    check_output_size((batch_size, height, width, channels))


def indexed_family_values(
    values: Mapping[str, object],
    *,
    count: int,
    member_prefix: str,
    subject: str,
) -> list[object | None]:
    """Order an indexed input family and preserve missing members below count."""

    if count == 0:
        if not values:
            raise ValueError(f"{subject} must not be empty")
        return list(values.values())
    if count < 1:
        raise ValueError(f"{subject} count must be at least 1")

    indexed: dict[int, object] = {}
    for suffix, value in values.items():
        normalized = suffix.removeprefix(member_prefix)
        if not normalized.isdigit() or int(normalized) < 1:
            raise ValueError(f"{subject} member suffix must be a positive integer, got {suffix!r}")
        index = int(normalized)
        if index <= count:
            if index in indexed:
                raise ValueError(f"{subject} has duplicate member index {index}")
            indexed[index] = value
    if 1 not in indexed:
        raise ValueError(f"{subject} requires member 1")
    return [indexed.get(index) for index in range(1, count + 1)]


def image_array(image: object, *, subject: str = "image") -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[3] not in (1, 3, 4):
        raise ValueError(
            f"{subject} must have BHWC shape with 1, 3, or 4 channels, got {array.shape}"
        )
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{subject} dimensions must be non-zero, got {array.shape}")
    return np.ascontiguousarray(array)


def mask_array(mask: object, *, subject: str = "mask") -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{subject} must have non-empty BHW shape, got {array.shape}")
    return np.ascontiguousarray(array)


def _linear_resize_axis(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    source_size = int(array.shape[axis])
    coordinates = (np.arange(size, dtype=np.float64) + 0.5) * source_size / size - 0.5
    lower = np.floor(coordinates).astype(np.int64)
    weights = (coordinates - lower).astype(np.float32)
    upper = np.clip(lower + 1, 0, source_size - 1)
    lower = np.clip(lower, 0, source_size - 1)
    shape = [1] * array.ndim
    shape[axis] = size
    weight_array = weights.reshape(shape)
    low_values = np.take(array, lower, axis=axis)
    high_values = np.take(array, upper, axis=axis)
    return low_values + (high_values - low_values) * weight_array


def _cubic_weight(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    coefficient = -0.75
    near = ((coefficient + 2.0) * absolute - (coefficient + 3.0)) * absolute * absolute + 1.0
    far = (coefficient * absolute - 5.0 * coefficient) * absolute + 8.0 * coefficient
    far = far * absolute - 4.0 * coefficient
    return np.where(absolute <= 1.0, near, np.where(absolute < 2.0, far, 0.0)).astype(np.float32)


def _cubic_resize_axis(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    source_size = int(array.shape[axis])
    coordinates = (np.arange(size, dtype=np.float64) + 0.5) * source_size / size - 0.5
    base = np.floor(coordinates).astype(np.int64)
    output = np.zeros((*array.shape[:axis], size, *array.shape[axis + 1 :]), dtype=np.float32)
    shape = [1] * array.ndim
    shape[axis] = size
    for offset in (-1, 0, 1, 2):
        indices = np.clip(base + offset, 0, source_size - 1)
        weights = _cubic_weight(coordinates - (base + offset)).reshape(shape)
        output += np.take(array, indices, axis=axis) * weights
    return output


def _adaptive_average_axis(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    source_size = int(array.shape[axis])
    output_shape = list(array.shape)
    output_shape[axis] = size
    output = np.empty(output_shape, dtype=np.float32)
    source_slices: list[slice] = [slice(None)] * array.ndim
    output_slices: list[slice | int] = [slice(None)] * array.ndim
    for index in range(size):
        start = index * source_size // size
        end = math.ceil((index + 1) * source_size / size)
        source_slices[axis] = slice(start, end)
        output_slices[axis] = index
        output[tuple(output_slices)] = array[tuple(source_slices)].mean(
            axis=axis,
            dtype=np.float32,
        )
    return output


def _resize_separable(
    channels: np.ndarray,
    width: int,
    height: int,
    resize_axis: Callable[[np.ndarray, int, int], np.ndarray],
) -> np.ndarray:
    source_height, source_width = int(channels.shape[1]), int(channels.shape[2])
    if height * source_width <= source_height * width:
        return resize_axis(resize_axis(channels, height, 1), width, 2)
    return resize_axis(resize_axis(channels, width, 2), height, 1)


def resize_array(
    array: np.ndarray,
    width: int,
    height: int,
    interpolation: Interpolation,
) -> np.ndarray:
    if interpolation not in RESAMPLING:
        raise ValueError(f"unknown interpolation mode: {interpolation}")
    is_mask = array.ndim == 3
    channels = array[:, :, :, None] if is_mask else array
    batch, source_height, source_width, count = channels.shape
    check_output_size((batch, height, width, count))
    if interpolation in ("nearest", "nearest-exact"):
        offset = 0.5 if interpolation == "nearest-exact" else 0.0
        rows = np.minimum(
            np.floor((np.arange(height) + offset) * source_height / height).astype(np.int64),
            source_height - 1,
        )
        columns = np.minimum(
            np.floor((np.arange(width) + offset) * source_width / width).astype(np.int64),
            source_width - 1,
        )
        output = channels[:, rows[:, None], columns[None, :], :]
        return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)
    if interpolation == "bilinear":
        output = _resize_separable(channels, width, height, _linear_resize_axis)
        return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)
    if interpolation == "area":
        output = _resize_separable(channels, width, height, _adaptive_average_axis)
        return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)
    if interpolation == "bicubic":
        output = _resize_separable(channels, width, height, _cubic_resize_axis)
        return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)
    output = np.empty((batch, height, width, count), dtype=np.float32)
    if interpolation == "lanczos":
        for batch_index in range(batch):
            pixels = channels[batch_index]
            pixels = pixels[:, :, 0] if count == 1 else pixels
            raster = np.clip(255.0 * pixels, 0, 255).astype(np.uint8)
            resized = np.asarray(
                Image.fromarray(raster).resize(
                    (width, height),
                    resample=Image.Resampling.LANCZOS,
                ),
                dtype=np.float32,
            )
            output[batch_index] = resized[:, :, None] / 255.0 if count == 1 else resized / 255.0
        return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)
    resampling = RESAMPLING[interpolation]
    for batch_index in range(batch):
        for channel_index in range(count):
            plane = Image.fromarray(channels[batch_index, :, :, channel_index], mode="F")
            output[batch_index, :, :, channel_index] = np.asarray(
                plane.resize((width, height), resample=resampling), dtype=np.float32
            )
    return np.ascontiguousarray(output[:, :, :, 0] if is_mask else output)


def resize_to_fill(
    array: np.ndarray,
    width: int,
    height: int,
    interpolation: Interpolation,
) -> np.ndarray:
    source_height, source_width = int(array.shape[1]), int(array.shape[2])
    source_aspect = source_width / source_height
    target_aspect = width / height
    if source_aspect > target_aspect:
        crop_width = max(1, round(source_height * target_aspect))
        left = (source_width - crop_width) // 2
        array = array[:, :, left : left + crop_width, ...]
    elif source_aspect < target_aspect:
        crop_height = max(1, round(source_width / target_aspect))
        top = (source_height - crop_height) // 2
        array = array[:, top : top + crop_height, :, ...]
    return resize_array(array, width, height, interpolation)


def normalized_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    return np.broadcast_to(x[None, :], (height, width)), np.broadcast_to(
        y[:, None], (height, width)
    )


def linear_ramp(width: int, height: int, angle: float) -> np.ndarray:
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    x, y = normalized_grid(width, height)
    radians = math.radians(angle)
    dx, dy = math.cos(radians), math.sin(radians)
    projection = (x - 0.5) * dx + (y - 0.5) * dy
    corners = np.array(
        [
            (-0.5 * dx) + (-0.5 * dy),
            (0.5 * dx) + (-0.5 * dy),
            (-0.5 * dx) + (0.5 * dy),
            (0.5 * dx) + (0.5 * dy),
        ],
        dtype=np.float32,
    )
    low, high = float(corners.min()), float(corners.max())
    if high == low:
        return np.zeros((height, width), dtype=np.float32)
    return np.asarray((projection - low) / (high - low), dtype=np.float32)


def radial_ramp(
    width: int, height: int, center_x: float, center_y: float, radius: float
) -> np.ndarray:
    if not all(math.isfinite(value) for value in (center_x, center_y, radius)) or radius <= 0:
        raise ValueError("gradient center must be finite and radius must be positive")
    x, y = normalized_grid(width, height)
    distance = np.sqrt(np.square(x - center_x) + np.square(y - center_y))
    return np.asarray(np.clip(distance / radius, 0.0, 1.0), dtype=np.float32)


def color_from_int(value: int) -> np.ndarray:
    if type(value) is not int or not 0 <= value <= 0xFFFFFF:
        raise ValueError("integer color must be between 0 and 0xFFFFFF")
    return np.asarray(
        [
            ((value >> 16) & 0xFF) / 255.0,
            ((value >> 8) & 0xFF) / 255.0,
            (value & 0xFF) / 255.0,
            1.0,
        ],
        dtype=np.float32,
    )


__all__ = [
    "INTERPOLATION_OPTIONS",
    "MAX_DIMENSION",
    "Interpolation",
    "RESAMPLING",
    "check_output_size",
    "color_from_int",
    "combo_input",
    "image_array",
    "indexed_family_values",
    "linear_ramp",
    "mask_array",
    "materialized_inputs",
    "number_input",
    "parse_color",
    "radial_ramp",
    "resize_array",
    "resize_to_fill",
    "validate_canvas",
]
