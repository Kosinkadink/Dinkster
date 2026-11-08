"""Array helpers for image comparison and isolated GLSL rendering."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from dinkster_api.v1 import IMAGE_WORKING_SET_LIMIT_BYTES
from PIL import Image

MAX_IMAGE_BYTES = IMAGE_WORKING_SET_LIMIT_BYTES

MAX_DIMENSION = 16_384

Interpolation = Literal["nearest", "nearest-exact", "bilinear", "bicubic", "lanczos", "area"]

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "nearest-exact": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}


def check_output_size(shape: tuple[int, ...]) -> None:
    if any(size < 1 for size in shape):
        raise ValueError(f"output dimensions must be positive, got {shape}")
    if any(size > MAX_DIMENSION for size in shape[-3:-1]):
        raise ValueError(f"output dimensions exceed {MAX_DIMENSION}: {shape}")
    if math.prod(shape) * np.dtype(np.float32).itemsize > MAX_IMAGE_BYTES:
        raise ValueError(f"output exceeds the {MAX_IMAGE_BYTES}-byte image operation limit")


def image_array(image: object, *, subject: str = "image") -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[3] not in (1, 3, 4):
        raise ValueError(
            f"{subject} must have BHWC shape with 1, 3, or 4 channels, got {array.shape}"
        )
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{subject} dimensions must be non-zero, got {array.shape}")
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
