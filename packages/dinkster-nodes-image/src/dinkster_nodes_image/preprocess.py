"""Deterministic control-hint preprocessing over BHWC image arrays."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import cv2
import numpy as np
from dinkster_api.v1 import (
    CORE_COMBO,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

from .geometry import BOOLEAN, FLOAT, IMAGE, INT, MASK
from .support import check_output_size as _check_output_size
from .support import combo_input as _combo
from .support import image_array as _image_array
from .support import mask_array as _mask_array
from .support import number_input as _number
from .support import resize_array as _resize_array

MAX_RESOLUTION = 16_384
MAX_SEED = 2**53 - 1
MODEL_EDGE_PROVIDER_CHOICE = "dinkster.preprocess.model_edges.providers"
MODEL_EDGE_PROVIDER_CHOICES = (
    MODEL_EDGE_PROVIDER_CHOICE,
    "dinkster.preprocess.lineart_realistic.providers",
    "dinkster.preprocess.lineart_anime.providers",
    "dinkster.preprocess.lineart_manga.providers",
    "dinkster.preprocess.anyline.providers",
    "dinkster.preprocess.teed.providers",
    "dinkster.preprocess.mlsd.providers",
)
MODEL_DEPTH_PROVIDER_CHOICE = "dinkster.preprocess.model_depth.providers"

_THIN_KERNELS_RAW = (
    np.array(((-1, -1, -1), (0, 1, 0), (1, 1, 1)), dtype=np.int32),
    np.array(((0, -1, -1), (1, 1, -1), (0, 1, 0)), dtype=np.int32),
)
_THIN_KERNELS = tuple(
    np.rot90(kernel, k=rotation, axes=(0, 1))
    for rotation in range(4)
    for kernel in _THIN_KERNELS_RAW
)
_PRUNE_KERNELS_RAW = (
    np.array(((-1, -1, -1), (-1, 1, -1), (0, 0, -1)), dtype=np.int32),
    np.array(((-1, -1, -1), (-1, 1, -1), (-1, 0, 0)), dtype=np.int32),
)
_PRUNE_KERNELS = tuple(
    np.rot90(kernel, k=rotation, axes=(0, 1))
    for rotation in range(4)
    for kernel in _PRUNE_KERNELS_RAW
)


def _validate_resolution(resolution: int) -> None:
    if resolution != 0 and not 64 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be 0 or between 64 and {MAX_RESOLUTION}")


def _hwc3(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError("image frame must use uint8 pixels")
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.shape[2] == 1:
        return np.ascontiguousarray(np.concatenate((array, array, array), axis=2))
    color = array[:, :, :3].astype(np.float32)
    alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    return np.ascontiguousarray(
        np.clip(color * alpha + 255.0 * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
    )


def _uint8_frames(image: object, *, ignore_alpha: bool = False) -> list[np.ndarray]:
    array = _image_array(image)
    if ignore_alpha and array.shape[3] == 4:
        array = array[..., :3]
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("image preprocessors require finite pixel values")
    source = np.clip(array, 0.0, 1.0) if minimum < 0.0 or maximum > 1.0 else array
    raster = np.empty(array.shape, dtype=np.uint8)
    np.multiply(source, 255.0, out=raster, casting="unsafe")
    return [_hwc3(frame) for frame in raster]


def _image_output(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("image preprocessor produced no frames")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("image preprocessor produced inconsistent batch dimensions")
    _check_output_size((len(frames), *shape))
    output = np.asarray(frames, dtype=np.float32)
    output /= 255.0
    return np.ascontiguousarray(output)


def _resize_with_pad(
    frame: np.ndarray,
    resolution: int,
    upscale_method: int,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    _validate_resolution(resolution)
    if resolution == 0:
        return frame, lambda result: np.ascontiguousarray(result)
    height, width = frame.shape[:2]
    scale = float(resolution) / float(min(height, width))
    target_height = int(np.round(float(height) * scale))
    target_width = int(np.round(float(width) * scale))
    pad_height = math.ceil(target_height / 64) * 64 - target_height
    pad_width = math.ceil(target_width / 64) * 64 - target_width
    _check_output_size((1, target_height + pad_height, target_width + pad_width, 3))
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=upscale_method if scale > 1.0 else cv2.INTER_AREA,
    )
    padded = np.pad(resized, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")

    def remove_pad(result: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(result[:target_height, :target_width, ...])

    return np.ascontiguousarray(padded), remove_pad


def _map_frames(
    image: object,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    ignore_alpha: bool = False,
) -> np.ndarray:
    return _image_output(
        [_hwc3(transform(frame)) for frame in _uint8_frames(image, ignore_alpha=ignore_alpha)]
    )


def _limited_unique_color_count(frame: np.ndarray, limit: int = 200) -> int:
    colors: set[tuple[int, ...]] = set()
    flat = frame.reshape(-1, frame.shape[2])
    for start in range(0, len(flat), 65_536):
        for color in np.unique(flat[start : start + 65_536], axis=0):
            colors.add(tuple(int(channel) for channel in color))
            if len(colors) >= limit:
                return limit
    return len(colors)


def _thin_once(image: np.ndarray, kernels: tuple[np.ndarray, ...]) -> tuple[np.ndarray, bool]:
    changed = False
    for kernel in kernels:
        matches = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel) > 127
        if np.any(matches):
            image[matches] = 0
            changed = True
    return image, not changed


def _thin_binary_edges(image: np.ndarray, *, prune: bool) -> np.ndarray:
    output = image
    for _ in range(32):
        output, done = _thin_once(output, _THIN_KERNELS)
        if done:
            break
    if prune:
        output, _ = _thin_once(output, _PRUNE_KERNELS)
    return output


def _suppress_non_maximum(image: np.ndarray) -> np.ndarray:
    filters = (
        np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)), dtype=np.uint8),
        np.array(((0, 1, 0), (0, 1, 0), (0, 1, 0)), dtype=np.uint8),
        np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.uint8),
        np.array(((0, 0, 1), (0, 1, 0), (1, 0, 0)), dtype=np.uint8),
    )
    output = np.zeros_like(image)
    for kernel in filters:
        np.putmask(output, cv2.dilate(image, kernel=kernel) == image, image)
    return output


def _high_quality_hint_resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame)
    _check_output_size((1, height, width, 3))
    source_area = int(frame.shape[0]) * int(frame.shape[1])
    target_area = height * width
    unique_colors = _limited_unique_color_count(frame)
    binary = unique_colors == 2 and int(np.min(frame)) < 16 and int(np.max(frame)) > 240
    one_pixel_edge = False
    if binary:
        opened = cv2.dilate(
            cv2.erode(frame, np.ones((3, 3), dtype=np.uint8), iterations=1),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        one_pixel_edge = np.count_nonzero(opened < frame) * 2 > np.count_nonzero(frame > 127)
    if 2 < unique_colors < 200:
        interpolation = cv2.INTER_NEAREST
    elif target_area < source_area:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    output = cv2.resize(frame, (width, height), interpolation=interpolation)
    if not binary:
        return np.ascontiguousarray(output)
    gray = np.clip(np.mean(output.astype(np.float32), axis=2), 0.0, 255.0).astype(np.uint8)
    if one_pixel_edge:
        gray = _suppress_non_maximum(gray)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if one_pixel_edge:
        thresholded = _thin_binary_edges(thresholded, prune=target_area > source_area)
    return np.ascontiguousarray(np.repeat(thresholded[:, :, None], 3, axis=2))


def _resize_hint_frame(frame: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    if mode == "stretch":
        return _high_quality_hint_resize(frame, width, height)
    source_height, source_width = int(frame.shape[0]), int(frame.shape[1])
    if mode == "fit":
        scale = min(height / source_height, width / source_width)
    elif mode == "fill":
        scale = max(height / source_height, width / source_width)
    else:
        raise ValueError(f"unknown hint resize mode: {mode}")
    resized_width = int(np.round(source_width * scale))
    resized_height = int(np.round(source_height * scale))
    _check_output_size((1, resized_height, resized_width, 3))
    resized = _high_quality_hint_resize(frame, resized_width, resized_height)
    if mode == "fill":
        top = max(0, (resized_height - height) // 2)
        left = max(0, (resized_width - width) // 2)
        return np.ascontiguousarray(resized[top : top + height, left : left + width])
    border = np.concatenate(
        (frame[0], frame[-1], frame[:, 0], frame[:, -1]),
        axis=0,
    )
    background = np.tile(np.median(border, axis=0).astype(np.uint8), (height, width, 1))
    top = max(0, (height - resized_height) // 2)
    left = max(0, (width - resized_width) // 2)
    background[top : top + resized_height, left : left + resized_width] = resized
    return np.ascontiguousarray(background)


def _canny(frame: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    return cv2.Canny(frame, low_threshold, high_threshold)


def _pyramid_canny(frame: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if min(height, width) < 5:
        raise ValueError("pyramid Canny requires image dimensions of at least 5 pixels")
    accumulated: np.ndarray | None = None
    for scale in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        scaled = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        edges = np.stack(
            [
                cv2.Canny(scaled[:, :, channel], low_threshold, high_threshold).astype(np.float32)
                / 255.0
                for channel in range(3)
            ],
            axis=2,
        )
        if accumulated is None:
            accumulated = edges
        else:
            accumulated = cv2.resize(
                accumulated,
                (edges.shape[1], edges.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            accumulated *= 0.75
            accumulated += edges * 0.25
    assert accumulated is not None
    combined = np.sum(accumulated, axis=2, dtype=np.float32)
    minimum = float(np.percentile(combined, 1))
    maximum = float(np.percentile(combined, 99))
    if maximum <= minimum:
        return np.zeros(combined.shape, dtype=np.uint8)
    combined -= minimum
    combined /= maximum - minimum
    return np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8)


class EdgePreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.edges",
            display_name="Preprocess Edges",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _combo("method", ("canny", "pyramid_canny"), "canny"),
                _number("low_threshold", INT, 100, minimum=0, maximum=255, step=1),
                _number("high_threshold", INT, 200, minimum=0, maximum=255, step=1),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("canny", "pyracanny", "edge map", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        method: str = "canny",
        low_threshold: int = 100,
        high_threshold: int = 200,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if not 0 <= low_threshold <= 255 or not 0 <= high_threshold <= 255:
            raise ValueError("Canny thresholds must be between 0 and 255")
        if method not in ("canny", "pyramid_canny"):
            raise ValueError(f"unknown edge preprocessor: {method}")

        def transform(frame: np.ndarray) -> np.ndarray:
            resized, remove_pad = _resize_with_pad(frame, resolution, cv2.INTER_CUBIC)
            result = (
                _canny(resized, low_threshold, high_threshold)
                if method == "canny"
                else _pyramid_canny(resized, low_threshold, high_threshold)
            )
            return remove_pad(result)

        return cls.outputs(image=_map_frames(image, transform, ignore_alpha=True))


class LineartPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart",
            display_name="Preprocess Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _number("gaussian_sigma", FLOAT, 6.0, minimum=0.01, maximum=100.0, step=0.01),
                _number("intensity_threshold", INT, 8, minimum=0, maximum=16, step=1),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("lineart", "line drawing", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        gaussian_sigma: float = 6.0,
        intensity_threshold: int = 8,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if not math.isfinite(gaussian_sigma) or gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be finite and positive")
        if not 0 <= intensity_threshold <= 16:
            raise ValueError("intensity_threshold must be between 0 and 16")

        def transform(frame: np.ndarray) -> np.ndarray:
            resized, remove_pad = _resize_with_pad(frame, resolution, cv2.INTER_CUBIC)
            source = resized.astype(np.float32)
            blurred = cv2.GaussianBlur(source, (0, 0), gaussian_sigma)
            intensity = np.clip(np.min(blurred - source, axis=2), 0.0, 255.0)
            selected = intensity[intensity > intensity_threshold]
            divisor = max(16.0, float(np.median(selected))) if selected.size else 16.0
            result = np.clip(intensity / divisor * 127.0, 0.0, 255.0).astype(np.uint8)
            return remove_pad(result)

        return cls.outputs(image=_map_frames(image, transform))


class ScribblePreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.scribble",
            display_name="Preprocess Scribble",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _combo("method", ("threshold", "xdog"), "threshold"),
                _number("threshold", INT, 32, minimum=1, maximum=64, step=1),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("scribble", "xdog", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        method: str = "threshold",
        threshold: int = 32,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if method not in ("threshold", "xdog"):
            raise ValueError(f"unknown scribble preprocessor: {method}")
        if not 1 <= threshold <= 64:
            raise ValueError("scribble threshold must be between 1 and 64")

        def transform(frame: np.ndarray) -> np.ndarray:
            upscale = cv2.INTER_AREA if method == "threshold" else cv2.INTER_CUBIC
            resized, remove_pad = _resize_with_pad(frame, resolution, upscale)
            if method == "threshold":
                result = np.full(resized.shape, 255, dtype=np.uint8)
                result[np.min(resized, axis=2) < 127] = 0
            else:
                source = resized.astype(np.float32)
                narrow = cv2.GaussianBlur(source, (0, 0), 0.5)
                wide = cv2.GaussianBlur(source, (0, 0), 5.0)
                dog = np.clip(255.0 - np.min(wide - narrow, axis=2), 0.0, 255.0).astype(np.uint8)
                result = np.zeros(resized.shape, dtype=np.uint8)
                result[2 * (255 - dog) > threshold] = 255
            return remove_pad(result)

        return cls.outputs(image=_map_frames(image, transform))


class BinaryPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.binary",
            display_name="Preprocess Binary Image",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _number("threshold", INT, 100, minimum=0, maximum=255, step=1),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("binary", "threshold", "otsu", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        threshold: int = 100,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if not 0 <= threshold <= 255:
            raise ValueError("binary threshold must be between 0 and 255")

        def transform(frame: np.ndarray) -> np.ndarray:
            resized, remove_pad = _resize_with_pad(frame, resolution, cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            mode = cv2.THRESH_BINARY_INV
            value = threshold
            if threshold in (0, 255):
                mode += cv2.THRESH_OTSU
                value = 0
            _, binary = cv2.threshold(gray, value, 255, mode)
            return remove_pad(255 - binary)

        return cls.outputs(image=_map_frames(image, transform))


def _palette_hint(frame: np.ndarray, resolution: int) -> np.ndarray:
    _validate_resolution(resolution)
    height, width = frame.shape[:2]
    size = min(height, width) if resolution == 0 else resolution
    if height < width:
        target_height = size
        target_width = int(round(width / height * size))
    else:
        target_width = size
        target_height = int(round(height / width * size))
    _check_output_size((1, target_height, target_width, 3))
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    reduced = cv2.resize(
        resized,
        (max(1, target_width // 64), max(1, target_height // 64)),
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.resize(
        reduced,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


class ColorHintPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.color_hint",
            display_name="Preprocess Color Hint",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _combo("method", ("palette", "luminance", "intensity"), "palette"),
                _number("gamma", FLOAT, 1.0, minimum=0.1, maximum=2.0, step=0.01),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("color map", "recolor", "luminance", "intensity", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        method: str = "palette",
        gamma: float = 1.0,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if method not in ("palette", "luminance", "intensity"):
            raise ValueError(f"unknown color hint preprocessor: {method}")
        if not math.isfinite(gamma) or not 0.1 <= gamma <= 2.0:
            raise ValueError("gamma must be between 0.1 and 2.0")

        def transform(frame: np.ndarray) -> np.ndarray:
            if method == "palette":
                return _palette_hint(frame, resolution)
            resized, remove_pad = _resize_with_pad(frame, resolution, cv2.INTER_CUBIC)
            conversion = cv2.COLOR_BGR2LAB if method == "luminance" else cv2.COLOR_BGR2HSV
            channel = cv2.cvtColor(resized, conversion)[:, :, 0 if method == "luminance" else 2]
            result = np.power(channel.astype(np.float32) / 255.0, gamma)
            return remove_pad(np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8))

        return cls.outputs(image=_map_frames(image, transform))


def _noise_disk(
    height: int,
    width: int,
    frequency: int,
    generator: np.random.Generator,
) -> np.ndarray:
    noise = generator.uniform(
        low=0.0,
        high=1.0,
        size=((height // frequency) + 2, (width // frequency) + 2, 1),
    )
    noise = cv2.resize(
        noise,
        (width + 2 * frequency, height + 2 * frequency),
        interpolation=cv2.INTER_CUBIC,
    )
    noise = noise[frequency : frequency + height, frequency : frequency + width]
    noise -= np.min(noise)
    maximum = float(np.max(noise))
    if maximum > 0.0:
        noise /= maximum
    return noise[:, :, None] if noise.ndim == 2 else noise


class ContentShufflePreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.content_shuffle",
            display_name="Preprocess Content Shuffle",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _number("seed", INT, 0, minimum=0, maximum=MAX_SEED, step=1),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=0,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("shuffle image", "content shuffle", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        seed: int = 0,
        resolution: int = 512,
    ) -> Mapping[str, object]:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}")

        def transform(frame: np.ndarray) -> np.ndarray:
            resized, remove_pad = _resize_with_pad(frame, resolution, cv2.INTER_CUBIC)
            height, width = resized.shape[:2]
            generator = np.random.Generator(np.random.PCG64(seed))
            x = _noise_disk(height, width, 256, generator) * float(width - 1)
            y = _noise_disk(height, width, 256, generator) * float(height - 1)
            flow = np.concatenate((x, y), axis=2).astype(np.float32)
            return remove_pad(cv2.remap(resized, flow[:, :, 0], flow[:, :, 1], cv2.INTER_LINEAR))

        return cls.outputs(image=_map_frames(image, transform))


def _blur_for_tile(frame: np.ndarray, strength: float) -> np.ndarray:
    kernel_size = int(strength)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=strength / 2.0)


def _fast_guided_filter(frame: np.ndarray, radius: int, epsilon: float, scale: float) -> np.ndarray:
    source = frame.astype(np.float32) / 255.0
    height, width = source.shape[:2]
    sub_width, sub_height = int(width / scale), int(height / scale)
    if sub_width < 1 or sub_height < 1:
        raise ValueError("guided-filter scale exceeds the image dimensions")
    guide = cv2.resize(source, (sub_width, sub_height), interpolation=cv2.INTER_NEAREST)
    window = 2 * int(radius / scale) + 1
    red, green, blue = (guide[:, :, index] for index in range(3))
    red_mean = cv2.blur(red, (window, window))
    green_mean = cv2.blur(green, (window, window))
    blue_mean = cv2.blur(blue, (window, window))
    rr = cv2.blur(red**2, (window, window)) - red_mean**2 + epsilon
    rg = cv2.blur(red * green, (window, window)) - red_mean * green_mean
    rb = cv2.blur(red * blue, (window, window)) - red_mean * blue_mean
    gg = cv2.blur(green**2, (window, window)) - green_mean**2 + epsilon
    gb = cv2.blur(green * blue, (window, window)) - green_mean * blue_mean
    bb = cv2.blur(blue**2, (window, window)) - blue_mean**2 + epsilon
    rr_inv = gg * bb - gb * gb
    rg_inv = gb * rb - rg * bb
    rb_inv = rg * gb - gg * rb
    gg_inv = rr * bb - rb * rb
    gb_inv = rb * rg - rr * gb
    bb_inv = rr * gg - rg * rg
    covariance = rr_inv * rr + rg_inv * rg + rb_inv * rb
    rr_inv /= covariance
    rg_inv /= covariance
    rb_inv /= covariance
    gg_inv /= covariance
    gb_inv /= covariance
    bb_inv /= covariance

    output = np.array(source)
    for channel in range(3):
        plane = guide[:, :, channel]
        plane_mean = cv2.blur(plane, (window, window))
        red_cov = cv2.blur(red * plane, (window, window)) - red_mean * plane_mean
        green_cov = cv2.blur(green * plane, (window, window)) - green_mean * plane_mean
        blue_cov = cv2.blur(blue * plane, (window, window)) - blue_mean * plane_mean
        a_red = rr_inv * red_cov + rg_inv * green_cov + rb_inv * blue_cov
        a_green = rg_inv * red_cov + gg_inv * green_cov + gb_inv * blue_cov
        a_blue = rb_inv * red_cov + gb_inv * green_cov + bb_inv * blue_cov
        offset = plane_mean - a_red * red_mean - a_green * green_mean - a_blue * blue_mean
        coefficients = [
            cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
            for value in (
                cv2.blur(a_red, (window, window)),
                cv2.blur(a_green, (window, window)),
                cv2.blur(a_blue, (window, window)),
                cv2.blur(offset, (window, window)),
            )
        ]
        output[:, :, channel] = (
            coefficients[0] * source[:, :, 0]
            + coefficients[1] * source[:, :, 1]
            + coefficients[2] * source[:, :, 2]
            + coefficients[3]
        )
    return np.clip((255.0 * output).astype(np.uint8), 0, 255)


class TileHintPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.tile_hint",
            display_name="Preprocess Tile Hint",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _combo("method", ("pyramid", "guided", "simple"), "pyramid"),
                _number("iterations", INT, 3, minimum=1, maximum=10, step=1),
                _number("scale_factor", FLOAT, 1.0, minimum=1.0, maximum=8.0, step=0.01),
                _number("blur_strength", FLOAT, 2.0, minimum=1.0, maximum=10.0, step=0.1),
                _number("radius", INT, 7, minimum=1, maximum=20, step=1),
                _number("epsilon", FLOAT, 0.01, minimum=0.001, maximum=0.1, step=0.001),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("tile preprocessor", "guided filter", "ttplanet", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        method: str = "pyramid",
        iterations: int = 3,
        scale_factor: float = 1.0,
        blur_strength: float = 2.0,
        radius: int = 7,
        epsilon: float = 0.01,
    ) -> Mapping[str, object]:
        if method not in ("pyramid", "guided", "simple"):
            raise ValueError(f"unknown tile hint preprocessor: {method}")
        if not 1 <= iterations <= 10:
            raise ValueError("iterations must be between 1 and 10")
        if not math.isfinite(scale_factor) or not 1.0 <= scale_factor <= 8.0:
            raise ValueError("scale_factor must be between 1 and 8")
        if not math.isfinite(blur_strength) or not 1.0 <= blur_strength <= 10.0:
            raise ValueError("blur_strength must be between 1 and 10")
        if not 1 <= radius <= 20:
            raise ValueError("radius must be between 1 and 20")
        if not math.isfinite(epsilon) or not 0.001 <= epsilon <= 0.1:
            raise ValueError("epsilon must be between 0.001 and 0.1")

        def transform(frame: np.ndarray) -> np.ndarray:
            if method == "pyramid":
                height, width = int(frame.shape[0]), int(frame.shape[1])
                target_height = int(np.round(height / 64.0)) * 64
                target_width = int(np.round(width / 64.0)) * 64
                divisor = 2**iterations
                if target_height < divisor or target_width < divisor:
                    raise ValueError("image is too small for the requested pyramid iterations")
                result = cv2.resize(
                    frame,
                    (target_width // divisor, target_height // divisor),
                    interpolation=cv2.INTER_AREA,
                )
                for _ in range(iterations):
                    result = cv2.pyrUp(result)
                return result
            bgr = frame[:, :, ::-1]
            height, width = bgr.shape[:2]
            target_width, target_height = int(width / scale_factor), int(height / scale_factor)
            if target_width < 1 or target_height < 1:
                raise ValueError("tile scale exceeds the image dimensions")
            if method == "guided":
                processed = _fast_guided_filter(
                    _blur_for_tile(bgr, blur_strength), radius, epsilon, scale_factor
                )
                reduced = cv2.resize(
                    processed,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
                restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
            else:
                reduced = cv2.resize(
                    bgr,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
                restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LANCZOS4)
                restored = _blur_for_tile(restored, blur_strength)
            return restored[:, :, ::-1]

        return cls.outputs(image=_map_frames(image, transform))


class InpaintHintPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.inpaint_hint",
            display_name="Preprocess Inpaint Hint",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("mask", MASK),
                _combo("masked_value", ("negative_one", "black"), "negative_one"),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("inpaint preprocessor", "controlnet inpaint", "inpaint hint"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        mask: object,
        masked_value: str = "negative_one",
    ) -> Mapping[str, object]:
        if masked_value not in ("negative_one", "black"):
            raise ValueError(f"unknown masked pixel value: {masked_value}")
        source = _image_array(image)
        _check_output_size(tuple(int(size) for size in source.shape))
        selected = _mask_array(mask)
        if len(selected) not in (1, len(source)):
            raise ValueError("mask batch must contain one frame or match the image batch")
        if selected.shape[1:3] != source.shape[1:3]:
            selected = _resize_array(
                selected,
                int(source.shape[2]),
                int(source.shape[1]),
                "bilinear",
            )
        if len(selected) == 1 and len(source) > 1:
            selected = np.repeat(selected, len(source), axis=0)
        output = np.array(source, copy=True)
        output[selected > 0.5] = -1.0 if masked_value == "negative_one" else 0.0
        return cls.outputs(image=np.ascontiguousarray(output, dtype=np.float32))


class HintImageResize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.hint_resize",
            display_name="Resize Control Hint",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _number("target_width", INT, 512, minimum=64, maximum=8192, step=8),
                _number("target_height", INT, 512, minimum=64, maximum=8192, step=8),
                _combo("resize_mode", ("stretch", "fill", "fit"), "stretch"),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=(
                "enhance hint image",
                "controlnet hint resize",
                "HintImageEnchance",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        target_width: int = 512,
        target_height: int = 512,
        resize_mode: str = "stretch",
    ) -> Mapping[str, object]:
        if not 64 <= target_width <= 8192 or not 64 <= target_height <= 8192:
            raise ValueError("target dimensions must be between 64 and 8192")
        if resize_mode not in ("stretch", "fill", "fit"):
            raise ValueError(f"unknown hint resize mode: {resize_mode}")
        source = _image_array(image)
        _check_output_size((len(source), target_height, target_width, 3))
        return cls.outputs(
            image=_image_output(
                [
                    _resize_hint_frame(frame, target_width, target_height, resize_mode)
                    for frame in _uint8_frames(source)
                ]
            )
        )


class HintResolution(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.hint_resolution",
            display_name="Calculate Hint Resolution",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _number("target_width", INT, 512, minimum=64, maximum=8192, step=8),
                _number("target_height", INT, 512, minimum=64, maximum=8192, step=8),
                _combo("resize_mode", ("stretch", "fill", "fit"), "stretch"),
            ),
            outputs=(OutputSpec("resolution", INT),),
            search_terms=("pixel perfect resolution", "controlnet resolution", "hint size"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        target_width: int = 512,
        target_height: int = 512,
        resize_mode: str = "stretch",
    ) -> Mapping[str, object]:
        source = _image_array(image)
        if not 64 <= target_width <= 8192 or not 64 <= target_height <= 8192:
            raise ValueError("target dimensions must be between 64 and 8192")
        if resize_mode not in ("stretch", "fill", "fit"):
            raise ValueError(f"unknown hint resize mode: {resize_mode}")
        source_height, source_width = int(source.shape[1]), int(source.shape[2])
        height_scale = float(target_height) / float(source_height)
        width_scale = float(target_width) / float(source_width)
        scale = (
            min(height_scale, width_scale)
            if resize_mode == "fit"
            else max(height_scale, width_scale)
        )
        resolution = int(np.round(scale * float(min(source_height, source_width))))
        return cls.outputs(resolution=resolution)


def _model_edge_provider_input(choice: str) -> InputSpec:
    return InputSpec(
        "provider",
        TypeExpr.concrete(CORE_COMBO),
        required=False,
        widget=ComboWidget(
            remote_route=f"/api/choices/{choice}",
        ),
        hidden=True,
    )


def _model_edge_resolution(*, default: int = 512, step: int = 64) -> InputSpec:
    return _number(
        "resolution",
        INT,
        default,
        minimum=64,
        maximum=MAX_RESOLUTION,
        step=step,
    )


class _ModelEdgeProviderNode(Node):
    @classmethod
    def execute(cls, *args: Any, **inputs: Any) -> Mapping[str, object]:
        del args, inputs
        raise RuntimeError("model edge preprocessing requires an installed provider")


class ModelEdgePreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.model_edges",
            display_name="Preprocess Model Edges",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input(MODEL_EDGE_PROVIDER_CHOICE),
                InputSpec("safe", BOOLEAN, required=False, default=True),
                InputSpec("scribble", BOOLEAN, required=False, default=False),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("HED", "soft edge", "fake scribble", "controlnet"),
        )


class RealisticLineartPreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_realistic",
            display_name="Preprocess Realistic Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.lineart_realistic.providers"),
                InputSpec("coarse", BOOLEAN, required=False, default=False),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("realistic lineart", "coarse lineart", "controlnet"),
        )


class AnimeLineartPreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_anime",
            display_name="Preprocess Anime Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.lineart_anime.providers"),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("anime lineart", "controlnet"),
        )


class MangaLineartPreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.lineart_manga",
            display_name="Preprocess Manga Line Art",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.lineart_manga.providers"),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("manga lineart", "anime denoise", "controlnet"),
        )


class AnyLinePreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.anyline",
            display_name="Preprocess AnyLine",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.anyline.providers"),
                _combo(
                    "merge_with_lineart",
                    (
                        "lineart_standard",
                        "lineart_realisitic",
                        "lineart_anime",
                        "manga_line",
                    ),
                    "lineart_standard",
                ),
                _model_edge_resolution(default=1280, step=8),
                _number(
                    "lineart_lower_bound",
                    FLOAT,
                    0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                ),
                _number(
                    "lineart_upper_bound",
                    FLOAT,
                    1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                ),
                _number(
                    "object_min_size",
                    INT,
                    36,
                    minimum=1,
                    maximum=MAX_RESOLUTION,
                    step=1,
                ),
                _number(
                    "object_connectivity",
                    INT,
                    1,
                    minimum=1,
                    maximum=MAX_RESOLUTION,
                    step=1,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("AnyLine", "MTEED", "lineart", "controlnet"),
        )


class TEEDPreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.teed",
            display_name="Preprocess TEED Edges",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.teed.providers"),
                _number("safe_steps", INT, 2, minimum=0, maximum=10, step=1),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("TEED", "soft edge", "controlnet"),
        )


class MLSDPreprocessor(_ModelEdgeProviderNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.mlsd",
            display_name="Preprocess M-LSD Lines",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                _model_edge_provider_input("dinkster.preprocess.mlsd.providers"),
                _number(
                    "score_threshold",
                    FLOAT,
                    0.1,
                    minimum=0.01,
                    maximum=2.0,
                    step=0.01,
                ),
                _number(
                    "distance_threshold",
                    FLOAT,
                    0.1,
                    minimum=0.01,
                    maximum=20.0,
                    step=0.01,
                ),
                _model_edge_resolution(),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("M-LSD", "line segment", "controlnet"),
        )


class ModelDepthPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.model_depth",
            display_name="Preprocess Model Depth",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "model",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        options=(
                            ComboOption("auto", "Automatic"),
                            ComboOption("depth-anything-v3", "Depth Anything V3"),
                            ComboOption(
                                "depth-anything-v2-large",
                                "Depth Anything V2 Large",
                            ),
                        ),
                    ),
                ),
                InputSpec(
                    "provider",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    widget=ComboWidget(
                        remote_route=f"/api/choices/{MODEL_DEPTH_PROVIDER_CHOICE}",
                    ),
                    hidden=True,
                ),
                _number(
                    "resolution",
                    INT,
                    512,
                    minimum=64,
                    maximum=MAX_RESOLUTION,
                    step=64,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("depth", "relative depth", "Depth Anything V2", "controlnet"),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("model depth preprocessing requires an installed provider")


def preprocessor_choices() -> dict[str, tuple[str, ...]]:
    return {
        MODEL_DEPTH_PROVIDER_CHOICE: (),
        **dict.fromkeys(MODEL_EDGE_PROVIDER_CHOICES, ()),
    }


PREPROCESSOR_NODES: tuple[type[Node], ...] = (
    EdgePreprocessor,
    LineartPreprocessor,
    ScribblePreprocessor,
    BinaryPreprocessor,
    ColorHintPreprocessor,
    ContentShufflePreprocessor,
    TileHintPreprocessor,
    InpaintHintPreprocessor,
    HintImageResize,
    HintResolution,
    ModelEdgePreprocessor,
    RealisticLineartPreprocessor,
    AnimeLineartPreprocessor,
    MangaLineartPreprocessor,
    AnyLinePreprocessor,
    TEEDPreprocessor,
    MLSDPreprocessor,
    ModelDepthPreprocessor,
)


__all__ = [
    "MODEL_DEPTH_PROVIDER_CHOICE",
    "MODEL_EDGE_PROVIDER_CHOICE",
    "MODEL_EDGE_PROVIDER_CHOICES",
    "PREPROCESSOR_NODES",
    "AnimeLineartPreprocessor",
    "AnyLinePreprocessor",
    "BinaryPreprocessor",
    "ColorHintPreprocessor",
    "ContentShufflePreprocessor",
    "EdgePreprocessor",
    "HintImageResize",
    "HintResolution",
    "InpaintHintPreprocessor",
    "LineartPreprocessor",
    "MLSDPreprocessor",
    "MangaLineartPreprocessor",
    "ModelDepthPreprocessor",
    "ModelEdgePreprocessor",
    "RealisticLineartPreprocessor",
    "ScribblePreprocessor",
    "TEEDPreprocessor",
    "TileHintPreprocessor",
    "preprocessor_choices",
]
