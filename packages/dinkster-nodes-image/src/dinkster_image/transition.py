"""Deterministic image batch transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    image_math,
)

from .geometry import BOOLEAN, FLOAT, IMAGE, INT
from .migration import with_v1_migration
from .support import (
    MAX_DIMENSION,
    Interpolation,
    check_output_size,
    combo_input,
    image_array,
    indexed_family_values,
    materialized_inputs,
    number_input,
    resize_array,
)

masked_transition = image_math.masked_transition

_MODES = ("between_inputs", "within_batch", "join_batches")
_TRANSITIONS = (
    "horizontal_slide",
    "vertical_slide",
    "box",
    "circle",
    "horizontal_door",
    "vertical_door",
    "fade",
)
_EASINGS = (
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "bounce",
    "elastic",
    "glitchy",
    "exponential_ease_out",
)


def _ease(value: float, easing: str) -> float:
    if easing == "linear":
        return value
    if easing == "ease_in":
        return value * value
    if easing == "ease_out":
        return 1.0 - (1.0 - value) * (1.0 - value)
    if easing == "ease_in_out":
        return 3.0 * value * value - 2.0 * value * value * value
    if easing == "bounce":
        if value < 0.5:
            scaled = value * 2.0
            return (1.0 - (1.0 - scaled) * (1.0 - scaled)) * 0.5
        scaled = (value - 0.5) * 2.0
        return scaled * scaled * 0.5 + 0.5
    if easing == "elastic":
        return math.sin(13.0 * math.pi * value / 2.0) * math.pow(2.0, 10.0 * (value - 1.0))
    if easing == "glitchy":
        return value + 0.1 * math.sin(40.0 * value)
    if easing == "exponential_ease_out":
        return 1.0 - (1.0 - value) ** 4
    raise ValueError(f"unknown transition easing: {easing}")


def _blur_mask(mask: np.ndarray, blur_radius: float) -> np.ndarray:
    if not math.isfinite(blur_radius) or blur_radius < 0.0:
        raise ValueError("blur_radius must be finite and non-negative")
    if blur_radius == 0.0:
        return mask
    kernel_size = int(blur_radius * 2.0) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = blur_radius / 3.0
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma)).astype(np.float32)
    kernel /= kernel.sum(dtype=np.float64)
    horizontal = np.pad(mask, ((0, 0), (radius, radius)), mode="constant")
    horizontal_windows = np.lib.stride_tricks.sliding_window_view(horizontal, kernel_size, axis=1)
    horizontal = np.einsum("ijk,k->ij", horizontal_windows, kernel, dtype=np.float32, optimize=True)
    vertical = np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant")
    vertical_windows = np.lib.stride_tricks.sliding_window_view(vertical, kernel_size, axis=0)
    return np.einsum("ijk,k->ij", vertical_windows, kernel, dtype=np.float32, optimize=True)


def _transition_mask(
    height: int,
    width: int,
    alpha: float,
    transition: str,
    blur_radius: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    if transition == "horizontal_slide":
        mask[:, : round(width * alpha)] = 1.0
    elif transition == "vertical_slide":
        mask[: round(height * alpha), :] = 1.0
    elif transition == "box":
        box_width, box_height = round(width * alpha), round(height * alpha)
        left, top = (width - box_width) // 2, (height - box_height) // 2
        mask[top : top + box_height, left : left + box_width] = 1.0
    elif transition == "circle":
        radius = math.ceil(math.sqrt(width * width + height * height) * alpha / 2.0)
        x = np.arange(width, dtype=np.float32)[None, :]
        y = np.arange(height, dtype=np.float32)[:, None]
        mask[np.square(x - width // 2) + np.square(y - height // 2) <= radius * radius] = 1.0
    elif transition == "horizontal_door":
        bar = math.ceil(height * alpha / 2.0)
        if bar > 0:
            mask[:bar, :] = 1.0
            mask[-bar:, :] = 1.0
    elif transition == "vertical_door":
        bar = math.ceil(width * alpha / 2.0)
        if bar > 0:
            mask[:, :bar] = 1.0
            mask[:, -bar:] = 1.0
    elif transition == "fade":
        mask.fill(alpha)
    else:
        raise ValueError(f"unknown transition type: {transition}")
    return _blur_mask(mask, blur_radius)


def _transition_frame(
    first: np.ndarray,
    second: np.ndarray,
    *,
    alpha: float,
    transition: str,
    blur_radius: float,
    reverse: bool,
) -> np.ndarray:
    if reverse:
        first, second = second, first
        alpha = 1.0 - alpha
    mask = _transition_mask(first.shape[0], first.shape[1], alpha, transition, blur_radius)
    return masked_transition(first, second, mask)


def _transition_frames(
    first: np.ndarray,
    second: np.ndarray,
    *,
    frames: int,
    transition: str,
    easing: str,
    blur_radius: float,
    reverse: bool,
) -> np.ndarray:
    if frames < 2:
        raise ValueError(f"transitioning_frames must be at least 2, got {frames}")
    check_output_size((frames, int(first.shape[0]), int(first.shape[1]), int(first.shape[2])))
    output = [
        _transition_frame(
            first,
            second,
            alpha=_ease(index / (frames - 1), easing),
            transition=transition,
            blur_radius=blur_radius,
            reverse=reverse,
        )
        for index in range(frames)
    ]
    return np.ascontiguousarray(np.stack(output, axis=0), dtype=np.float32)


def _normalize_images(
    images: Mapping[str, object], input_count: int, shape_policy: str, interpolation: str
) -> list[np.ndarray]:
    if shape_policy not in ("strict", "resize_to_first"):
        raise ValueError(f"unknown shape policy: {shape_policy}")
    ordered = indexed_family_values(
        images,
        count=input_count,
        member_prefix="image_",
        subject="images",
    )
    arrays = [
        image_array(value, subject=f"image {index}") if value is not None else None
        for index, value in enumerate(ordered, start=1)
    ]
    first = arrays[0]
    assert first is not None
    normalized: list[np.ndarray] = []
    for index, array in enumerate(arrays, start=1):
        if array is None:
            normalized.append(np.zeros_like(first))
            continue
        if array.shape[3] != first.shape[3]:
            raise ValueError(f"image {index} channel count must match the first image")
        if array.shape[1:3] != first.shape[1:3]:
            if shape_policy == "strict":
                raise ValueError(f"image {index} dimensions must match the first image")
            array = resize_array(
                array,
                int(first.shape[2]),
                int(first.shape[1]),
                cast("Interpolation", interpolation),
            )
        normalized.append(array)
    return normalized


def _between_inputs(
    arrays: list[np.ndarray],
    *,
    frames: int,
    transition: str,
    easing: str,
    blur_radius: float,
    reverse: bool,
) -> np.ndarray:
    if len(arrays) < 2:
        raise ValueError("between_inputs requires at least two image inputs")
    total_frames = sum(len(array) for array in arrays) + (len(arrays) - 1) * frames
    check_output_size(
        (
            total_frames,
            int(arrays[0].shape[1]),
            int(arrays[0].shape[2]),
            int(arrays[0].shape[3]),
        )
    )
    output = [arrays[0]]
    previous = arrays[0][-1]
    for array in arrays[1:]:
        between = _transition_frames(
            previous,
            array[0],
            frames=frames,
            transition=transition,
            easing=easing,
            blur_radius=blur_radius,
            reverse=reverse,
        )
        output.extend((between, array))
        previous = array[-1]
    return np.ascontiguousarray(np.concatenate(output, axis=0))


def _within_batch(
    array: np.ndarray,
    *,
    frames: int,
    transition: str,
    easing: str,
    blur_radius: float,
    reverse: bool,
) -> np.ndarray:
    if len(array) == 1:
        return array
    check_output_size(
        (
            (len(array) - 1) * frames,
            int(array.shape[1]),
            int(array.shape[2]),
            int(array.shape[3]),
        )
    )
    return np.ascontiguousarray(
        np.concatenate(
            [
                _transition_frames(
                    array[index],
                    array[index + 1],
                    frames=frames,
                    transition=transition,
                    easing=easing,
                    blur_radius=blur_radius,
                    reverse=reverse,
                )
                for index in range(len(array) - 1)
            ],
            axis=0,
        )
    )


def _join_batches(
    first: np.ndarray,
    second: np.ndarray,
    *,
    start_index: int,
    frames: int,
    transition: str,
    easing: str,
    blur_radius: float,
    reverse: bool,
) -> np.ndarray:
    if start_index < 0:
        start_index += len(first)
    if not 0 <= start_index <= len(first):
        raise ValueError("start_index is outside the first image batch")
    count = min(frames, len(first) - start_index, len(second))
    check_output_size(
        (
            start_index + len(second),
            int(first.shape[1]),
            int(first.shape[2]),
            int(first.shape[3]),
        )
    )
    output: list[np.ndarray] = []
    if start_index:
        output.append(first[:start_index])
    for index in range(count):
        alpha = _ease(index / (count - 1) if count > 1 else 1.0, easing)
        frame = _transition_frame(
            first[start_index + index],
            second[index],
            alpha=alpha,
            transition=transition,
            blur_radius=blur_radius,
            reverse=reverse,
        )
        output.append(frame[None, ...])
    if count < len(second):
        output.append(second[count:])
    if not output:
        raise ValueError("join_batches produced no images")
    result = np.concatenate(output, axis=0)
    return np.ascontiguousarray(result, dtype=np.float32)


class ImageTransition(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.image.transition",
                display_name="Transition Images",
                category="image/batch",
                inputs=(
                    combo_input("transition", _TRANSITIONS, "fade"),
                    number_input("transitioning_frames", INT, 2, minimum=0, maximum=4096, step=1),
                    combo_input("easing", _EASINGS, "linear"),
                    number_input("blur_radius", FLOAT, 0.0, minimum=0.0, maximum=100.0, step=0.1),
                    InputSpec("reverse", BOOLEAN, required=False, default=False),
                    number_input(
                        "input_count", INT, 0, minimum=0, maximum=1000, step=1, hidden=True
                    ),
                ),
                combos=(
                    DynamicComboSpec(
                        "mode",
                        (
                            DynamicComboOption("between_inputs"),
                            DynamicComboOption("within_batch"),
                            DynamicComboOption(
                                "join_batches",
                                (
                                    number_input(
                                        "start_index",
                                        INT,
                                        0,
                                        minimum=-MAX_DIMENSION,
                                        maximum=MAX_DIMENSION,
                                        step=1,
                                    ),
                                ),
                            ),
                        ),
                        default="between_inputs",
                    ),
                    DynamicComboSpec(
                        "shape_policy",
                        (
                            DynamicComboOption("strict"),
                            DynamicComboOption(
                                "resize_to_first",
                                (
                                    combo_input(
                                        "resize_interpolation",
                                        ("nearest-exact", "bilinear", "bicubic", "lanczos", "area"),
                                        "lanczos",
                                        advanced=True,
                                    ),
                                ),
                            ),
                        ),
                        default="strict",
                    ),
                ),
                input_families=(
                    InputFamilySpec(
                        "images", IMAGE, min_members=1, max_members=1000, member_prefix="image_"
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "ImageBatchJoinWithTransition",
                    "TransitionImagesMulti",
                    "TransitionImagesInBatch",
                    "crossfade images",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        images: Mapping[str, object],
        mode: str = "between_inputs",
        transition: str = "fade",
        transitioning_frames: int = 2,
        easing: str = "linear",
        blur_radius: float = 0.0,
        reverse: bool = False,
        start_index: int = 0,
        shape_policy: str = "strict",
        resize_interpolation: str = "lanczos",
        input_count: int = 0,
    ) -> Mapping[str, object]:
        arrays = _normalize_images(images, input_count, shape_policy, resize_interpolation)
        if mode == "between_inputs":
            if transitioning_frames < 2:
                raise ValueError("between_inputs requires at least 2 transitioning frames")
            output = _between_inputs(
                arrays,
                frames=transitioning_frames,
                transition=transition,
                easing=easing,
                blur_radius=blur_radius,
                reverse=reverse,
            )
        elif mode == "within_batch":
            if len(arrays) != 1:
                raise ValueError("within_batch requires exactly one image input")
            if len(arrays[0]) > 1 and transitioning_frames < 2:
                raise ValueError("within_batch requires at least 2 transitioning frames")
            output = _within_batch(
                arrays[0],
                frames=transitioning_frames,
                transition=transition,
                easing=easing,
                blur_radius=blur_radius,
                reverse=reverse,
            )
        elif mode == "join_batches":
            if len(arrays) != 2:
                raise ValueError("join_batches requires exactly two image inputs")
            if transitioning_frames < 1:
                raise ValueError("join_batches requires at least 1 transitioning frame")
            output = _join_batches(
                arrays[0],
                arrays[1],
                start_index=start_index,
                frames=transitioning_frames,
                transition=transition,
                easing=easing,
                blur_radius=blur_radius,
                reverse=reverse,
            )
        else:
            raise ValueError(f"unknown image transition mode: {mode}")
        return cls.outputs(image=output)


TRANSITION_NODES: tuple[type[Node], ...] = (ImageTransition,)


__all__ = ["TRANSITION_NODES", "ImageTransition"]
