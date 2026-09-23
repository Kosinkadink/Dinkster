"""Deterministic CPU image compositing over BHWC NumPy arrays."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SlotVariant,
    annotate_mask,
    copy_media_semantics,
)

from .geometry import (
    BOOLEAN,
    FLOAT,
    IMAGE,
    INT,
    MASK,
)
from .migration import with_mask_polarity, with_v1_migration
from .support import INTERPOLATION_OPTIONS as _INTERPOLATION_OPTIONS
from .support import MAX_DIMENSION, Interpolation, materialized_inputs, normalize_mask_polarity
from .support import combo_input as _combo
from .support import image_array as _image_array
from .support import mask_array as _mask_array
from .support import resize_array as _resize_array

BLEND_MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "soft_light",
    "signed_difference",
)
PORTER_DUFF_MODES = (
    "ADD",
    "CLEAR",
    "DARKEN",
    "DST",
    "DST_ATOP",
    "DST_IN",
    "DST_OUT",
    "DST_OVER",
    "LIGHTEN",
    "MULTIPLY",
    "OVERLAY",
    "SCREEN",
    "SRC",
    "SRC_ATOP",
    "SRC_IN",
    "SRC_OUT",
    "SRC_OVER",
    "XOR",
)


def _repeat_batch(array: np.ndarray, batch: int) -> np.ndarray:
    return np.ascontiguousarray(array[np.arange(batch) % array.shape[0]])


def _reconcile_batches(
    arrays: Sequence[tuple[str, np.ndarray]],
    policy: str,
    *,
    destination_batch: int | None = None,
) -> tuple[np.ndarray, ...]:
    sizes = tuple(int(array.shape[0]) for _, array in arrays)
    if policy == "strict":
        if len(set(sizes)) != 1:
            raise ValueError(f"strict batch policy requires equal batches, got {sizes}")
        batch = sizes[0]
    elif policy == "singleton_broadcast":
        batch = max(sizes)
        invalid = tuple(size for size in sizes if size not in (1, batch))
        if invalid:
            raise ValueError(
                f"singleton_broadcast requires each batch to be 1 or {batch}, got {sizes}"
            )
    elif policy == "destination_repeat":
        if destination_batch is None:
            raise ValueError("destination_repeat requires a destination batch")
        batch = destination_batch
    elif policy == "truncate_to_shortest":
        batch = min(sizes)
    else:
        raise ValueError(f"unknown batch policy: {policy}")

    reconciled: list[np.ndarray] = []
    for _, array in arrays:
        if array.shape[0] == batch:
            reconciled.append(array)
        elif policy == "singleton_broadcast" and array.shape[0] == 1:
            reconciled.append(np.broadcast_to(array, (batch, *array.shape[1:])))
        elif policy == "strict":
            raise AssertionError("strict batches were validated above")
        elif policy == "truncate_to_shortest":
            reconciled.append(array[:batch])
        else:
            reconciled.append(_repeat_batch(array, batch))
    return tuple(reconciled)


def _match_channels(source: np.ndarray, destination_channels: int) -> np.ndarray:
    source_channels = int(source.shape[3])
    if source_channels == destination_channels:
        return source
    if source_channels == 4 and destination_channels == 3:
        return source[..., :3]
    if source_channels == 3 and destination_channels == 4:
        alpha = np.ones((*source.shape[:3], 1), dtype=np.float32)
        return np.concatenate((source, alpha), axis=3)
    if source_channels == 1 and destination_channels in (3, 4):
        rgb = np.repeat(source, 3, axis=3)
        if destination_channels == 3:
            return rgb
        alpha = np.ones((*source.shape[:3], 1), dtype=np.float32)
        return np.concatenate((rgb, alpha), axis=3)
    raise ValueError(
        f"source channels cannot be conformed from {source_channels} to {destination_channels}"
    )


def _blend(destination: np.ndarray, source: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return source
    if mode == "multiply":
        return destination * source
    if mode == "screen":
        return 1.0 - (1.0 - destination) * (1.0 - source)
    if mode == "overlay":
        return np.where(
            destination <= 0.5,
            2.0 * destination * source,
            1.0 - 2.0 * (1.0 - destination) * (1.0 - source),
        )
    if mode == "soft_light":
        curve = np.where(
            destination <= 0.25,
            ((16.0 * destination - 12.0) * destination + 4.0) * destination,
            np.sqrt(np.clip(destination, 0.0, None)),
        )
        return np.where(
            source <= 0.5,
            destination - (1.0 - 2.0 * source) * destination * (1.0 - destination),
            destination + (2.0 * source - 1.0) * (curve - destination),
        )
    if mode == "signed_difference":
        return destination - source
    raise ValueError(f"unknown blend mode: {mode}")


def _center_crop_resize(
    array: np.ndarray,
    width: int,
    height: int,
    interpolation: Interpolation,
) -> np.ndarray:
    source_height, source_width = int(array.shape[1]), int(array.shape[2])
    source_aspect = source_width / source_height
    target_aspect = width / height
    left = top = 0
    if source_aspect > target_aspect:
        left = round((source_width - source_width * (target_aspect / source_aspect)) / 2)
    elif source_aspect < target_aspect:
        top = round((source_height - source_height * (source_aspect / target_aspect)) / 2)
    cropped = array[:, top : source_height - top, left : source_width - left, ...]
    return _resize_array(cropped, width, height, interpolation)


class ImageComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        interpolation = _combo(
            "interpolation",
            _INTERPOLATION_OPTIONS,
            "bilinear",
            advanced=True,
        )
        mask_polarity = _combo(
            "mask_polarity",
            ("opacity", "transparency"),
            "opacity",
            advanced=True,
        )
        return with_mask_polarity(
            with_v1_migration(
                NodeSchema(
                    node_type="dinkster.image.composite",
                    version=2,
                    display_name="Composite Images",
                    category="image/composition",
                    inputs=(
                        InputSpec("destination", IMAGE),
                        InputSpec("source", IMAGE),
                        InputSpec(
                            "x",
                            INT,
                            required=False,
                            default=0,
                            widget=NumberWidget(min=-MAX_DIMENSION, max=MAX_DIMENSION, step=1),
                        ),
                        InputSpec(
                            "y",
                            INT,
                            required=False,
                            default=0,
                            widget=NumberWidget(min=-MAX_DIMENSION, max=MAX_DIMENSION, step=1),
                        ),
                        _combo("blend_mode", BLEND_MODES, "normal"),
                        InputSpec(
                            "factor",
                            FLOAT,
                            required=False,
                            default=1.0,
                            widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                        ),
                        InputSpec(
                            "clamp_output",
                            BOOLEAN,
                            required=False,
                            default=True,
                            advanced=True,
                        ),
                        InputSpec(
                            "preserve_destination_alpha",
                            BOOLEAN,
                            required=False,
                            default=False,
                            advanced=True,
                        ),
                        _combo(
                            "batch_policy",
                            ("singleton_broadcast", "strict", "destination_repeat"),
                            "singleton_broadcast",
                            advanced=True,
                        ),
                    ),
                    combos=(
                        DynamicComboSpec(
                            "source_resize",
                            (
                                DynamicComboOption("none"),
                                DynamicComboOption("stretch", (interpolation,)),
                                DynamicComboOption("fill", (interpolation,)),
                            ),
                            default="none",
                        ),
                    ),
                    slots=(
                        DynamicSlotSpec(
                            "mask",
                            variants=(SlotVariant("mask", MASK, inputs=(mask_polarity,)),),
                            required=False,
                        ),
                    ),
                    outputs=(OutputSpec("image", IMAGE, preview=True),),
                    search_terms=(
                        "blend images",
                        "paste image",
                        "ImageCompositeMasked",
                        "ImageBlend",
                        "ImageComposite+",
                        "CrossFadeImages",
                    ),
                )
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        destination: object,
        source: object,
        x: int = 0,
        y: int = 0,
        mask: object | None = None,
        blend_mode: str = "normal",
        factor: float = 1.0,
        source_resize: str = "none",
        interpolation: str = "bilinear",
        mask_polarity: str = "coverage",
        clamp_output: bool = True,
        preserve_destination_alpha: bool = False,
        batch_policy: str = "singleton_broadcast",
    ) -> Mapping[str, object]:
        if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError(f"factor must be between 0 and 1, got {factor}")
        mask_polarity = normalize_mask_polarity(mask_polarity)
        if mask_polarity not in ("coverage", "transparency"):
            raise ValueError(f"unknown mask polarity: {mask_polarity}")
        destination_array = _image_array(destination, subject="destination")
        source_array = _match_channels(
            _image_array(source, subject="source"), int(destination_array.shape[3])
        )
        if source_resize == "stretch":
            source_array = _resize_array(
                source_array,
                int(destination_array.shape[2]),
                int(destination_array.shape[1]),
                cast("Interpolation", interpolation),
            )
        elif source_resize == "fill":
            source_array = _center_crop_resize(
                source_array,
                int(destination_array.shape[2]),
                int(destination_array.shape[1]),
                cast("Interpolation", interpolation),
            )
        elif source_resize != "none":
            raise ValueError(f"unknown source resize mode: {source_resize}")
        mask_array: np.ndarray | None = None
        if mask is not None:
            mask_array = _mask_array(mask)
            if mask_array.shape[1:3] != source_array.shape[1:3]:
                mask_array = _resize_array(
                    mask_array,
                    int(source_array.shape[2]),
                    int(source_array.shape[1]),
                    "bilinear",
                )
        named_arrays = [("destination", destination_array), ("source", source_array)]
        if mask_array is not None:
            named_arrays.append(("mask", mask_array))
        reconciled = _reconcile_batches(
            named_arrays,
            batch_policy,
            destination_batch=int(destination_array.shape[0]),
        )
        destination_array, source_array = reconciled[:2]
        if mask_array is not None:
            mask_array = reconciled[2]

        output = np.array(destination_array, copy=True)
        destination_height, destination_width = output.shape[1:3]
        source_height, source_width = source_array.shape[1:3]
        destination_left = max(0, x)
        destination_top = max(0, y)
        destination_right = min(destination_width, x + source_width)
        destination_bottom = min(destination_height, y + source_height)
        if destination_right <= destination_left or destination_bottom <= destination_top:
            return cls.outputs(
                image=copy_media_semantics(destination, np.ascontiguousarray(output))
            )

        source_left = destination_left - x
        source_top = destination_top - y
        source_right = source_left + destination_right - destination_left
        source_bottom = source_top + destination_bottom - destination_top
        destination_view = output[
            :, destination_top:destination_bottom, destination_left:destination_right, :
        ]
        source_view = source_array[:, source_top:source_bottom, source_left:source_right, :]
        blended = _blend(destination_view, source_view, blend_mode)
        if mask_array is None:
            alpha: float | np.ndarray = factor
        else:
            mask_view = np.clip(
                mask_array[:, source_top:source_bottom, source_left:source_right, None],
                0.0,
                1.0,
            )
            if mask_polarity == "transparency":
                mask_view = 1.0 - mask_view
            alpha = factor * mask_view
        composed = destination_view * (1.0 - alpha) + blended * alpha
        if clamp_output:
            composed = np.clip(composed, 0.0, 1.0)
        if preserve_destination_alpha and composed.shape[3] == 4:
            composed[..., 3] = destination_view[..., 3]
        output[:, destination_top:destination_bottom, destination_left:destination_right, :] = (
            composed
        )
        return cls.outputs(image=copy_media_semantics(destination, np.ascontiguousarray(output)))


def _porter_duff(
    source: np.ndarray,
    source_opacity: np.ndarray,
    destination: np.ndarray,
    destination_opacity: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_premultiplied = source * source_opacity
    destination_premultiplied = destination * destination_opacity
    if mode == "ADD":
        output_opacity = np.clip(source_opacity + destination_opacity, 0.0, 1.0)
        output = np.clip(source_premultiplied + destination_premultiplied, 0.0, 1.0)
    elif mode == "CLEAR":
        output_opacity = np.zeros_like(destination_opacity)
        output = np.zeros_like(destination_premultiplied)
    elif mode == "DARKEN":
        output_opacity = source_opacity + destination_opacity - source_opacity * destination_opacity
        output = (
            (1.0 - destination_opacity) * source_premultiplied
            + (1.0 - source_opacity) * destination_premultiplied
            + np.minimum(
                source_premultiplied * destination_opacity,
                destination_premultiplied * source_opacity,
            )
        )
    elif mode == "DST":
        output_opacity = destination_opacity
        output = destination_premultiplied
    elif mode == "DST_ATOP":
        output_opacity = source_opacity
        output = (
            source_opacity * destination_premultiplied
            + (1.0 - destination_opacity) * source_premultiplied
        )
    elif mode == "DST_IN":
        output_opacity = source_opacity * destination_opacity
        output = destination_premultiplied * source_opacity
    elif mode == "DST_OUT":
        output_opacity = (1.0 - source_opacity) * destination_opacity
        output = (1.0 - source_opacity) * destination_premultiplied
    elif mode == "DST_OVER":
        output_opacity = destination_opacity + (1.0 - destination_opacity) * source_opacity
        output = destination_premultiplied + (1.0 - destination_opacity) * source_premultiplied
    elif mode == "LIGHTEN":
        output_opacity = source_opacity + destination_opacity - source_opacity * destination_opacity
        output = (
            (1.0 - destination_opacity) * source_premultiplied
            + (1.0 - source_opacity) * destination_premultiplied
            + np.maximum(
                source_premultiplied * destination_opacity,
                destination_premultiplied * source_opacity,
            )
        )
    elif mode == "MULTIPLY":
        output_opacity = source_opacity + destination_opacity - source_opacity * destination_opacity
        output = (
            (1.0 - destination_opacity) * source_premultiplied
            + (1.0 - source_opacity) * destination_premultiplied
            + source_premultiplied * destination_premultiplied
        )
    elif mode == "OVERLAY":
        output_opacity = source_opacity + destination_opacity - source_opacity * destination_opacity
        overlap = np.where(
            2.0 * destination_premultiplied < destination_opacity,
            2.0 * source_premultiplied * destination_premultiplied,
            source_opacity * destination_opacity
            - 2.0
            * (source_opacity - source_premultiplied)
            * (destination_opacity - destination_premultiplied),
        )
        output = (
            (1.0 - destination_opacity) * source_premultiplied
            + (1.0 - source_opacity) * destination_premultiplied
            + overlap
        )
    elif mode == "SCREEN":
        output_opacity = source_opacity + destination_opacity - source_opacity * destination_opacity
        output = (
            source_premultiplied
            + destination_premultiplied
            - source_premultiplied * destination_premultiplied
        )
    elif mode == "SRC":
        output_opacity = source_opacity
        output = source_premultiplied
    elif mode == "SRC_ATOP":
        output_opacity = destination_opacity
        output = (
            destination_opacity * source_premultiplied
            + (1.0 - source_opacity) * destination_premultiplied
        )
    elif mode == "SRC_IN":
        output_opacity = source_opacity * destination_opacity
        output = source_premultiplied * destination_opacity
    elif mode == "SRC_OUT":
        output_opacity = (1.0 - destination_opacity) * source_opacity
        output = (1.0 - destination_opacity) * source_premultiplied
    elif mode == "SRC_OVER":
        output_opacity = source_opacity + (1.0 - source_opacity) * destination_opacity
        output = source_premultiplied + (1.0 - source_opacity) * destination_premultiplied
    elif mode == "XOR":
        output_opacity = (1.0 - destination_opacity) * source_opacity + (
            1.0 - source_opacity
        ) * destination_opacity
        output = (1.0 - destination_opacity) * source_premultiplied + (
            1.0 - source_opacity
        ) * destination_premultiplied
    else:
        raise ValueError(f"unknown Porter-Duff mode: {mode}")
    straight = np.divide(
        output,
        output_opacity,
        out=np.zeros_like(output),
        where=output_opacity > 1e-5,
    )
    return np.clip(straight, 0.0, 1.0), output_opacity


class PorterDuffComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_mask_polarity(
            NodeSchema(
                node_type="dinkster.image.porter_duff",
                display_name="Porter-Duff Composite",
                category="image/composition",
                inputs=(
                    InputSpec("source", IMAGE),
                    InputSpec("source_alpha_mask", MASK, mask_semantic="alpha"),
                    InputSpec("destination", IMAGE),
                    InputSpec("destination_alpha_mask", MASK, mask_semantic="alpha"),
                    _combo("mode", PORTER_DUFF_MODES, "SRC_OVER"),
                    _combo(
                        "alpha_mask_polarity",
                        ("transparency", "opacity"),
                        "transparency",
                    ),
                    _combo(
                        "batch_policy",
                        ("singleton_broadcast", "strict", "truncate_to_shortest"),
                        "singleton_broadcast",
                        advanced=True,
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("alpha_mask", MASK, preview=True, mask_semantic="alpha"),
                ),
                search_terms=("alpha composite", "PorterDuffImageComposite", "premultiplied alpha"),
            )
        )

    @classmethod
    def execute(
        cls,
        *,
        source: object,
        source_alpha_mask: object,
        destination: object,
        destination_alpha_mask: object,
        mode: str = "SRC_OVER",
        mask_polarity: str = "transparency",
        batch_policy: str = "singleton_broadcast",
    ) -> Mapping[str, object]:
        mask_polarity = normalize_mask_polarity(mask_polarity)
        if mask_polarity not in ("transparency", "coverage"):
            raise ValueError(f"unknown alpha mask polarity: {mask_polarity}")
        source_array = _image_array(source, subject="source")
        destination_array = _image_array(destination, subject="destination")
        if source_array.shape[3] != destination_array.shape[3]:
            raise ValueError(
                "source and destination channels must match, got "
                f"{source_array.shape[3]} and {destination_array.shape[3]}"
            )
        source_alpha = _mask_array(source_alpha_mask, subject="source_alpha_mask")
        destination_alpha = _mask_array(destination_alpha_mask, subject="destination_alpha_mask")
        destination_height, destination_width = destination_array.shape[1:3]
        if source_array.shape[1:3] != destination_array.shape[1:3]:
            source_array = _center_crop_resize(
                source_array, destination_width, destination_height, "bicubic"
            )
        if source_alpha.shape[1:3] != destination_array.shape[1:3]:
            source_alpha = _center_crop_resize(
                source_alpha, destination_width, destination_height, "bicubic"
            )
        if destination_alpha.shape[1:3] != destination_array.shape[1:3]:
            destination_alpha = _center_crop_resize(
                destination_alpha, destination_width, destination_height, "bicubic"
            )
        source_array, source_alpha, destination_array, destination_alpha = _reconcile_batches(
            (
                ("source", source_array),
                ("source_alpha_mask", source_alpha),
                ("destination", destination_array),
                ("destination_alpha_mask", destination_alpha),
            ),
            batch_policy,
        )
        source_opacity = source_alpha[..., None]
        destination_opacity = destination_alpha[..., None]
        if mask_polarity == "transparency":
            source_opacity = 1.0 - source_opacity
            destination_opacity = 1.0 - destination_opacity
        image, opacity = _porter_duff(
            source_array,
            source_opacity,
            destination_array,
            destination_opacity,
            mode,
        )
        alpha_mask = opacity[..., 0]
        if mask_polarity == "transparency":
            alpha_mask = 1.0 - alpha_mask
        return cls.outputs(
            image=copy_media_semantics(destination, np.ascontiguousarray(image)),
            alpha_mask=annotate_mask(
                np.ascontiguousarray(alpha_mask), polarity=mask_polarity, semantic="alpha"
            ),
        )


COMPOSITION_NODES: tuple[type[Node], ...] = (ImageComposite, PorterDuffComposite)


__all__ = [
    "BLEND_MODES",
    "COMPOSITION_NODES",
    "PORTER_DUFF_MODES",
    "ImageComposite",
    "PorterDuffComposite",
]
