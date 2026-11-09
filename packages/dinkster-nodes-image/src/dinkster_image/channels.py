"""Deterministic image channel and alpha operations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import (
    ConditionalWidgetGroup,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    annotate_image,
    annotate_mask,
    copy_media_semantics,
    media_semantics,
)

from .composition import _reconcile_batches, _repeat_batch  # pyright: ignore[reportPrivateUsage]
from .geometry import IMAGE, MASK
from .migration import with_mask_polarity
from .support import combo_input as _combo
from .support import image_array as _image_array
from .support import mask_array as _mask_array
from .support import normalize_mask_polarity
from .support import resize_array as _resize_array

COLOR_SPACES = ("rgb", "ycbcr")
ALPHA_MASK_POLARITIES = ("coverage", "transparency")


def _rgb(array: np.ndarray) -> np.ndarray:
    if array.shape[3] == 1:
        return np.repeat(array, 3, axis=3)
    return array[..., :3]


def _unassociated_rgb(array: np.ndarray) -> np.ndarray:
    return np.divide(
        array[..., :3],
        array[..., 3:4],
        out=np.zeros_like(array[..., :3]),
        where=array[..., 3:4] != 0,
    )


def _split_planes(rgb: np.ndarray, color_space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if color_space == "rgb":
        return rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    if color_space == "ycbcr":
        red, green, blue = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        blue_difference = 0.5 + (blue - luminance) * 0.564
        red_difference = 0.5 + (red - luminance) * 0.713
        return luminance, blue_difference, red_difference
    raise ValueError(f"unknown color space: {color_space}")


def _alpha_mask(array: np.ndarray, polarity: str) -> np.ndarray:
    polarity = normalize_mask_polarity(polarity)
    if polarity not in ALPHA_MASK_POLARITIES:
        raise ValueError(f"unknown alpha mask polarity: {polarity}")
    opacity = array[..., 3] if array.shape[3] == 4 else np.ones(array.shape[:3], np.float32)
    return 1.0 - opacity if polarity == "transparency" else opacity


class ImageChannelSplit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_mask_polarity(
            NodeSchema(
                node_type="dinkster.image.channels.split",
                display_name="Split Image Channels",
                category="image/channels",
                inputs=(
                    InputSpec("image", IMAGE),
                    _combo("color_space", COLOR_SPACES, "rgb"),
                    _combo(
                        "channel_layout",
                        ("single", "rgb_repeated"),
                        "single",
                    ),
                    _combo(
                        "alpha_mask_polarity",
                        ALPHA_MASK_POLARITIES,
                        "transparency",
                    ),
                    _combo(
                        "single_channel_image",
                        ("rgb", "preserve"),
                        "rgb",
                    ),
                ),
                widget_groups=(
                    ConditionalWidgetGroup(
                        "channel_layout",
                        ("single",),
                        ("single_channel_image",),
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),
                    OutputSpec("channel_1", IMAGE, preview=True, alpha_policy="drop"),
                    OutputSpec("channel_2", IMAGE, preview=True, alpha_policy="drop"),
                    OutputSpec("channel_3", IMAGE, preview=True, alpha_policy="drop"),
                    OutputSpec("alpha_mask", MASK, preview=True, mask_semantic="alpha"),
                ),
                search_terms=(
                    "split RGB",
                    "split YCbCr",
                    "extract alpha",
                    "SplitImageWithAlpha",
                    "SplitImageChannels",
                ),
            )
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        color_space: str = "rgb",
        channel_layout: str = "single",
        mask_polarity: str = "transparency",
        single_channel_image: str = "rgb",
    ) -> Mapping[str, object]:
        array = _image_array(image)
        rgb = np.ascontiguousarray(
            _unassociated_rgb(array)
            if media_semantics(image).get("alpha") == "premultiplied"
            else _rgb(array),
            dtype=np.float32,
        )
        if single_channel_image not in ("rgb", "preserve"):
            raise ValueError(f"unknown single-channel image policy: {single_channel_image}")
        output_image = array if single_channel_image == "preserve" and array.shape[3] == 1 else rgb
        channel_1, channel_2, channel_3 = _split_planes(rgb, color_space)
        if channel_layout == "rgb_repeated":
            channel_1 = np.repeat(channel_1, 3, axis=3)
            channel_2 = np.repeat(channel_2, 3, axis=3)
            channel_3 = np.repeat(channel_3, 3, axis=3)
        elif channel_layout != "single":
            raise ValueError(f"unknown channel layout: {channel_layout}")
        return cls.outputs(
            image=copy_media_semantics(image, output_image),
            channel_1=np.ascontiguousarray(channel_1, dtype=np.float32),
            channel_2=np.ascontiguousarray(channel_2, dtype=np.float32),
            channel_3=np.ascontiguousarray(channel_3, dtype=np.float32),
            alpha_mask=annotate_mask(
                np.ascontiguousarray(_alpha_mask(array, mask_polarity), dtype=np.float32),
                polarity=normalize_mask_polarity(mask_polarity),
                semantic="alpha",
            ),
        )


def _plane(image: object, subject: str) -> np.ndarray:
    array = _image_array(image, subject=subject)
    return np.mean(array, axis=3, keepdims=True, dtype=np.float32)


def _merge_planes(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    channel_3: np.ndarray,
    color_space: str,
) -> np.ndarray:
    if color_space == "rgb":
        return np.concatenate((channel_1, channel_2, channel_3), axis=3)
    if color_space == "ycbcr":
        luminance = channel_1
        blue_difference = channel_2 - 0.5
        red_difference = channel_3 - 0.5
        red = luminance + 1.403 * red_difference
        green = luminance - 0.714 * red_difference - 0.344 * blue_difference
        blue = luminance + 1.773 * blue_difference
        return np.concatenate((red, green, blue), axis=3)
    raise ValueError(f"unknown color space: {color_space}")


class ImageChannelMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_mask_polarity(
            NodeSchema(
                node_type="dinkster.image.channels.merge",
                display_name="Merge Image Channels",
                category="image/channels",
                inputs=(
                    InputSpec("channel_1", IMAGE, alpha_policy="drop"),
                    InputSpec("channel_2", IMAGE, alpha_policy="drop"),
                    InputSpec("channel_3", IMAGE, alpha_policy="drop"),
                    _combo("color_space", COLOR_SPACES, "rgb"),
                    InputSpec("alpha_mask", MASK, required=False, default=None),
                    _combo(
                        "alpha_mask_polarity",
                        ALPHA_MASK_POLARITIES,
                        "transparency",
                    ),
                    _combo(
                        "batch_policy",
                        ("singleton_broadcast", "strict"),
                        "singleton_broadcast",
                        advanced=True,
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "merge RGB",
                    "merge YCbCr",
                    "MergeImageChannels",
                    "ImageYUVToRGB",
                ),
            )
        )

    @classmethod
    def execute(
        cls,
        *,
        channel_1: object,
        channel_2: object,
        channel_3: object,
        color_space: str = "rgb",
        alpha_mask: object | None = None,
        mask_polarity: str = "transparency",
        batch_policy: str = "singleton_broadcast",
    ) -> Mapping[str, object]:
        mask_polarity = normalize_mask_polarity(mask_polarity)
        if mask_polarity not in ALPHA_MASK_POLARITIES:
            raise ValueError(f"unknown alpha mask polarity: {mask_polarity}")
        first = _plane(channel_1, "channel_1")
        second = _plane(channel_2, "channel_2")
        third = _plane(channel_3, "channel_3")
        if first.shape[1:3] != second.shape[1:3] or first.shape[1:3] != third.shape[1:3]:
            raise ValueError(
                "channel spatial dimensions must match, got "
                f"{first.shape[1:3]}, {second.shape[1:3]}, and {third.shape[1:3]}"
            )
        mask_array: np.ndarray | None = None
        if alpha_mask is not None:
            mask_array = _mask_array(alpha_mask, subject="alpha_mask")
            if mask_array.shape[1:3] != first.shape[1:3]:
                mask_array = _resize_array(
                    mask_array, int(first.shape[2]), int(first.shape[1]), "bilinear"
                )
        arrays = [("channel_1", first), ("channel_2", second), ("channel_3", third)]
        if mask_array is not None:
            arrays.append(("alpha_mask", mask_array))
        reconciled = _reconcile_batches(arrays, batch_policy)
        first, second, third = reconciled[:3]
        rgb = _merge_planes(first, second, third, color_space)
        if mask_array is not None:
            mask_array = reconciled[3]
            opacity = np.clip(mask_array, 0.0, 1.0)
            if mask_polarity == "transparency":
                opacity = 1.0 - opacity
            rgb = np.concatenate((rgb, opacity[..., None]), axis=3)
        return cls.outputs(image=np.ascontiguousarray(rgb, dtype=np.float32))


class ImageAlphaJoin(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_mask_polarity(
            NodeSchema(
                node_type="dinkster.image.alpha.join",
                display_name="Join Image with Alpha",
                category="image/channels",
                inputs=(
                    InputSpec("image", IMAGE),
                    InputSpec("alpha_mask", MASK, mask_semantic="alpha"),
                    _combo(
                        "alpha_mask_polarity",
                        ALPHA_MASK_POLARITIES,
                        "transparency",
                    ),
                    _combo(
                        "batch_policy",
                        ("singleton_broadcast", "strict", "cyclic_repeat"),
                        "singleton_broadcast",
                        advanced=True,
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True, alpha_policy="create_if_missing"),
                ),
                search_terms=("add alpha", "RGBA", "JoinImageWithAlpha"),
            )
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        alpha_mask: object,
        mask_polarity: str = "transparency",
        batch_policy: str = "singleton_broadcast",
    ) -> Mapping[str, object]:
        mask_polarity = normalize_mask_polarity(mask_polarity)
        if mask_polarity not in ALPHA_MASK_POLARITIES:
            raise ValueError(f"unknown alpha mask polarity: {mask_polarity}")
        image_array = _image_array(image)
        mask_array = _mask_array(alpha_mask, subject="alpha_mask")
        if mask_array.shape[1:3] != image_array.shape[1:3]:
            mask_array = _resize_array(
                mask_array, int(image_array.shape[2]), int(image_array.shape[1]), "bilinear"
            )
        if batch_policy == "cyclic_repeat":
            batch = max(int(image_array.shape[0]), int(mask_array.shape[0]))
            image_array = _repeat_batch(image_array, batch)
            mask_array = _repeat_batch(mask_array, batch)
        else:
            image_array, mask_array = _reconcile_batches(
                (("image", image_array), ("alpha_mask", mask_array)), batch_policy
            )
        opacity = np.clip(mask_array, 0.0, 1.0)
        if mask_polarity == "transparency":
            opacity = 1.0 - opacity
        rgb = (
            _unassociated_rgb(image_array)
            if media_semantics(image).get("alpha") == "premultiplied"
            else _rgb(image_array)
        )
        rgba = np.concatenate((rgb, opacity[..., None]), axis=3)
        return cls.outputs(
            image=annotate_image(
                copy_media_semantics(image, np.ascontiguousarray(rgba, dtype=np.float32)),
                alpha="straight",
            )
        )


class ImageAlphaPremultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.alpha.premultiply",
            display_name="Premultiply Image Alpha",
            category="image/channels",
            inputs=(InputSpec("image", IMAGE, alpha_policy="require"),),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="require"),),
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        array = _image_array(image)
        if array.shape[3] != 4:
            raise ValueError("premultiply requires an RGBA image")
        output = array.copy()
        output[..., :3] *= output[..., 3:4]
        return cls.outputs(
            image=annotate_image(copy_media_semantics(image, output), alpha="premultiplied")
        )


class ImageAlphaUnpremultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.alpha.unpremultiply",
            display_name="Unpremultiply Image Alpha",
            category="image/channels",
            inputs=(InputSpec("image", IMAGE, alpha_policy="require"),),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="require"),),
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        array = _image_array(image)
        if array.shape[3] != 4:
            raise ValueError("unpremultiply requires an RGBA image")
        output = array.copy()
        output[..., :3] = _unassociated_rgb(array)
        return cls.outputs(
            image=annotate_image(copy_media_semantics(image, output), alpha="straight")
        )


CHANNEL_NODES: tuple[type[Node], ...] = (
    ImageChannelSplit,
    ImageChannelMerge,
    ImageAlphaJoin,
    ImageAlphaPremultiply,
    ImageAlphaUnpremultiply,
)


__all__ = [
    "ALPHA_MASK_POLARITIES",
    "CHANNEL_NODES",
    "COLOR_SPACES",
    "ImageAlphaJoin",
    "ImageAlphaPremultiply",
    "ImageAlphaUnpremultiply",
    "ImageChannelMerge",
    "ImageChannelSplit",
]
