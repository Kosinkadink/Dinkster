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
IMAGE_COLOR_SPACES = ("sRGB", "HDR", "HDR PQ", "linear")
ALPHA_MASK_POLARITIES = ("coverage", "transparency")

_REC709_TO_REC2020 = np.asarray(
    (
        (0.6274038959346991, 0.3292830383778837, 0.0433130656874172),
        (0.0690972893582320, 0.9195403950754587, 0.0113623155663092),
        (0.0163914388751503, 0.0880133078772259, 0.8955952532476238),
    ),
    dtype=np.float32,
)
_REC2020_TO_REC709 = np.asarray(
    (
        (1.6604910021084338, -0.5876411387885494, -0.0728498633198846),
        (-0.1245504745215905, 1.1328998971259600, -0.0083494226043695),
        (-0.0181507633549053, -0.1005788980080076, 1.1187296613629125),
    ),
    dtype=np.float32,
)
_REC709_LUMA = np.asarray(
    (0.2126390058715103, 0.7151686787677559, 0.0721923153607337), dtype=np.float32
)
_REC2020_LUMA = np.asarray((0.2627, 0.6780, 0.0593), dtype=np.float32)
_PQ_M1, _PQ_M2 = 2610.0 / 16384.0, 2523.0 / 32.0
_PQ_C1, _PQ_C2, _PQ_C3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
_SDR_WHITE_NITS = 203.0
_HLG_PEAK_NITS = 1000.0
_HLG_GAMMA = 1.2
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991072928


def _convert_primaries(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.matmul(rgb, matrix.T, dtype=np.float32)


def _luminance(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(rgb * weights, axis=-1, keepdims=True, dtype=np.float32)


def _tone_map_luminance(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    luminance = np.maximum(_luminance(rgb, weights), 0.0)
    peak = max(float(np.max(luminance)), 1.0)
    if peak <= 1.0001:
        return rgb
    return rgb * ((1.0 + luminance / (peak * peak)) / (1.0 + luminance))


def _compress_gamut(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    luminance = np.clip(_luminance(rgb, weights), 0.0, 1.0)
    chroma = rgb - luminance
    minimum = np.min(rgb, axis=-1, keepdims=True)
    maximum = np.max(rgb, axis=-1, keepdims=True)
    tiny = np.finfo(np.float32).tiny
    upper = (1.0 - luminance) / np.maximum(maximum - luminance, tiny)
    lower = luminance / np.maximum(luminance - minimum, tiny)
    saturation = np.clip(np.minimum(upper, lower), 0.0, 1.0)
    compressed = luminance + chroma * saturation
    return np.where((minimum >= -1e-5) & (maximum <= 1.00001), rgb, compressed).clip(0.0, 1.0)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    low = rgb / 12.92
    high = np.power((np.maximum(rgb, 0.0) + 0.055) / 1.055, 2.4)
    return np.where(rgb <= 0.04045, low, high)


class ImageColorSpace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.color_space",
            display_name="Convert Image Color Space",
            category="image/color",
            inputs=(
                InputSpec("image", IMAGE),
                _combo("source", IMAGE_COLOR_SPACES, "sRGB"),
                _combo("destination", IMAGE_COLOR_SPACES, "sRGB"),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("ImageColorSpace", "sRGB", "HDR", "PQ", "HLG", "linear"),
        )

    @classmethod
    def execute(
        cls, *, image: object, source: str = "sRGB", destination: str = "sRGB"
    ) -> Mapping[str, object]:
        if source not in IMAGE_COLOR_SPACES:
            raise ValueError(f"unsupported source color space: {source}")
        if destination not in IMAGE_COLOR_SPACES:
            raise ValueError(f"unsupported destination color space: {destination}")
        array = _image_array(image)
        if source == destination:
            return cls.outputs(image=copy_media_semantics(image, np.ascontiguousarray(array)))

        rgb = np.asarray(array[..., :3], dtype=np.float32)
        if source == "sRGB":
            rgb = _convert_primaries(_srgb_to_linear(rgb), _REC709_TO_REC2020) * _SDR_WHITE_NITS
        elif source == "linear":
            rgb = _convert_primaries(rgb, _REC709_TO_REC2020) * _SDR_WHITE_NITS
        elif source == "HDR":
            low = np.square(rgb) / 3.0
            high = (np.exp((np.maximum(rgb, 0.5) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
            rgb = np.where(rgb <= 0.5, low, high)
            luminance = np.maximum(_luminance(rgb, _REC2020_LUMA), 0.0)
            rgb *= np.power(luminance, _HLG_GAMMA - 1.0) * _HLG_PEAK_NITS
        else:
            p = np.expm1(np.log(np.maximum(rgb, 0.0)) / _PQ_M2)
            numerator = np.maximum(p + (1.0 - _PQ_C1), 0.0)
            denominator = (_PQ_C2 - _PQ_C3) - _PQ_C3 * p
            rgb = np.power(numerator / denominator, 1.0 / _PQ_M1) * 10000.0

        if destination == "linear":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709)
        elif destination == "sRGB":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709)
            rgb = _compress_gamut(_tone_map_luminance(rgb, _REC709_LUMA), _REC709_LUMA)
            rgb = np.where(
                rgb <= 0.0031308,
                rgb * 12.92,
                1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
            )
        elif destination == "HDR":
            rgb /= _HLG_PEAK_NITS
            if source == "HDR PQ":
                rgb = _tone_map_luminance(rgb, _REC2020_LUMA)
            luminance = np.maximum(_luminance(rgb, _REC2020_LUMA), np.finfo(np.float32).tiny)
            rgb *= np.power(luminance, 1.0 / _HLG_GAMMA - 1.0)
            if source == "HDR PQ":
                rgb = _compress_gamut(rgb, _REC2020_LUMA)
            low = np.sqrt(3.0 * np.maximum(rgb, 0.0))
            high = _HLG_A * np.log(12.0 * np.maximum(rgb, 1.0 / 12.0) - _HLG_B) + _HLG_C
            rgb = np.where(rgb <= 1.0 / 12.0, low, high)
        else:
            p = np.power(np.maximum(rgb, 0.0) / 10000.0, _PQ_M1)
            p = ((_PQ_C1 - 1.0) + (_PQ_C2 - _PQ_C3) * p) / (1.0 + _PQ_C3 * p)
            rgb = np.exp(np.log1p(p) * _PQ_M2)

        if array.shape[3] == 4:
            rgb = np.concatenate((rgb, array[..., 3:4]), axis=3)
        return cls.outputs(
            image=copy_media_semantics(image, np.ascontiguousarray(rgb, dtype=np.float32))
        )


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
    ImageColorSpace,
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
    "IMAGE_COLOR_SPACES",
    "ImageAlphaJoin",
    "ImageAlphaPremultiply",
    "ImageAlphaUnpremultiply",
    "ImageChannelMerge",
    "ImageChannelSplit",
    "ImageColorSpace",
]
