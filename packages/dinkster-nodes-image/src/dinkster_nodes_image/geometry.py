"""Deterministic CPU image geometry over BHWC NumPy arrays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)
from PIL import Image, ImageColor

from .migration import with_resize_migrations, with_v1_migration
from .support import (
    INTERPOLATION_OPTIONS as _INTERPOLATION_OPTIONS,
)
from .support import (
    MAX_DIMENSION,
    Interpolation,
    materialized_inputs,
)
from .support import (
    RESAMPLING as _RESAMPLING,
)
from .support import (
    check_output_size as _check_output_size,
)
from .support import (
    combo_input as _combo,
)
from .support import (
    image_array as _image_array,
)
from .support import (
    mask_array as _mask_array,
)
from .support import (
    resize_array as _resize_array,
)
from .types import REGION_TYPE, Region

IMAGE_TYPE = "dinkster.image"
MASK_TYPE = "dinkster.mask"

IMAGE = TypeExpr.concrete(IMAGE_TYPE)
MASK = TypeExpr.concrete(MASK_TYPE)
RESIZE_INPUT = TypeExpr.variable("input_type", (IMAGE_TYPE, MASK_TYPE))
IMAGE_OR_MASK = TypeExpr.union(IMAGE_TYPE, MASK_TYPE)
REGION = TypeExpr.concrete(REGION_TYPE)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
INT_LIST = TypeExpr.list_of(INT)

ResizeMode = Literal["stretch", "fit", "fill", "pad"]
PaddingStyle = Literal["constant", "edge_average", "edge_pixel", "blurred_background"]
Divisibility = Literal["none", "nearest", "crop", "pad"]
Rounding = Literal["floor", "round", "ceil", "expand"]

_TRANSFORM_INTERPOLATION_OPTIONS = ("nearest", "bilinear", "bicubic")


def _image_or_mask_array(value: object, *, subject: str) -> tuple[np.ndarray, bool]:
    is_mask = np.asarray(value).ndim == 3
    array = _mask_array(value, subject=subject) if is_mask else _image_array(value, subject=subject)
    return array, is_mask


def _scaled_size(width: int, height: int, target_width: int, target_height: int) -> tuple[int, int]:
    scale = min(target_width / width, target_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _validate_resize_mode(mode: str, interpolation: str) -> None:
    if mode not in ("stretch", "fit", "fill", "pad"):
        raise ValueError(f"unknown resize mode: {mode}")
    if interpolation not in _INTERPOLATION_OPTIONS:
        raise ValueError(f"unknown interpolation mode: {interpolation}")


def _resize_content(
    array: np.ndarray,
    width: int,
    height: int,
    mode: ResizeMode,
    interpolation: Interpolation,
    anchor: str,
    fit_rounding: str = "round",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    _validate_resize_mode(mode, interpolation)
    source_height, source_width = int(array.shape[1]), int(array.shape[2])
    if mode == "stretch":
        return _resize_array(array, width, height, interpolation), (0, 0, 0, 0)
    if mode == "fill":
        return _resize_fill_at(array, width, height, interpolation, anchor), (0, 0, 0, 0)
    scaled_width, scaled_height = _scaled_size(source_width, source_height, width, height)
    if fit_rounding == "floor":
        scale = min(width / source_width, height / source_height)
        scaled_width, scaled_height = int(source_width * scale), int(source_height * scale)
    resized = _resize_array(array, scaled_width, scaled_height, interpolation)
    if mode == "fit":
        return resized, (0, 0, 0, 0)
    return resized, _placement_padding(width, height, scaled_width, scaled_height, anchor)


def _target_dimensions(
    source_width: int,
    source_height: int,
    target: str,
    width: int,
    height: int,
    size: int,
    factor: float,
    megapixels: float,
    resolution_steps: int,
) -> tuple[int, int]:
    if resolution_steps < 1:
        raise ValueError("resolution_steps must be positive")
    if target == "dimensions":
        if width < 0 or height < 0:
            raise ValueError(f"target dimensions must be non-negative, got {(width, height)}")
        if width == 0 and height == 0:
            result = source_width, source_height
        elif width == 0:
            result = max(1, round(source_width * height / source_height)), height
        elif height == 0:
            result = width, max(1, round(source_height * width / source_width))
        else:
            result = width, height
    elif target == "width":
        if width < 0:
            raise ValueError(f"width must be non-negative, got {width}")
        result = (
            (source_width, source_height)
            if width == 0
            else (width, max(1, round(source_height * width / source_width)))
        )
    elif target == "height":
        if height < 0:
            raise ValueError(f"height must be non-negative, got {height}")
        result = (
            (source_width, source_height)
            if height == 0
            else (max(1, round(source_width * height / source_height)), height)
        )
    elif target == "longest":
        if size < 1:
            raise ValueError(f"size must be positive, got {size}")
        scale = size / max(source_width, source_height)
        result = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    elif target == "shortest":
        if size < 1:
            raise ValueError(f"size must be positive, got {size}")
        scale = size / min(source_width, source_height)
        result = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    elif target == "factor":
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"factor must be finite and positive, got {factor}")
        result = max(1, round(source_width * factor)), max(1, round(source_height * factor))
    elif target == "total_pixels":
        if not math.isfinite(megapixels) or megapixels <= 0:
            raise ValueError(f"megapixels must be finite and positive, got {megapixels}")
        scale = math.sqrt(megapixels * 1024 * 1024 / (source_width * source_height))
        result = (
            max(
                resolution_steps, round(source_width * scale / resolution_steps) * resolution_steps
            ),
            max(
                resolution_steps,
                round(source_height * scale / resolution_steps) * resolution_steps,
            ),
        )
    else:
        raise ValueError(f"unknown resize target: {target}")
    if result[0] < 1 or result[1] < 1:
        raise ValueError(f"target dimensions must be positive, got {result}")
    return result


def _multiple_cover_dimensions(
    source_width: int,
    source_height: int,
    multiple_of: int,
) -> tuple[int, int] | None:
    if multiple_of <= 1:
        return None
    final_width = _floor_multiple(source_width, multiple_of)
    final_height = _floor_multiple(source_height, multiple_of)
    if final_width == 0 or final_height == 0:
        return None
    if (final_width, final_height) == (source_width, source_height):
        return None
    width_scale = final_width / source_width
    height_scale = final_height / source_height
    if width_scale >= height_scale:
        return final_width, max(final_height, math.ceil(source_height * width_scale))
    return max(final_width, math.ceil(source_width * height_scale)), final_height


def _floor_multiple(value: int, multiple: int) -> int:
    return value - value % multiple if multiple > 1 else value


def _placement_padding(
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    anchor: str,
) -> tuple[int, int, int, int]:
    if anchor not in ("center", "top", "bottom", "left", "right"):
        raise ValueError(f"unknown resize anchor: {anchor}")
    if anchor in ("center", "top", "bottom"):
        left = (width - resized_width) // 2
        right = width - resized_width - left
    elif anchor == "left":
        left, right = 0, width - resized_width
    elif anchor == "right":
        left, right = width - resized_width, 0
    else:
        raise ValueError(f"unknown resize anchor: {anchor}")
    if anchor in ("center", "left", "right"):
        top = (height - resized_height) // 2
        bottom = height - resized_height - top
    elif anchor == "top":
        top, bottom = 0, height - resized_height
    else:
        top, bottom = height - resized_height, 0
    return left, right, top, bottom


def _padding_color(value: str, channels: int) -> np.ndarray:
    color = [0, 0, 0]
    if "," in value:
        try:
            components = [float(component.strip()) for component in value.split(",")]
            if all(0.0 <= component <= 1.0 for component in components):
                color = [int(component * 255) for component in components]
            else:
                color = [int(component) for component in components]
        except ValueError:
            pass
    elif value.startswith("#") or (
        value.lstrip("#").isalnum() and not value.lstrip("#").replace(".", "", 1).isdigit()
    ):
        payload = value.lstrip("#")
        if len(payload) in (6, 8) and all(
            character in "0123456789abcdefABCDEF" for character in payload
        ):
            color = [int(payload[index : index + 2], 16) for index in range(0, len(payload), 2)]
        else:
            try:
                color = list(ImageColor.getrgb(value))
            except ValueError:
                pass
    else:
        try:
            component = float(value.strip())
            channel = int(component * 255) if 0.0 <= component <= 1.0 else int(component)
            color = [channel, channel, channel]
        except ValueError:
            pass
    color_array = np.clip(np.asarray(color, dtype=np.float32), 0.0, 255.0) / np.float32(255.0)
    if channels == 1 and color_array.size >= 3:
        color_array = np.asarray(
            [0.2126 * color_array[0] + 0.7152 * color_array[1] + 0.0722 * color_array[2]],
            dtype=np.float32,
        )
    elif color_array.size == 1:
        color_array = np.repeat(color_array, channels)
    if channels == 4 and color_array.size == 3:
        color_array = np.concatenate((color_array, np.ones(1, dtype=np.float32)))
    if color_array.size != channels:
        raise ValueError(
            f"padding color has {color_array.size} channels for a {channels}-channel image"
        )
    return color_array


def _gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3.0 * sigma))
    positions = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel = np.asarray(kernel / kernel.sum(), dtype=np.float32)

    horizontal = np.pad(array, ((0, 0), (0, 0), (radius, radius), (0, 0)))
    horizontal_output = np.zeros_like(array)
    for offset, weight in enumerate(kernel):
        horizontal_output += horizontal[:, :, offset : offset + array.shape[2], :] * weight

    vertical = np.pad(horizontal_output, ((0, 0), (radius, radius), (0, 0), (0, 0)))
    output = np.zeros_like(array)
    for offset, weight in enumerate(kernel):
        output += vertical[:, offset : offset + array.shape[1], :, :] * weight
    return output


def _pad_array(
    array: np.ndarray,
    padding: tuple[int, int, int, int],
    style: PaddingStyle,
    pad_value: float,
    pad_color: str,
) -> np.ndarray:
    left, right, top, bottom = padding
    if min(padding) < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    if not any(padding):
        return array
    batch, height, width, channels = array.shape
    padded_height, padded_width = height + top + bottom, width + left + right
    _check_output_size((batch, padded_height, padded_width, channels))

    if style == "blurred_background":
        scale = max(padded_width / width, padded_height / height)
        background_width = max(1, round(width * scale))
        background_height = max(1, round(height * scale))
        output = _resize_array(array, background_width, background_height, "bilinear")
        crop_left = max(0, (background_width - padded_width) // 2)
        crop_top = max(0, (background_height - padded_height) // 2)
        output = output[
            :,
            crop_top : crop_top + padded_height,
            crop_left : crop_left + padded_width,
            :,
        ]
        missing_height = padded_height - int(output.shape[1])
        missing_width = padded_width - int(output.shape[2])
        if missing_height or missing_width:
            fix_top = max(0, missing_height // 2)
            fix_left = max(0, missing_width // 2)
            output = np.pad(
                output,
                (
                    (0, 0),
                    (fix_top, max(0, missing_height - fix_top)),
                    (fix_left, max(0, missing_width - fix_left)),
                    (0, 0),
                ),
                mode="edge",
            )
        sigma = max(1.0, 0.006 * min(padded_height, padded_width))
        output = _gaussian_blur(output, sigma)
        if channels >= 3:
            red, green, blue = output[..., 0:1], output[..., 1:2], output[..., 2:3]
            luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            rgb = np.concatenate((red, green, blue), axis=3)
            output[..., :3] = rgb * 0.8 + np.repeat(luma, 3, axis=3) * 0.2
        output = np.asarray(np.clip(output * 0.35, 0.0, 1.0), dtype=np.float32)
    elif style == "edge_pixel":
        output = np.pad(
            array,
            ((0, 0), (top, bottom), (left, right), (0, 0)),
            mode="edge",
        )
    else:
        output = np.zeros(
            (batch, padded_height, padded_width, channels),
            dtype=np.float32,
        )
        if style == "edge_average":
            output[:, :top, :, :] = array[:, 0, :, :].mean(axis=1)[:, None, None, :]
            output[:, top + height :, :, :] = array[:, -1, :, :].mean(axis=1)[:, None, None, :]
            output[:, :, :left, :] = array[:, :, 0, :].mean(axis=1)[:, None, None, :]
            output[:, :, left + width :, :] = array[:, :, -1, :].mean(axis=1)[:, None, None, :]
        elif style == "constant":
            output[...] = (
                _padding_color(pad_color, channels) if pad_color else np.float32(pad_value)
            )
        else:
            raise ValueError(f"unknown padding style: {style}")

    output[:, top : top + height, left : left + width, :] = array
    return np.ascontiguousarray(output)


def _resize_fill_at(
    array: np.ndarray,
    width: int,
    height: int,
    interpolation: Interpolation,
    anchor: str,
) -> np.ndarray:
    source_height, source_width = int(array.shape[1]), int(array.shape[2])
    source_aspect = source_width / source_height
    target_aspect = width / height
    crop_width, crop_height = source_width, source_height
    if source_aspect > target_aspect:
        crop_width = max(1, round(source_height * target_aspect))
    elif source_aspect < target_aspect:
        crop_height = max(1, round(source_width / target_aspect))
    if anchor in ("center", "top", "bottom"):
        left = (source_width - crop_width) // 2
    elif anchor == "left":
        left = 0
    elif anchor == "right":
        left = source_width - crop_width
    else:
        raise ValueError(f"unknown resize anchor: {anchor}")
    if anchor in ("center", "left", "right"):
        top = (source_height - crop_height) // 2
    elif anchor == "top":
        top = 0
    else:
        top = source_height - crop_height
    cropped = array[:, top : top + crop_height, left : left + crop_width, ...]
    return _resize_array(cropped, width, height, interpolation)


def _pad_companion_mask(
    mask: np.ndarray | None,
    padding: tuple[int, int, int, int],
    *,
    batch: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    if not any(padding):
        return mask
    left, right, top, bottom = padding
    output = np.ones((batch, height + top + bottom, width + left + right), dtype=np.float32)
    output[:, top : top + height, left : left + width] = 0.0 if mask is None else mask
    return output


def _crop_array_at(array: np.ndarray, width: int, height: int, anchor: str) -> np.ndarray:
    source_height, source_width = int(array.shape[1]), int(array.shape[2])
    left, _right, top, _bottom = _placement_padding(
        source_width,
        source_height,
        width,
        height,
        anchor,
    )
    return np.ascontiguousarray(array[:, top : top + height, left : left + width, ...])


def _resize_condition_allows(
    apply: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> bool:
    conditions = {
        "always": True,
        "only_if_bigger": source_width > target_width or source_height > target_height,
        "only_if_smaller": source_width < target_width or source_height < target_height,
        "only_if_bigger_area": source_width * source_height > target_width * target_height,
        "only_if_smaller_area": source_width * source_height < target_width * target_height,
    }
    if apply not in conditions:
        raise ValueError(f"unknown resize apply condition: {apply}")
    return conditions[apply]


def _divisible_dimensions(
    width: int,
    height: int,
    multiple_of: int,
    divisibility: Divisibility,
) -> tuple[int, int]:
    if multiple_of < 0:
        raise ValueError(f"multiple_of must be non-negative, got {multiple_of}")
    if divisibility == "none" or multiple_of <= 1:
        return width, height
    if divisibility == "nearest":
        return (
            max(1, round(width / multiple_of) * multiple_of),
            max(1, round(height / multiple_of) * multiple_of),
        )
    if divisibility == "crop":
        return max(1, _floor_multiple(width, multiple_of)), max(
            1, _floor_multiple(height, multiple_of)
        )
    if divisibility == "pad":
        return (
            math.ceil(width / multiple_of) * multiple_of,
            math.ceil(height / multiple_of) * multiple_of,
        )
    raise ValueError(f"unknown divisibility finalization: {divisibility}")


def _resize_with_intent(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    target_width: int,
    target_height: int,
    mode: ResizeMode,
    interpolation: Interpolation,
    apply: str,
    mode_anchor: str,
    mode_padding: PaddingStyle,
    pad_value: float,
    pad_color: str,
    divisibility: Divisibility,
    multiple_of: int,
    final_anchor: str,
    final_padding: PaddingStyle,
    final_pad_value: float,
    final_pad_color: str,
    fit_rounding: str = "round",
) -> tuple[np.ndarray, np.ndarray | None]:
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    _validate_resize_mode(mode, interpolation)
    padding_styles = ("constant", "edge_average", "edge_pixel", "blurred_background")
    if mode == "pad" and mode_padding not in padding_styles:
        raise ValueError(f"unknown padding style: {mode_padding}")
    if divisibility == "pad" and final_padding not in padding_styles:
        raise ValueError(f"unknown padding style: {final_padding}")
    if _resize_condition_allows(
        apply,
        source_width,
        source_height,
        target_width,
        target_height,
    ):
        output, padding = _resize_content(
            image,
            target_width,
            target_height,
            mode,
            interpolation,
            mode_anchor,
            fit_rounding,
        )
        output_mask = (
            None
            if mask is None
            else _resize_content(
                mask,
                target_width,
                target_height,
                mode,
                interpolation,
                mode_anchor,
                fit_rounding,
            )[0]
        )
        content_height, content_width = int(output.shape[1]), int(output.shape[2])
        output_mask = _pad_companion_mask(
            output_mask,
            padding,
            batch=int(output.shape[0]),
            height=content_height,
            width=content_width,
        )
        output = _pad_array(output, padding, mode_padding, pad_value, pad_color)
    else:
        output, output_mask = image, mask

    output_height, output_width = int(output.shape[1]), int(output.shape[2])
    final_width, final_height = _divisible_dimensions(
        output_width,
        output_height,
        multiple_of,
        divisibility,
    )
    if (final_width, final_height) == (output_width, output_height):
        return np.ascontiguousarray(output), (
            None if output_mask is None else np.ascontiguousarray(output_mask)
        )
    if divisibility == "nearest":
        output = _resize_array(output, final_width, final_height, interpolation)
        if output_mask is not None:
            output_mask = _resize_array(output_mask, final_width, final_height, interpolation)
    elif divisibility == "crop":
        output = _crop_array_at(output, final_width, final_height, final_anchor)
        if output_mask is not None:
            output_mask = _crop_array_at(output_mask, final_width, final_height, final_anchor)
    else:
        padding = _placement_padding(
            final_width,
            final_height,
            output_width,
            output_height,
            final_anchor,
        )
        output_mask = _pad_companion_mask(
            output_mask,
            padding,
            batch=int(output.shape[0]),
            height=output_height,
            width=output_width,
        )
        output = _pad_array(
            output,
            padding,
            final_padding,
            final_pad_value,
            final_pad_color,
        )
    return np.ascontiguousarray(output), (
        None if output_mask is None else np.ascontiguousarray(output_mask)
    )


class ImageResize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        dynamic_inputs = (
            InputSpec(
                "width",
                INT,
                required=False,
                default=512,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
                display_name="Width",
            ),
            InputSpec(
                "height",
                INT,
                required=False,
                default=512,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
                display_name="Height",
            ),
            InputSpec(
                "size",
                INT,
                required=False,
                default=512,
                widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1),
                display_name="Size",
            ),
            InputSpec(
                "factor",
                FLOAT,
                required=False,
                default=1.0,
                widget=NumberWidget(min=0.01, max=64.0, step=0.01),
                display_name="Scale factor",
            ),
            InputSpec(
                "megapixels",
                FLOAT,
                required=False,
                default=1.0,
                widget=NumberWidget(min=0.001, max=268.0, step=0.01),
                display_name="Megapixels",
            ),
            InputSpec(
                "multiple_of",
                INT,
                required=False,
                default=8,
                widget=NumberWidget(min=0, max=1024, step=1),
                display_name="Multiple of",
            ),
            InputSpec(
                "resolution_steps",
                INT,
                required=False,
                default=1,
                widget=NumberWidget(min=1, max=1024, step=1),
                advanced=True,
                display_name="Resolution steps",
            ),
            InputSpec(
                "pad_value",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                display_name="Value",
            ),
            InputSpec(
                "pad_color",
                STRING,
                required=False,
                default="",
                widget=StringWidget(multiline=False),
                display_name="Color override",
                advanced=True,
            ),
            InputSpec(
                "reference",
                IMAGE_OR_MASK,
                required=False,
                default=None,
                display_name="Reference image or mask",
            ),
            _combo(
                "mode_anchor",
                ("center", "top", "bottom", "left", "right"),
                "center",
                display_name="Anchor",
            ),
            InputSpec(
                "final_pad_value",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                display_name="Value",
            ),
            InputSpec(
                "final_pad_color",
                STRING,
                required=False,
                default="",
                widget=StringWidget(multiline=False),
                display_name="Color override",
                advanced=True,
            ),
            _combo(
                "final_anchor",
                ("center", "top", "bottom", "left", "right"),
                "center",
                display_name="Final anchor",
            ),
        )
        by_id = {item.id: item for item in dynamic_inputs}

        def option(key: str, *ids: str) -> DynamicComboOption:
            return DynamicComboOption(key, tuple(by_id[input_id] for input_id in ids))

        target = DynamicComboSpec(
            "target",
            (
                option("dimensions", "width", "height"),
                option("width", "width"),
                option("height", "height"),
                option("longest", "size"),
                option("shortest", "size"),
                option("factor", "factor"),
                option("total_pixels", "megapixels", "resolution_steps"),
                option("match", "reference"),
                option("multiple_cover", "multiple_of"),
            ),
            default="dimensions",
            display_name="Target",
        )
        mode_padding = DynamicComboSpec(
            "mode_padding",
            (
                option("constant", "pad_value", "pad_color"),
                option("edge_average"),
                option("edge_pixel"),
                option("blurred_background"),
            ),
            default="constant",
            display_name="Padding style",
        )
        mode = DynamicComboSpec(
            "mode",
            (
                option("stretch"),
                option("fit"),
                option("fill", "mode_anchor"),
                DynamicComboOption("pad", (by_id["mode_anchor"], mode_padding)),
            ),
            default="stretch",
            display_name="Mode",
        )
        final_padding = DynamicComboSpec(
            "final_padding",
            (
                option("constant", "final_pad_value", "final_pad_color"),
                option("edge_average"),
                option("edge_pixel"),
                option("blurred_background"),
            ),
            default="constant",
            display_name="Final pad",
        )
        divisibility = DynamicComboSpec(
            "divisibility",
            (
                option("none"),
                option("nearest", "multiple_of"),
                option("crop", "multiple_of", "final_anchor"),
                DynamicComboOption(
                    "pad",
                    (by_id["multiple_of"], by_id["final_anchor"], final_padding),
                ),
            ),
            default="none",
            display_name="Divisibility finalization",
        )

        return with_resize_migrations(
            NodeSchema(
                node_type="dinkster.image.resize",
                version=4,
                display_name="Resize Image/Mask",
                category="image/geometry",
                inputs=(
                    InputSpec("image", RESIZE_INPUT),
                    InputSpec("mask", MASK, required=False, default=None),
                    _combo("fit_rounding", ("round", "floor"), "round", advanced=True),
                    _combo(
                        "apply",
                        (
                            "always",
                            "only_if_bigger",
                            "only_if_smaller",
                            "only_if_bigger_area",
                            "only_if_smaller_area",
                        ),
                        "always",
                        advanced=True,
                        display_name="Apply condition",
                    ),
                    _combo(
                        "interpolation",
                        _INTERPOLATION_OPTIONS,
                        "bilinear",
                        advanced=True,
                        display_name="Interpolation",
                    ),
                ),
                combos=(target, mode, divisibility),
                outputs=(
                    OutputSpec("image", RESIZE_INPUT, preview=True),
                    OutputSpec("mask", MASK, optional=True, preview=True),
                ),
                search_terms=(
                    "scale image",
                    "fit image",
                    "fill image",
                    "resize and pad",
                    "ImageScale",
                    "ImageScaleBy",
                    "ImageScaleToTotalPixels",
                    "ImageResizeKJ",
                    "ImageResizeKJv2",
                    "ImageResize+",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        mask: object | None = None,
        apply: str = "always",
        interpolation: str = "bilinear",
        target: str = "dimensions",
        width: int = 512,
        height: int = 512,
        size: int = 512,
        factor: float = 1.0,
        megapixels: float = 1.0,
        resolution_steps: int = 1,
        reference: object | None = None,
        mode: str = "stretch",
        mode_anchor: str = "center",
        mode_padding: str = "constant",
        pad_value: float = 0.0,
        pad_color: str = "",
        divisibility: str = "none",
        multiple_of: int = 8,
        final_anchor: str = "center",
        final_padding: str = "constant",
        final_pad_value: float = 0.0,
        final_pad_color: str = "",
        fit_rounding: str = "round",
    ) -> Mapping[str, object]:
        if fit_rounding not in ("round", "floor"):
            raise ValueError(f"unknown fit rounding: {fit_rounding}")
        if not math.isfinite(pad_value) or not math.isfinite(final_pad_value):
            raise ValueError("padding values must be finite")
        image_array, input_is_mask = _image_or_mask_array(image, subject="image or mask")
        mask_array = None if mask is None else _mask_array(mask)
        if mask_array is not None:
            image_batch = int(image_array.shape[0])
            if mask_array.shape[0] == 1 and image_batch > 1:
                mask_array = np.repeat(mask_array, image_batch, axis=0)
            elif mask_array.shape[0] != image_batch:
                raise ValueError(
                    "mask batch must match image or be one, got "
                    f"{mask_array.shape[0]} and {image_batch}"
                )
            if mask_array.shape[1:3] != image_array.shape[1:3]:
                mask_array = _resize_array(
                    mask_array,
                    int(image_array.shape[2]),
                    int(image_array.shape[1]),
                    "bilinear",
                )
        if target == "match":
            if reference is None:
                raise ValueError("match resize requires reference")
            reference_array, _reference_is_mask = _image_or_mask_array(
                reference,
                subject="reference image or mask",
            )
            width, height = int(reference_array.shape[2]), int(reference_array.shape[1])
            target = "dimensions"
        if target == "multiple_cover":
            multiple_cover = _multiple_cover_dimensions(
                int(image_array.shape[2]),
                int(image_array.shape[1]),
                multiple_of,
            )
            if multiple_cover is None:
                return cls.outputs(
                    image=image_array,
                    mask=ABSENT if mask_array is None else mask_array,
                )
            target = "dimensions"
            width, height = multiple_cover
            mode = "stretch"
            divisibility = "crop"
            final_anchor = "center"
        target_width, target_height = _target_dimensions(
            int(image_array.shape[2]),
            int(image_array.shape[1]),
            target,
            width,
            height,
            size,
            factor,
            megapixels,
            resolution_steps,
        )
        one_side_noop = (target == "width" and width == 0) or (target == "height" and height == 0)
        if one_side_noop:
            apply = "only_if_bigger_area"
        resized, resized_mask = _resize_with_intent(
            image_array[..., None] if input_is_mask else image_array,
            mask_array,
            target_width=target_width,
            target_height=target_height,
            mode=cast("ResizeMode", mode),
            interpolation=cast("Interpolation", interpolation),
            apply=apply,
            mode_anchor=mode_anchor,
            mode_padding=cast("PaddingStyle", mode_padding),
            pad_value=pad_value,
            pad_color=pad_color,
            divisibility=cast("Divisibility", divisibility),
            multiple_of=multiple_of,
            final_anchor=final_anchor,
            final_padding=cast("PaddingStyle", final_padding),
            final_pad_value=final_pad_value,
            final_pad_color=final_pad_color,
            fit_rounding=fit_rounding,
        )
        return cls.outputs(
            image=resized[..., 0] if input_is_mask else resized,
            mask=ABSENT if resized_mask is None else resized_mask,
        )


def _pad_for_outpaint(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    feathering: int,
) -> tuple[np.ndarray, np.ndarray]:
    if min(left, top, right, bottom, feathering) < 0:
        raise ValueError("padding and feathering must be non-negative")
    batch, height, width, channels = image.shape
    output_shape = (batch, height + top + bottom, width + left + right, channels)
    _check_output_size(output_shape)
    output = np.full(output_shape, 0.5, dtype=np.float32)
    output[:, top : top + height, left : left + width, :] = image

    mask = np.ones((1, output_shape[1], output_shape[2]), dtype=np.float32)
    inner = np.zeros((height, width), dtype=np.float32)
    if feathering > 0 and feathering * 2 < height and feathering * 2 < width:
        rows = np.arange(height, dtype=np.int64)[:, None]
        columns = np.arange(width, dtype=np.int64)[None, :]
        distances = np.minimum.reduce(
            (
                np.broadcast_to(rows if top else height, (height, width)),
                np.broadcast_to(height - rows if bottom else height, (height, width)),
                np.broadcast_to(columns if left else width, (height, width)),
                np.broadcast_to(width - columns if right else width, (height, width)),
            )
        )
        edge = distances < feathering
        inner[edge] = np.square((feathering - distances[edge]) / feathering)
    mask[:, top : top + height, left : left + width] = inner
    return output, mask


def _transform_planes(
    image: np.ndarray,
    operation: str,
    *,
    angle: float,
    steps: int,
    x: float,
    y: float,
    units: str,
    expand: bool,
    interpolation: Interpolation,
    fill: float,
) -> np.ndarray:
    if operation == "rotate_90":
        return np.ascontiguousarray(np.rot90(image, k=-(steps % 4), axes=(1, 2)))
    if operation == "flip_horizontal":
        return np.ascontiguousarray(image[:, :, ::-1, :])
    if operation == "flip_vertical":
        return np.ascontiguousarray(image[:, ::-1, :, :])
    if operation not in ("rotate", "translate", "shear"):
        raise ValueError(f"unknown image transform: {operation}")
    if interpolation not in _TRANSFORM_INTERPOLATION_OPTIONS:
        raise ValueError(f"unknown transform interpolation mode: {interpolation}")
    batch, height, width, channels = image.shape
    resampling = _RESAMPLING[interpolation]
    dx = x * width if units == "fraction" else x
    dy = y * height if units == "fraction" else y
    if units not in ("pixels", "fraction"):
        raise ValueError(f"unknown transform units: {units}")
    transformed: list[np.ndarray] = []
    for batch_index in range(batch):
        planes: list[np.ndarray] = []
        for channel_index in range(channels):
            plane = Image.fromarray(image[batch_index, :, :, channel_index], mode="F")
            if operation == "rotate":
                result = plane.rotate(
                    -angle,
                    resample=resampling,
                    expand=expand,
                    fillcolor=float(fill),
                )
            else:
                if operation == "translate":
                    matrix = (1.0, 0.0, -dx, 0.0, 1.0, -dy)
                else:
                    shear_x = dx / height
                    shear_y = dy / width
                    matrix = (
                        1.0,
                        -shear_x,
                        shear_x * (height - 1) / 2,
                        -shear_y,
                        1.0,
                        shear_y * (width - 1) / 2,
                    )
                result = plane.transform(
                    (width, height),
                    Image.Transform.AFFINE,
                    matrix,
                    resample=resampling,
                    fillcolor=float(fill),
                )
            planes.append(np.asarray(result, dtype=np.float32))
        transformed.append(np.stack(planes, axis=2))
    output = np.stack(transformed, axis=0).astype(np.float32, copy=False)
    _check_output_size(cast("tuple[int, ...]", output.shape))
    return np.ascontiguousarray(output)


class ImageTransform(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        dynamic_inputs = (
            InputSpec(
                "angle",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=-360.0, max=360.0, step=0.1),
            ),
            InputSpec(
                "steps",
                INT,
                required=False,
                default=1,
                widget=NumberWidget(min=-4, max=4, step=1),
            ),
            InputSpec(
                "x",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(step=0.01),
            ),
            InputSpec(
                "y",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(step=0.01),
            ),
            _combo("units", ("pixels", "fraction"), "pixels"),
            _combo("expand", ("false", "true"), "false"),
            _combo(
                "interpolation",
                _TRANSFORM_INTERPOLATION_OPTIONS,
                "bilinear",
                advanced=True,
            ),
            InputSpec(
                "fill",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                advanced=True,
            ),
            InputSpec(
                "left",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
            ),
            InputSpec(
                "top",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
            ),
            InputSpec(
                "right",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
            ),
            InputSpec(
                "bottom",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
            ),
            InputSpec(
                "feathering",
                INT,
                required=False,
                default=40,
                widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
                advanced=True,
            ),
        )
        by_id = {item.id: item for item in dynamic_inputs}

        def option(key: str, *ids: str) -> DynamicComboOption:
            return DynamicComboOption(key, tuple(by_id[input_id] for input_id in ids))

        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.transform",
                version=2,
                display_name="Transform Image",
                category="image/geometry",
                inputs=(InputSpec("image", IMAGE),),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        (
                            option("rotate_90", "steps"),
                            option("rotate", "angle", "expand", "interpolation", "fill"),
                            option("flip_horizontal"),
                            option("flip_vertical"),
                            option("translate", "x", "y", "units", "interpolation", "fill"),
                            option("shear", "x", "y", "units", "interpolation", "fill"),
                            option("pad", "left", "top", "right", "bottom", "feathering"),
                        ),
                        default="rotate_90",
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("mask", MASK, optional=True, preview=True),
                ),
                search_terms=(
                    "rotate image",
                    "flip image",
                    "translate image",
                    "shear image",
                    "ImageRotate",
                    "ImageFlip",
                    "ImageTransformKJ",
                    "Transform Image (mtb)",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        operation: str = "rotate_90",
        angle: float = 0.0,
        steps: int = 1,
        x: float = 0.0,
        y: float = 0.0,
        units: str = "pixels",
        expand: str = "false",
        interpolation: str = "bilinear",
        fill: float = 0.0,
        left: int = 0,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
        feathering: int = 40,
    ) -> Mapping[str, object]:
        if not all(math.isfinite(value) for value in (angle, x, y, fill)):
            raise ValueError("transform parameters must be finite")
        image_array = _image_array(image)
        if operation == "pad":
            output, mask = _pad_for_outpaint(
                image_array,
                left,
                top,
                right,
                bottom,
                feathering,
            )
            return cls.outputs(image=output, mask=mask)
        output = _transform_planes(
            image_array,
            operation,
            angle=angle,
            steps=steps,
            x=x,
            y=y,
            units=units,
            expand=expand == "true",
            interpolation=cast("Interpolation", interpolation),
            fill=fill,
        )
        return cls.outputs(image=output, mask=ABSENT)


class MakeRegion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.region.make",
            display_name="Create Region",
            category="image/region",
            inputs=(
                InputSpec("x", FLOAT, required=False, default=0.0, widget=NumberWidget(step=1.0)),
                InputSpec("y", FLOAT, required=False, default=0.0, widget=NumberWidget(step=1.0)),
                InputSpec(
                    "width",
                    FLOAT,
                    required=False,
                    default=512.0,
                    widget=NumberWidget(min=0.0, step=1.0),
                ),
                InputSpec(
                    "height",
                    FLOAT,
                    required=False,
                    default=512.0,
                    widget=NumberWidget(min=0.0, step=1.0),
                ),
            ),
            outputs=(OutputSpec("region", REGION),),
            search_terms=("bounding box", "bbox", "rectangle"),
        )

    @classmethod
    def execute(
        cls, *, x: float = 0.0, y: float = 0.0, width: float = 512.0, height: float = 512.0
    ) -> Mapping[str, object]:
        return cls.outputs(region=Region(x=x, y=y, width=width, height=height))


class RegionInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.region.info",
            display_name="Region Information",
            category="image/region",
            inputs=(
                InputSpec("region", REGION),
                _combo(
                    "integer_rounding",
                    ("floor", "round", "ceil", "expand"),
                    "expand",
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec("region", REGION),
                OutputSpec("x", FLOAT),
                OutputSpec("y", FLOAT),
                OutputSpec("width", FLOAT),
                OutputSpec("height", FLOAT),
                OutputSpec("integer_x", INT),
                OutputSpec("integer_y", INT),
                OutputSpec("integer_width", INT),
                OutputSpec("integer_height", INT),
            ),
            search_terms=("bounding box coordinates", "bbox info", "region size"),
        )

    @classmethod
    def execute(cls, *, region: Region, integer_rounding: str = "expand") -> Mapping[str, object]:
        left, top, right, bottom = _integer_region(
            region,
            cast("Rounding", integer_rounding),
            0,
        )
        return cls.outputs(
            region=region,
            x=float(region.x),
            y=float(region.y),
            width=float(region.width),
            height=float(region.height),
            integer_x=left,
            integer_y=top,
            integer_width=right - left,
            integer_height=bottom - top,
        )


def _integer_region(region: Region, rounding: Rounding, padding: int) -> tuple[int, int, int, int]:
    left = region.x - padding
    top = region.y - padding
    right = region.right + padding
    bottom = region.bottom + padding
    if rounding == "expand":
        return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
    functions = {"floor": math.floor, "round": round, "ceil": math.ceil}
    function = functions.get(rounding)
    if function is None:
        raise ValueError(f"unknown region rounding mode: {rounding}")
    return function(left), function(top), function(right), function(bottom)


def _region_from_mask(mask: np.ndarray, threshold: float) -> Region:
    if not math.isfinite(threshold):
        raise ValueError(f"mask_threshold must be finite, got {threshold}")
    selected = np.any(mask > threshold, axis=0)
    rows, columns = np.nonzero(selected)
    if rows.size == 0:
        raise ValueError("mask does not contain any selected pixels")
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return Region(x=left, y=top, width=right - left, height=bottom - top)


def _crop_spatial(
    source: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    outside: str,
    fill: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    source_height, source_width = int(source.shape[1]), int(source.shape[2])
    if outside == "comfy":
        width, height = right - left, bottom - top
        left, top = min(left, source_width - 1), min(top, source_height - 1)
        right, bottom = left + width, top + height
        output = np.ascontiguousarray(source[:, top:bottom, left:right, ...])
    elif outside == "clip":
        left, top = max(0, left), max(0, top)
        right, bottom = min(source_width, right), min(source_height, bottom)
        if right <= left or bottom <= top:
            raise ValueError("crop region does not intersect the image")
        output = np.ascontiguousarray(source[:, top:bottom, left:right, ...])
    elif outside == "pad":
        output_width, output_height = right - left, bottom - top
        output_shape = (int(source.shape[0]), output_height, output_width, *source.shape[3:])
        checked_shape = output_shape if source.ndim == 4 else (*output_shape, 1)
        _check_output_size(cast("tuple[int, ...]", checked_shape))
        output = np.full(output_shape, fill, dtype=np.float32)
        source_left, source_top = max(0, left), max(0, top)
        source_right, source_bottom = min(source_width, right), min(source_height, bottom)
        if source_right > source_left and source_bottom > source_top:
            output[
                :,
                source_top - top : source_bottom - top,
                source_left - left : source_right - left,
                ...,
            ] = source[:, source_top:source_bottom, source_left:source_right, ...]
    else:
        raise ValueError(f"unknown outside-image policy: {outside}")
    return output, (left, top, right, bottom)


class ImageCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        dynamic_inputs = (
            _combo(
                "placement",
                (
                    "coordinates",
                    "top_left",
                    "top_center",
                    "top_right",
                    "right_center",
                    "bottom_right",
                    "bottom_center",
                    "bottom_left",
                    "left_center",
                    "center",
                ),
                "coordinates",
                advanced=True,
            ),
            InputSpec("x", FLOAT, required=False, default=0.0, widget=NumberWidget(step=1.0)),
            InputSpec("y", FLOAT, required=False, default=0.0, widget=NumberWidget(step=1.0)),
            InputSpec(
                "width",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, step=1.0),
            ),
            InputSpec(
                "height",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, step=1.0),
            ),
            InputSpec(
                "fill",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
            ),
            InputSpec(
                "mask_threshold",
                FLOAT,
                required=False,
                default=0.0,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                advanced=True,
            ),
            InputSpec(
                "mask_blur",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=512, step=1),
                advanced=True,
            ),
        )
        by_id = {item.id: item for item in dynamic_inputs}

        def option(key: str, *ids: str) -> DynamicComboOption:
            return DynamicComboOption(key, tuple(by_id[input_id] for input_id in ids))

        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.crop",
                version=2,
                display_name="Crop Image",
                category="image/geometry",
                inputs=(
                    InputSpec("image", IMAGE),
                    InputSpec(
                        "padding",
                        INT,
                        required=False,
                        default=0,
                        widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
                    ),
                    _combo(
                        "rounding",
                        ("floor", "round", "ceil", "expand"),
                        "expand",
                        advanced=True,
                    ),
                ),
                combos=(
                    DynamicComboSpec(
                        "source",
                        (
                            option("coordinates", "placement", "x", "y", "width", "height"),
                            DynamicComboOption(
                                "region",
                                (
                                    InputSpec("region", REGION),
                                    InputSpec("mask", MASK, required=False, default=None),
                                    by_id["mask_blur"],
                                ),
                            ),
                            DynamicComboOption(
                                "mask",
                                (
                                    InputSpec("mask", MASK),
                                    by_id["mask_threshold"],
                                    by_id["mask_blur"],
                                ),
                            ),
                        ),
                        default="coordinates",
                    ),
                    DynamicComboSpec(
                        "outside",
                        (
                            option("clip"),
                            option("pad", "fill"),
                            option("comfy"),
                        ),
                        default="clip",
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("mask", MASK, optional=True, preview=True),
                    OutputSpec("region", REGION),
                ),
                search_terms=(
                    "ImageCrop",
                    "ImageCrop+",
                    "Crop (mtb)",
                    "BatchCropFromMask",
                    "image inset crop",
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        source: str | None = None,
        region: Region | None = None,
        mask: object | None = None,
        placement: str = "coordinates",
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        padding: int = 0,
        rounding: str = "expand",
        outside: str = "clip",
        fill: float = 0.0,
        mask_threshold: float = 0.0,
        mask_blur: int = 0,
    ) -> Mapping[str, object]:
        if not math.isfinite(fill):
            raise ValueError(f"fill must be finite, got {fill}")
        if source not in (None, "coordinates", "region", "mask"):
            raise ValueError(f"unknown crop source: {source}")
        image_array = _image_array(image)
        source_height, source_width = int(image_array.shape[1]), int(image_array.shape[2])
        source_mask: np.ndarray | None = None
        if mask is not None:
            source_mask = _mask_array(mask)
            if source_mask.shape[1:3] != image_array.shape[1:3]:
                raise ValueError("mask dimensions must match image dimensions")
            source_mask = _broadcast_batch(source_mask, int(image_array.shape[0]), "mask")
            if mask_blur > 0:
                source_mask = _gaussian_blur_mask(source_mask, mask_blur // 2)
            elif mask_blur < 0:
                raise ValueError("mask_blur must be non-negative")
        if source == "region":
            if region is None:
                raise ValueError("region crop requires a region")
            selected = region
        elif source == "mask":
            if source_mask is None:
                raise ValueError("mask crop requires a mask")
            selected = _region_from_mask(source_mask, mask_threshold)
        elif source is None and region is not None:
            selected = region
        elif source is None and source_mask is not None:
            selected = _region_from_mask(source_mask, mask_threshold)
        else:
            if placement != "coordinates":
                selected_width = min(source_width, width)
                selected_height = min(source_height, height)
                if selected_width <= 0 or selected_height <= 0:
                    raise ValueError("placed crop width and height must be positive")
                center_x = round((source_width - selected_width) / 2)
                center_y = round((source_height - selected_height) / 2)
                origins = {
                    "top_left": (0, 0),
                    "top_center": (center_x, 0),
                    "top_right": (source_width - selected_width, 0),
                    "right_center": (source_width - selected_width, center_y),
                    "bottom_right": (
                        source_width - selected_width,
                        source_height - selected_height,
                    ),
                    "bottom_center": (center_x, source_height - selected_height),
                    "bottom_left": (0, source_height - selected_height),
                    "left_center": (0, center_y),
                    "center": (center_x, center_y),
                }
                origin = origins.get(placement)
                if origin is None:
                    raise ValueError(f"unknown crop placement: {placement}")
                x, y = origin[0] + x, origin[1] + y
                width, height = selected_width, selected_height
            selected = Region(
                x=x,
                y=y,
                width=source_width - x if width == 0 else width,
                height=source_height - y if height == 0 else height,
            )
        left, top, right, bottom = _integer_region(selected, cast("Rounding", rounding), padding)
        if right <= left or bottom <= top:
            raise ValueError(f"crop region is empty after rounding: {(left, top, right, bottom)}")
        output, (left, top, right, bottom) = _crop_spatial(
            image_array, left, top, right, bottom, outside, fill
        )
        output_mask: object = ABSENT
        if source_mask is not None:
            output_mask, _ = _crop_spatial(
                source_mask,
                left,
                top,
                right,
                bottom,
                outside,
                0.0,
            )
        used = Region(x=left, y=top, width=right - left, height=bottom - top)
        return cls.outputs(image=output, mask=output_mask, region=used)


def _broadcast_batch(array: np.ndarray, batch: int, subject: str) -> np.ndarray:
    if array.shape[0] == batch:
        return array
    if array.shape[0] == 1:
        return np.broadcast_to(array, (batch, *array.shape[1:]))
    raise ValueError(f"{subject} batch must be 1 or {batch}, got {array.shape[0]}")


def _gaussian_blur_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask
    if radius >= mask.shape[1] or radius >= mask.shape[2]:
        raise ValueError("blur kernel exceeds the image dimensions")
    sigma = 0.3 * radius + 0.5
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma), dtype=np.float32)
    kernel /= kernel.sum(dtype=np.float32)

    horizontal = np.pad(mask, ((0, 0), (0, 0), (radius, radius)), mode="reflect")
    horizontal_windows = np.lib.stride_tricks.sliding_window_view(
        horizontal,
        kernel.size,
        axis=2,
    )
    horizontal_blur = np.einsum(
        "bhwk,k->bhw",
        horizontal_windows,
        kernel,
        dtype=np.float32,
    )
    vertical = np.pad(horizontal_blur, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    vertical_windows = np.lib.stride_tricks.sliding_window_view(
        vertical,
        kernel.size,
        axis=1,
    )
    return np.einsum(
        "bhwk,k->bhw",
        vertical_windows,
        kernel,
        dtype=np.float32,
    )


class ImageUncrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.uncrop",
            display_name="Uncrop Image",
            category="image/geometry",
            inputs=(
                InputSpec("base", IMAGE),
                InputSpec("crop", IMAGE),
                InputSpec("region", REGION),
                InputSpec("mask", MASK, required=False, default=None),
                InputSpec(
                    "opacity",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "border_blending",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    advanced=True,
                ),
                _combo(
                    "rounding",
                    ("floor", "round", "ceil", "expand"),
                    "expand",
                    advanced=True,
                ),
                _combo("interpolation", _INTERPOLATION_OPTIONS, "bilinear", advanced=True),
                _combo("outside", ("error", "ignore"), "error", advanced=True),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=(
                "stitch crop",
                "BatchUncrop",
                "ImageUncropByMask",
                "Uncrop (mtb)",
                "InpaintStitchImproved",
                "easy imageUncropFromBBOX",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        base: object,
        crop: object,
        region: Region,
        mask: object | None = None,
        opacity: float = 1.0,
        border_blending: float = 0.0,
        rounding: str = "expand",
        interpolation: str = "bilinear",
        outside: str = "error",
    ) -> Mapping[str, object]:
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be between 0 and 1, got {opacity}")
        if not math.isfinite(border_blending) or not 0.0 <= border_blending <= 1.0:
            raise ValueError(f"border_blending must be between 0 and 1, got {border_blending}")
        if outside not in ("error", "ignore"):
            raise ValueError(f"unknown uncrop outside-image policy: {outside}")
        base_array = _image_array(base, subject="base")
        crop_array = _image_array(crop, subject="crop")
        if base_array.shape[3] != crop_array.shape[3]:
            raise ValueError(
                "base and crop channels must match, got "
                f"{base_array.shape[3]} and {crop_array.shape[3]}"
            )
        left, top, right, bottom = _integer_region(region, cast("Rounding", rounding), padding=0)
        target_width, target_height = right - left, bottom - top
        if target_width < 1 or target_height < 1:
            raise ValueError("uncrop region must have positive dimensions")
        if crop_array.shape[1:3] != (target_height, target_width):
            crop_array = _resize_array(
                crop_array,
                target_width,
                target_height,
                cast("Interpolation", interpolation),
            )
        mask_array: np.ndarray | None = None
        if mask is not None:
            mask_array = _mask_array(mask)
            if mask_array.shape[1:3] != (target_height, target_width):
                mask_array = _resize_array(
                    mask_array,
                    target_width,
                    target_height,
                    cast("Interpolation", interpolation),
                )
        batch = max(
            int(base_array.shape[0]),
            int(crop_array.shape[0]),
            1 if mask_array is None else int(mask_array.shape[0]),
        )
        output = np.array(_broadcast_batch(base_array, batch, "base"), copy=True)
        crop_array = _broadcast_batch(crop_array, batch, "crop")
        if mask_array is not None:
            mask_array = _broadcast_batch(mask_array, batch, "mask")
        base_height, base_width = int(output.shape[1]), int(output.shape[2])
        destination_left, destination_top = max(0, left), max(0, top)
        destination_right, destination_bottom = min(base_width, right), min(base_height, bottom)
        if destination_right <= destination_left or destination_bottom <= destination_top:
            if outside == "ignore":
                return cls.outputs(image=np.ascontiguousarray(output))
            raise ValueError("uncrop region does not intersect the base image")
        source_left, source_top = destination_left - left, destination_top - top
        source_right = source_left + destination_right - destination_left
        source_bottom = source_top + destination_bottom - destination_top
        source = crop_array[:, source_top:source_bottom, source_left:source_right, :]
        if mask_array is None:
            alpha = np.full((*source.shape[:3], 1), opacity, dtype=np.float32)
        else:
            alpha = (
                np.clip(
                    mask_array[:, source_top:source_bottom, source_left:source_right, None],
                    0.0,
                    1.0,
                )
                * opacity
            )
        blend_radius = int(max(target_width, target_height) * border_blending * 0.5)
        if blend_radius > 0:
            blend_mask = np.zeros((batch, base_height, base_width), dtype=np.float32)
            blend_mask[
                :,
                destination_top:destination_bottom,
                destination_left:destination_right,
            ] = 1.0
            blurred = _gaussian_blur_mask(blend_mask, blend_radius)
            alpha *= blurred[
                :,
                destination_top:destination_bottom,
                destination_left:destination_right,
                None,
            ]
        destination = output[
            :, destination_top:destination_bottom, destination_left:destination_right, :
        ]
        output[:, destination_top:destination_bottom, destination_left:destination_right, :] = (
            source * alpha + destination * (1.0 - alpha)
        )
        return cls.outputs(image=np.ascontiguousarray(output))


class ImageInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.info",
            display_name="Image Information",
            category="image/inspect",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "histogram_bins",
                    INT,
                    required=False,
                    default=256,
                    widget=NumberWidget(min=2, max=4096, step=1),
                ),
            ),
            outputs=(
                OutputSpec("width", INT),
                OutputSpec("height", INT),
                OutputSpec("count", INT),
                OutputSpec("channels", INT),
                OutputSpec("mean", FLOAT),
                OutputSpec("minimum", FLOAT),
                OutputSpec("maximum", FLOAT),
                OutputSpec("histogram", INT_LIST),
            ),
            search_terms=(
                "image size",
                "image count",
                "image statistics",
                "GetImageSizeAndCount",
                "GetImageSize+",
                "ImpactImageInfo",
                "Image Size to Number",
            ),
        )

    @classmethod
    def execute(cls, *, image: object, histogram_bins: int = 256) -> Mapping[str, object]:
        if not 2 <= histogram_bins <= 4096:
            raise ValueError(f"histogram_bins must be between 2 and 4096, got {histogram_bins}")
        array = _image_array(image)
        histogram, _ = np.histogram(np.clip(array, 0.0, 1.0), bins=histogram_bins, range=(0, 1))
        return cls.outputs(
            width=int(array.shape[2]),
            height=int(array.shape[1]),
            count=int(array.shape[0]),
            channels=int(array.shape[3]),
            mean=float(array.mean(dtype=np.float64)),
            minimum=float(array.min()),
            maximum=float(array.max()),
            histogram=[int(value) for value in histogram],
        )


GEOMETRY_NODES: tuple[type[Node], ...] = (
    MakeRegion,
    RegionInfo,
    ImageResize,
    ImageTransform,
    ImageCrop,
    ImageUncrop,
    ImageInfo,
)


__all__ = [
    "GEOMETRY_NODES",
    "ImageCrop",
    "ImageInfo",
    "ImageResize",
    "ImageTransform",
    "ImageUncrop",
    "MakeRegion",
    "RegionInfo",
]
