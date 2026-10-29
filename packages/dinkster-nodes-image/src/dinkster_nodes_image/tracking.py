"""Temporal crop and reinsertion over image batches and materialized VIDEO values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    coerce_video,
    disassemble_video,
    effective_video_facts,
)

from .geometry import ImageUncrop
from .support import (
    INTERPOLATION_OPTIONS,
    MAX_DIMENSION,
    check_output_size,
    combo_input,
    image_array,
    mask_array,
    materialized_inputs,
)
from .types import Detection, Region

IMAGE_TYPE = "dinkster.image"
VIDEO_TYPE = "comfy.VIDEO"

MEDIA = TypeExpr.variable("media_type", (IMAGE_TYPE, VIDEO_TYPE))
BASE_MEDIA = TypeExpr.variable("base_media_type", (IMAGE_TYPE, VIDEO_TYPE))
IMAGE_OR_VIDEO = TypeExpr.union(IMAGE_TYPE, VIDEO_TYPE)
MASK = TypeExpr.concrete("dinkster.mask")
REGION = TypeExpr.concrete("dinkster.region")
REGION_LIST = TypeExpr.list_of(REGION)
REGIONS_OR_DETECTIONS = TypeExpr.list_of(TypeExpr.union("dinkster.region", "dinkster.detection"))
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)


@dataclass(frozen=True)
class _Media:
    frames: np.ndarray
    video: bool
    fps: Fraction | None = None
    audio: object = None
    bit_depth: int = 8
    color_space: str = "sRGB"
    color: Mapping[str, object] | None = None


def _materialize_media(value: object, subject: str) -> _Media:
    if not isinstance(value, Mapping):
        return _Media(image_array(value, subject=subject), False)

    try:
        video = coerce_video(cast("Mapping[str, object]", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{subject} must be an IMAGE or VIDEO value") from exc
    facts = effective_video_facts(video)
    rate = cast("Fraction | None", facts["fps"])
    components = cast("Mapping[str, object] | None", video.get("components"))
    if components is not None and not cast("list[object]", video["edits"]):
        frames = image_array(components["images"], subject=f"{subject} frames")
        return _Media(
            frames,
            True,
            cast(Fraction, components["fps"]),
            components["audio"],
            cast(int, components["bit_depth"]),
            cast(str, components["color_space"]),
            cast("Mapping[str, object]", components["color"]),
        )

    parts = disassemble_video(video)
    frames = image_array(parts["images"], subject=f"{subject} frames")
    probe = cast("Mapping[str, object]", video["probe"])
    color_space = cast(str, parts["color_space"])
    bit_depth = cast("int | None", parts["bit_depth"])
    if color_space not in ("sRGB", "HDR", "HDR PQ") or bit_depth not in (8, 10):
        raise ValueError(
            f"{subject} VIDEO metadata cannot be represented by a materialized component VIDEO"
        )
    if rate is None:
        rate = Fraction(str(parts["fps"]))
    return _Media(
        frames,
        True,
        rate,
        parts["audio"],
        bit_depth,
        color_space,
        {key: probe[key] for key in ("primaries", "transfer", "matrix", "range")},
    )


def _output_media(media: _Media, frames: np.ndarray) -> object:
    if not media.video:
        return frames
    assert media.fps is not None and media.color is not None
    return coerce_video(
        {
            "components": {
                "images": frames,
                "audio": media.audio,
                "fps": media.fps,
                "bit_depth": media.bit_depth,
                "color_space": media.color_space,
                "color": dict(media.color),
            },
            "edits": [],
        }
    )


def _visible_region(
    region: Region, width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    if (
        region.width == 0
        or region.height == 0
        or region.right <= 0
        or region.bottom <= 0
        or region.x >= width
        or region.y >= height
    ):
        return 0, 0, 0, 0
    left = max(0, math.floor(region.x - padding))
    top = max(0, math.floor(region.y - padding))
    right = min(width, math.ceil(region.right + padding))
    bottom = min(height, math.ceil(region.bottom + padding))
    return left, top, right, bottom


def _mask_region(mask: np.ndarray, threshold: float) -> Region | None:
    rows, columns = np.nonzero(mask > threshold)
    if rows.size == 0:
        return None
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return Region(left, top, right - left, bottom - top)


def _window(center: float, size: int, limit: int) -> int:
    return min(max(0, math.floor(center - size / 2)), limit - size)


def _validate_count(actual: int, expected: int, subject: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{subject} count must equal the media frame count, got {actual} and {expected}"
        )


class TrackedCrop(Node):
    """Crop one independently tracked selection per frame into a fixed-size batch."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.tracked_crop",
            display_name="Tracked Crop",
            category="image/geometry",
            description=(
                "Crop one tracked selection per frame. The largest visible selection sets the "
                "fixed output size; empty frames remain as zero crops and zero-area regions."
            ),
            inputs=(
                InputSpec("media", MEDIA),
                InputSpec(
                    "padding",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=MAX_DIMENSION, step=1),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "source",
                    (
                        DynamicComboOption(
                            "masks",
                            (
                                InputSpec("masks", MASK),
                                InputSpec(
                                    "threshold",
                                    FLOAT,
                                    required=False,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "bounding_boxes",
                            (InputSpec("bounding_boxes", REGIONS_OR_DETECTIONS),),
                        ),
                    ),
                    default="masks",
                ),
            ),
            outputs=(
                OutputSpec("media", MEDIA, preview=True),
                OutputSpec("regions", REGION_LIST),
                OutputSpec("masks", MASK, preview=True),
            ),
            search_terms=("tracked crop", "crop sequence", "crop video by mask"),
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        media: object,
        source: str = "masks",
        masks: object | None = None,
        bounding_boxes: Sequence[object] | None = None,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> Mapping[str, object]:
        if type(padding) is not int or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")

        source_masks: np.ndarray | None = None
        boxes: tuple[object, ...] | None = None
        if source == "masks":
            if masks is None:
                raise ValueError("mask tracking requires masks")
            source_masks = mask_array(masks, subject="masks")
        elif source == "bounding_boxes":
            if bounding_boxes is None:
                raise ValueError("bounding-box tracking requires bounding_boxes")
            boxes = tuple(bounding_boxes)
            if any(not isinstance(box, (Region, Detection)) for box in boxes):
                raise TypeError("bounding_boxes must contain Region or Detection values")
        else:
            raise ValueError(f"unknown tracking source: {source}")

        value = _materialize_media(media, "media")
        frames = value.frames
        count, height, width, channels = frames.shape

        coverage: list[np.ndarray | None] = []
        selected: list[Region | None] = []
        if source == "masks":
            assert source_masks is not None
            _validate_count(int(source_masks.shape[0]), count, "mask")
            if source_masks.shape[1:3] != (height, width):
                raise ValueError("mask dimensions must match the media frames")
            for frame_mask in source_masks:
                selected.append(_mask_region(frame_mask, threshold))
                coverage.append(frame_mask)
        else:
            assert boxes is not None
            _validate_count(len(boxes), count, "bounding box")
            for box in boxes:
                if isinstance(box, Detection):
                    selected.append(box.region)
                    if box.mask is not None and box.mask.shape != (height, width):
                        raise ValueError("detection mask dimensions must match the media frames")
                    coverage.append(box.mask)
                elif isinstance(box, Region):
                    selected.append(box)
                    coverage.append(None)

        visible = [
            None if region is None else _visible_region(region, width, height, padding)
            for region in selected
        ]
        visible = [
            bounds
            if bounds is not None and bounds[2] > bounds[0] and bounds[3] > bounds[1]
            else None
            for bounds in visible
        ]
        output_width = max((bounds[2] - bounds[0] for bounds in visible if bounds), default=1)
        output_height = max((bounds[3] - bounds[1] for bounds in visible if bounds), default=1)
        check_output_size((count, output_height, output_width, channels))
        check_output_size((count, output_height, output_width, 1))

        cropped = np.zeros((count, output_height, output_width, channels), dtype=np.float32)
        cropped_masks = np.zeros((count, output_height, output_width), dtype=np.float32)
        regions: list[Region] = []
        for index, bounds in enumerate(visible):
            if bounds is None:
                regions.append(Region(0, 0, 0, 0))
                continue
            left, top, right, bottom = bounds
            window_left = _window((left + right) / 2, output_width, width)
            window_top = _window((top + bottom) / 2, output_height, height)
            window_right = window_left + output_width
            window_bottom = window_top + output_height
            cropped[index] = frames[index, window_top:window_bottom, window_left:window_right]
            frame_coverage = coverage[index]
            if frame_coverage is None:
                cropped_masks[index] = 1.0
            else:
                cropped_masks[index] = frame_coverage[
                    window_top:window_bottom, window_left:window_right
                ]
            regions.append(Region(window_left, window_top, output_width, output_height))

        return cls.outputs(
            media=_output_media(value, cropped),
            regions=regions,
            masks=np.ascontiguousarray(cropped_masks),
        )


class TrackedUncrop(Node):
    """Reinsert one processed crop per frame without temporal broadcasting."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.tracked_uncrop",
            display_name="Tracked Uncrop",
            category="image/geometry",
            description=(
                "Reinsert processed crops at their matching tracked regions. Zero-area regions "
                "leave the corresponding base frame unchanged."
            ),
            inputs=(
                InputSpec("base", BASE_MEDIA),
                InputSpec("crop", IMAGE_OR_VIDEO),
                InputSpec("regions", REGION_LIST),
                InputSpec("masks", MASK, required=False, default=None),
                InputSpec(
                    "opacity",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                combo_input("interpolation", INTERPOLATION_OPTIONS, "bilinear", advanced=True),
            ),
            outputs=(OutputSpec("media", BASE_MEDIA, preview=True),),
            search_terms=("tracked uncrop", "reinsert crop sequence", "composite tracked crop"),
        )

    @classmethod
    def execute(
        cls,
        *,
        base: object,
        crop: object,
        regions: Sequence[object],
        masks: object | None = None,
        opacity: float = 1.0,
        interpolation: str = "bilinear",
    ) -> Mapping[str, object]:
        if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be between 0 and 1, got {opacity}")
        if interpolation not in INTERPOLATION_OPTIONS:
            raise ValueError(f"unknown interpolation mode: {interpolation}")
        tracked_regions = tuple(regions)
        if any(not isinstance(region, Region) for region in tracked_regions):
            raise TypeError("regions must contain Region values")
        checked_regions = cast("tuple[Region, ...]", tracked_regions)
        for region in checked_regions:
            if region.width > MAX_DIMENSION or region.height > MAX_DIMENSION:
                raise ValueError(f"tracked region dimensions cannot exceed {MAX_DIMENSION}")
        crop_masks = None if masks is None else mask_array(masks, subject="masks")

        base_value = _materialize_media(base, "base")
        count = int(base_value.frames.shape[0])
        _validate_count(len(checked_regions), count, "region")
        if crop_masks is not None:
            _validate_count(int(crop_masks.shape[0]), count, "mask")
        crop_value = _materialize_media(crop, "crop")
        _validate_count(int(crop_value.frames.shape[0]), count, "crop frame")
        if base_value.frames.shape[3] != crop_value.frames.shape[3]:
            raise ValueError("base and crop channels must match")
        if base_value.video and crop_value.video and base_value.fps != crop_value.fps:
            raise ValueError("base and crop VIDEO timeline rates must match")

        check_output_size(base_value.frames.shape)
        output = np.array(base_value.frames, copy=True)
        for index, region in enumerate(checked_regions):
            if region.width == 0 or region.height == 0:
                continue
            result = ImageUncrop.execute(
                base=output[index : index + 1],
                crop=crop_value.frames[index : index + 1],
                region=region,
                mask=None if crop_masks is None else crop_masks[index : index + 1],
                opacity=opacity,
                interpolation=interpolation,
                outside="ignore",
            )
            output[index] = cast(np.ndarray, result["image"])[0]
        return cls.outputs(media=_output_media(base_value, np.ascontiguousarray(output)))


TRACKING_NODES: tuple[type[Node], ...] = (TrackedCrop, TrackedUncrop)

__all__ = ["TRACKING_NODES", "TrackedCrop", "TrackedUncrop"]
