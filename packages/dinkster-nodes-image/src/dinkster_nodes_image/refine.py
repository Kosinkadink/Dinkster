"""Deterministic tiled-refine geometry and blend masks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from dinkster_api.v1 import (
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)
from PIL import Image, ImageFilter

from .filters import MAX_FILTER_RADIUS
from .geometry import BOOLEAN, INT, MASK, REGION
from .support import MAX_DIMENSION
from .support import combo_input as _combo
from .support import number_input as _number
from .support import validate_canvas as _validate_canvas
from .types import Region

STRING = TypeExpr.concrete(CORE_STRING)
REGION_LIST = TypeExpr.list_of(REGION)
INT_LIST = TypeExpr.list_of(INT)
STRING_LIST = TypeExpr.list_of(STRING)

MAX_TILE_SIZE = 8192
_PHASES = ("redraw", "seam_fix")
_REDRAW_MODES = ("linear", "chess", "none")
_SEAM_FIX_MODES = ("none", "band_pass", "half_tile", "half_tile_intersections")
_MASK_KINDS = ("rectangle", "tent_horizontal", "tent_vertical", "radial_inverted")


@dataclass(frozen=True)
class _RefineJob:
    region: Region
    crop: Region
    sample_width: int
    sample_height: int
    mask_kind: str
    mask_blur: int


def _bounded_integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _validate_plan_inputs(
    *,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    phase: str,
    mode: str,
    mask_blur: int,
    tile_padding: int,
    force_uniform_tiles: bool,
    seam_fix_mode: str,
    seam_fix_width: int,
    seam_fix_mask_blur: int,
    seam_fix_padding: int,
) -> None:
    _bounded_integer("width", width, 1, MAX_DIMENSION)
    _bounded_integer("height", height, 1, MAX_DIMENSION)
    _validate_canvas(width, height, 1)
    _bounded_integer("tile_width", tile_width, 8, MAX_TILE_SIZE)
    _bounded_integer("tile_height", tile_height, 8, MAX_TILE_SIZE)
    _bounded_integer("mask_blur", mask_blur, 0, MAX_FILTER_RADIUS)
    _bounded_integer("tile_padding", tile_padding, 0, MAX_TILE_SIZE)
    _bounded_integer("seam_fix_width", seam_fix_width, 8, MAX_TILE_SIZE)
    _bounded_integer("seam_fix_mask_blur", seam_fix_mask_blur, 0, MAX_FILTER_RADIUS)
    _bounded_integer("seam_fix_padding", seam_fix_padding, 0, MAX_TILE_SIZE)
    if type(force_uniform_tiles) is not bool:
        raise ValueError("force_uniform_tiles must be a boolean")
    if phase not in _PHASES:
        raise ValueError(f"unknown tiled refine phase: {phase}")
    if mode not in _REDRAW_MODES:
        raise ValueError(f"unknown tiled refine mode: {mode}")
    if seam_fix_mode not in _SEAM_FIX_MODES:
        raise ValueError(f"unknown seam fix mode: {seam_fix_mode}")


def _fix_crop_region(
    region: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = region
    if x2 < width:
        x2 -= 1
    if y2 < height:
        y2 -= 1
    return x1, y1, x2, y2


def _crop_region(
    region: Region, width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    x1 = max(int(region.x) - padding, 0)
    y1 = max(int(region.y) - padding, 0)
    x2 = min(int(region.right) + padding, width)
    y2 = min(int(region.bottom) + padding, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("tile region does not intersect the canvas")
    return _fix_crop_region((x1, y1, x2, y2), width, height)


def _expand_crop(
    region: tuple[int, int, int, int],
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)
    return x1, y1, x2, y2


def _round_sample_size(tile_width: int, tile_height: int, padding: int) -> tuple[int, int]:
    return round((tile_width + padding) / 8) * 8, round((tile_height + padding) / 8) * 8


def _prepare_crop(
    region: Region,
    *,
    width: int,
    height: int,
    padding: int,
    sample_width: int,
    sample_height: int,
    uniform: bool,
) -> tuple[Region, int, int]:
    crop = _crop_region(region, width, height, padding)
    x1, y1, x2, y2 = crop
    crop_width, crop_height = x2 - x1, y2 - y1
    if uniform:
        crop_ratio = crop_width / crop_height
        sample_ratio = sample_width / sample_height
        if crop_ratio > sample_ratio:
            target_width = crop_width
            target_height = round(crop_width / sample_ratio)
        else:
            target_width = round(crop_height * sample_ratio)
            target_height = crop_height
    else:
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        sample_width, sample_height = target_width, target_height
    x1, y1, x2, y2 = _expand_crop(crop, width, height, target_width, target_height)
    return Region(x1, y1, x2 - x1, y2 - y1), sample_width, sample_height


def _job(
    region: Region,
    *,
    width: int,
    height: int,
    padding: int,
    sample_width: int,
    sample_height: int,
    uniform: bool,
    mask_kind: str,
    mask_blur: int,
) -> _RefineJob:
    crop, actual_sample_width, actual_sample_height = _prepare_crop(
        region,
        width=width,
        height=height,
        padding=padding,
        sample_width=sample_width,
        sample_height=sample_height,
        uniform=uniform,
    )
    return _RefineJob(
        region,
        crop,
        actual_sample_width,
        actual_sample_height,
        mask_kind,
        mask_blur,
    )


class TileRefinePlan(Node):
    """Plan UltimateSDUpscale-style redraw and seam jobs without image IO.

    Tile sampling dimensions use Python round, including its ties-to-even rule.
    Redraw rectangle regions span ``tile_width + 1`` by ``tile_height + 1``
    pixels before canvas clipping because the reference extension paints them
    with PIL ``ImageDraw.rectangle``, whose endpoints are inclusive; crops and
    blend masks derive from that painted extent. Seam-fix regions are pasted
    gradients with exclusive extents.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.tile_refine_plan",
            display_name="Plan Tiled Refine",
            category="image/tiles",
            inputs=(
                InputSpec("width", INT, widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1)),
                InputSpec("height", INT, widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1)),
                _number("tile_width", INT, 512, minimum=8, maximum=MAX_TILE_SIZE, step=1),
                _number("tile_height", INT, 512, minimum=8, maximum=MAX_TILE_SIZE, step=1),
                _combo("phase", _PHASES, "redraw"),
                _combo("mode", _REDRAW_MODES, "linear"),
                _number("mask_blur", INT, 8, minimum=0, maximum=MAX_FILTER_RADIUS, step=1),
                _number("tile_padding", INT, 32, minimum=0, maximum=MAX_TILE_SIZE, step=1),
                InputSpec("force_uniform_tiles", BOOLEAN, required=False, default=True),
                _combo("seam_fix_mode", _SEAM_FIX_MODES, "none"),
                _number("seam_fix_width", INT, 64, minimum=8, maximum=MAX_TILE_SIZE, step=1),
                _number(
                    "seam_fix_mask_blur",
                    INT,
                    8,
                    minimum=0,
                    maximum=MAX_FILTER_RADIUS,
                    step=1,
                ),
                _number("seam_fix_padding", INT, 16, minimum=0, maximum=MAX_TILE_SIZE, step=1),
            ),
            outputs=(
                OutputSpec("regions", REGION_LIST),
                OutputSpec("crops", REGION_LIST),
                OutputSpec("sample_widths", INT_LIST),
                OutputSpec("sample_heights", INT_LIST),
                OutputSpec("mask_kinds", STRING_LIST),
                OutputSpec("mask_blurs", INT_LIST),
                OutputSpec("count", INT),
            ),
            search_terms=("UltimateSDUpscale", "upscale tiles", "seam fix"),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        tile_width: int = 512,
        tile_height: int = 512,
        phase: str = "redraw",
        mode: str = "linear",
        mask_blur: int = 8,
        tile_padding: int = 32,
        force_uniform_tiles: bool = True,
        seam_fix_mode: str = "none",
        seam_fix_width: int = 64,
        seam_fix_mask_blur: int = 8,
        seam_fix_padding: int = 16,
    ) -> Mapping[str, object]:
        _validate_plan_inputs(
            width=width,
            height=height,
            tile_width=tile_width,
            tile_height=tile_height,
            phase=phase,
            mode=mode,
            mask_blur=mask_blur,
            tile_padding=tile_padding,
            force_uniform_tiles=force_uniform_tiles,
            seam_fix_mode=seam_fix_mode,
            seam_fix_width=seam_fix_width,
            seam_fix_mask_blur=seam_fix_mask_blur,
            seam_fix_padding=seam_fix_padding,
        )
        rows = math.ceil(height / tile_height)
        columns = math.ceil(width / tile_width)
        jobs: list[_RefineJob] = []

        if phase == "redraw" and mode != "none":
            sample_width, sample_height = _round_sample_size(tile_width, tile_height, tile_padding)
            parities = (None,) if mode == "linear" else (0, 1)
            for parity in parities:
                for y_index in range(rows):
                    for x_index in range(columns):
                        if parity is not None and (x_index + y_index) % 2 != parity:
                            continue
                        region = Region(
                            x_index * tile_width,
                            y_index * tile_height,
                            tile_width + 1,
                            tile_height + 1,
                        )
                        jobs.append(
                            _job(
                                region,
                                width=width,
                                height=height,
                                padding=tile_padding,
                                sample_width=sample_width,
                                sample_height=sample_height,
                                uniform=force_uniform_tiles,
                                mask_kind="rectangle",
                                mask_blur=mask_blur,
                            )
                        )
        elif phase == "seam_fix" and seam_fix_mode != "none":
            tile_sample_width, tile_sample_height = _round_sample_size(
                tile_width, tile_height, seam_fix_padding
            )
            if seam_fix_mode == "band_pass":
                for boundary in range(1, columns):
                    region = Region(
                        boundary * tile_width - seam_fix_width // 2,
                        0,
                        seam_fix_width,
                        height,
                    )
                    jobs.append(
                        _job(
                            region,
                            width=width,
                            height=height,
                            padding=seam_fix_padding,
                            sample_width=seam_fix_width,
                            sample_height=height,
                            uniform=False,
                            mask_kind="tent_horizontal",
                            mask_blur=0,
                        )
                    )
                for boundary in range(1, rows):
                    region = Region(
                        0,
                        boundary * tile_height - seam_fix_width // 2,
                        width,
                        seam_fix_width,
                    )
                    jobs.append(
                        _job(
                            region,
                            width=width,
                            height=height,
                            padding=seam_fix_padding,
                            sample_width=width,
                            sample_height=seam_fix_width,
                            uniform=False,
                            mask_kind="tent_vertical",
                            mask_blur=0,
                        )
                    )
            else:
                for y_index in range(rows - 1):
                    for x_index in range(columns):
                        region = Region(
                            x_index * tile_width,
                            y_index * tile_height + tile_height // 2,
                            tile_width,
                            tile_height,
                        )
                        jobs.append(
                            _job(
                                region,
                                width=width,
                                height=height,
                                padding=seam_fix_padding,
                                sample_width=tile_sample_width,
                                sample_height=tile_sample_height,
                                uniform=force_uniform_tiles,
                                mask_kind="tent_vertical",
                                mask_blur=seam_fix_mask_blur,
                            )
                        )
                for y_index in range(rows):
                    for x_index in range(columns - 1):
                        region = Region(
                            x_index * tile_width + tile_width // 2,
                            y_index * tile_height,
                            tile_width,
                            tile_height,
                        )
                        jobs.append(
                            _job(
                                region,
                                width=width,
                                height=height,
                                padding=seam_fix_padding,
                                sample_width=tile_sample_width,
                                sample_height=tile_sample_height,
                                uniform=force_uniform_tiles,
                                mask_kind="tent_horizontal",
                                mask_blur=seam_fix_mask_blur,
                            )
                        )
                if seam_fix_mode == "half_tile_intersections":
                    for y_index in range(rows - 1):
                        for x_index in range(columns - 1):
                            region = Region(
                                x_index * tile_width + tile_width // 2,
                                y_index * tile_height + tile_height // 2,
                                tile_width,
                                tile_height,
                            )
                            jobs.append(
                                _job(
                                    region,
                                    width=width,
                                    height=height,
                                    padding=0,
                                    sample_width=tile_sample_width,
                                    sample_height=tile_sample_height,
                                    uniform=force_uniform_tiles,
                                    mask_kind="radial_inverted",
                                    mask_blur=seam_fix_mask_blur,
                                )
                            )

        return cls.outputs(
            regions=[job.region for job in jobs],
            crops=[job.crop for job in jobs],
            sample_widths=[job.sample_width for job in jobs],
            sample_heights=[job.sample_height for job in jobs],
            mask_kinds=[job.mask_kind for job in jobs],
            mask_blurs=[job.mask_blur for job in jobs],
            count=len(jobs),
        )


def _integer_mask_region(region: Region) -> tuple[int, int, int, int]:
    values = (region.x, region.y, region.width, region.height)
    if any(float(value) != int(value) for value in values):
        raise ValueError("tile blend region coordinates and dimensions must be integers")
    x, y, width, height = (int(value) for value in values)
    if not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION:
        raise ValueError(f"tile blend region dimensions must be between 1 and {MAX_DIMENSION}")
    return x, y, width, height


class TileBlendMask(Node):
    """Create a clipped full-canvas mask for one tiled-refine job.

    Tent samples follow ``1 - abs(2 * (position + 0.5) / size - 1)``. The
    radial profile is ``max(0, 1 - sqrt(dx**2 + dy**2))`` over normalized
    pixel-center offsets and approximates the reference extension's
    bicubic-resized radial gradient.
    Blur follows the reference's 8-bit Pillow GaussianBlur path exactly.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mask.tile_blend",
            display_name="Tile Blend Mask",
            category="mask/create",
            inputs=(
                InputSpec("width", INT, widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1)),
                InputSpec("height", INT, widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1)),
                InputSpec("region", REGION),
                _combo("kind", _MASK_KINDS, "rectangle"),
                _number("blur", INT, 0, minimum=0, maximum=MAX_FILTER_RADIUS, step=1),
            ),
            outputs=(OutputSpec("mask", MASK, preview=True),),
            search_terms=("UltimateSDUpscale", "tile mask", "seam mask"),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        region: object,
        kind: str = "rectangle",
        blur: int = 0,
    ) -> Mapping[str, object]:
        _bounded_integer("width", width, 1, MAX_DIMENSION)
        _bounded_integer("height", height, 1, MAX_DIMENSION)
        _validate_canvas(width, height, 1)
        _bounded_integer("blur", blur, 0, MAX_FILTER_RADIUS)
        if not isinstance(region, Region):
            raise TypeError("region must be a Region")
        if kind not in _MASK_KINDS:
            raise ValueError(f"unknown tile blend mask kind: {kind}")
        x, y, region_width, region_height = _integer_mask_region(region)
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + region_width), min(height, y + region_height)
        output = np.zeros((1, height, width), dtype=np.float32)
        if right > left and bottom > top:
            local_columns = np.arange(left - x, right - x, dtype=np.float32)
            local_rows = np.arange(top - y, bottom - y, dtype=np.float32)
            if kind == "rectangle":
                profile = np.ones((bottom - top, right - left), dtype=np.float32)
            elif kind == "tent_horizontal":
                ramp = 1.0 - np.abs(2.0 * (local_columns + 0.5) / region_width - 1.0)
                profile = np.broadcast_to(ramp, (bottom - top, right - left))
            elif kind == "tent_vertical":
                ramp = 1.0 - np.abs(2.0 * (local_rows + 0.5) / region_height - 1.0)
                profile = np.broadcast_to(ramp[:, None], (bottom - top, right - left))
            else:
                normalized_x = (local_columns + 0.5 - region_width / 2) / (region_width / 2)
                normalized_y = (local_rows + 0.5 - region_height / 2) / (region_height / 2)
                distance = np.sqrt(
                    np.square(normalized_y[:, None]) + np.square(normalized_x[None, :])
                )
                profile = np.maximum(0.0, 1.0 - distance)
            output[0, top:bottom, left:right] = profile
        if blur:
            raster = np.rint(np.clip(output[0], 0.0, 1.0) * 255.0).astype(np.uint8)
            blurred = Image.fromarray(raster, mode="L").filter(ImageFilter.GaussianBlur(blur))
            output = np.asarray(blurred, dtype=np.float32)[None, ...] / 255.0
        return cls.outputs(mask=np.ascontiguousarray(output, dtype=np.float32))


REFINE_NODES: tuple[type[Node], ...] = (TileRefinePlan, TileBlendMask)


__all__ = ["REFINE_NODES", "TileBlendMask", "TileRefinePlan"]
