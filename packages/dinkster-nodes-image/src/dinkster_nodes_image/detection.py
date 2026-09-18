"""Deterministic CPU detection construction, inspection, and mask conversion."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .mask import dilate_mask
from .support import MAX_DIMENSION
from .support import combo_input as _combo
from .support import mask_array as _mask_array
from .support import number_input as _number
from .support import validate_canvas as _validate_canvas
from .types import DETECTION_TYPE, REGION_TYPE, Detection, Region

MASK = TypeExpr.concrete("dinkster.mask")
REGION = TypeExpr.concrete(REGION_TYPE)
DETECTION = TypeExpr.concrete(DETECTION_TYPE)
DETECTIONS = TypeExpr.list_of(DETECTION)
MASKS = TypeExpr.list_of(MASK)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)

_SORT_KEYS = ("score", "area", "left", "top", "label")


def _detection_items(detections: Sequence[object]) -> list[Detection]:
    items: list[Detection] = []
    for index, item in enumerate(detections):
        if not isinstance(item, Detection):
            raise TypeError(f"detections[{index}] must be a detection, got {type(item).__name__}")
        items.append(item)
    return items


class MakeDetection(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.make",
            display_name="Create Detection",
            category="image/detection",
            inputs=(
                InputSpec("region", REGION),
                InputSpec("label", STRING, required=False, default="object", widget=StringWidget()),
                _number("score", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
                InputSpec("mask", MASK, required=False),
            ),
            outputs=(OutputSpec("detection", DETECTION),),
            search_terms=("bbox annotation", "SEGS", "labeled region"),
        )

    @classmethod
    def execute(
        cls,
        *,
        region: Region,
        label: str = "object",
        score: float = 1.0,
        mask: object = ABSENT,
    ) -> Mapping[str, object]:
        detection_mask: np.ndarray | None = None
        if mask is not ABSENT:
            array = _mask_array(mask)
            if array.shape[0] != 1:
                raise ValueError(
                    f"detection mask must be a single mask, got batch {array.shape[0]}"
                )
            detection_mask = array[0]
        return cls.outputs(
            detection=Detection(label=label, score=score, region=region, mask=detection_mask)
        )


class DetectionInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.info",
            display_name="Detection Information",
            category="image/detection",
            inputs=(InputSpec("detection", DETECTION),),
            outputs=(
                OutputSpec("region", REGION),
                OutputSpec("label", STRING),
                OutputSpec("score", FLOAT),
                OutputSpec("area", FLOAT),
                OutputSpec("has_mask", BOOLEAN),
            ),
            search_terms=("detection label", "detection score", "bbox info"),
        )

    @classmethod
    def execute(cls, *, detection: Detection) -> Mapping[str, object]:
        return cls.outputs(
            region=detection.region,
            label=detection.label,
            score=float(detection.score),
            area=detection.area,
            has_mask=detection.mask is not None,
        )


class DetectionFilter(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.filter",
            display_name="Filter Detections",
            category="image/detection",
            inputs=(
                InputSpec("detections", DETECTIONS),
                InputSpec("labels", STRING, required=False, default="", widget=StringWidget()),
                _number("min_score", FLOAT, 0.0, minimum=0.0, maximum=1.0, step=0.01),
                _number("min_area", FLOAT, 0.0, minimum=0.0, step=1.0),
                _number("max_area", FLOAT, 0.0, minimum=0.0, step=1.0),
                _number("max_results", INT, 0, minimum=0, step=1),
            ),
            outputs=(
                OutputSpec("detections", DETECTIONS),
                OutputSpec("count", INT),
            ),
            search_terms=(
                "SEGS filter",
                "detection threshold",
                "filter by label",
                "confidence filter",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        detections: Sequence[object],
        labels: str = "",
        min_score: float = 0.0,
        min_area: float = 0.0,
        max_area: float = 0.0,
        max_results: int = 0,
    ) -> Mapping[str, object]:
        if not all(math.isfinite(value) for value in (min_score, min_area, max_area)):
            raise ValueError("detection filter bounds must be finite")
        if max_results < 0:
            raise ValueError("max_results must be non-negative")
        allowed = frozenset(part.strip() for part in labels.split(",") if part.strip())
        kept: list[Detection] = []
        for detection in _detection_items(detections):
            if allowed and detection.label not in allowed:
                continue
            if detection.score < min_score:
                continue
            if detection.area < min_area:
                continue
            if max_area > 0.0 and detection.area > max_area:
                continue
            kept.append(detection)
        if max_results > 0:
            kept = kept[:max_results]
        return cls.outputs(detections=kept, count=len(kept))


class DetectionSort(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.sort",
            display_name="Sort Detections",
            category="image/detection",
            inputs=(
                InputSpec("detections", DETECTIONS),
                _combo("key", _SORT_KEYS, "score"),
                InputSpec("descending", BOOLEAN, required=False, default=True),
            ),
            outputs=(OutputSpec("detections", DETECTIONS),),
            search_terms=("SEGS ordered filter", "sort by confidence", "largest detection"),
        )

    @classmethod
    def execute(
        cls, *, detections: Sequence[object], key: str = "score", descending: bool = True
    ) -> Mapping[str, object]:
        if key not in _SORT_KEYS:
            raise ValueError(f"unknown detection sort key: {key}")
        items = _detection_items(detections)
        if key == "label":
            ordered = sorted(items, key=lambda detection: detection.label, reverse=descending)
        else:
            numeric: dict[str, Callable[[Detection], float]] = {
                "score": lambda detection: detection.score,
                "area": lambda detection: detection.area,
                "left": lambda detection: float(detection.region.x),
                "top": lambda detection: float(detection.region.y),
            }
            ordered = sorted(items, key=numeric[key], reverse=descending)
        return cls.outputs(detections=ordered)


class DetectionsToMasks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.to_masks",
            display_name="Detections to Masks",
            category="image/detection",
            inputs=(
                InputSpec("detections", DETECTIONS),
                _number("width", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                _number("height", INT, 512, minimum=1, maximum=MAX_DIMENSION, step=1),
                _number("dilation", INT, 0, minimum=-4096, maximum=4096, step=1),
                InputSpec("tapered_corners", BOOLEAN, required=False, default=True, advanced=True),
            ),
            outputs=(
                OutputSpec("masks", MASKS),
                OutputSpec("combined", MASK, preview=True),
            ),
            search_terms=(
                "SegsToCombinedMask",
                "SEGS to mask list",
                "detection mask",
                "bbox to mask",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        detections: Sequence[object],
        width: int = 512,
        height: int = 512,
        dilation: int = 0,
        tapered_corners: bool = True,
    ) -> Mapping[str, object]:
        _validate_canvas(width, height, 1)
        items = _detection_items(detections)
        masks: list[np.ndarray] = []
        combined = np.zeros((1, height, width), dtype=np.float32)
        for index, detection in enumerate(items):
            if detection.mask is not None:
                if detection.mask.shape != (height, width):
                    raise ValueError(
                        f"detections[{index}] mask shape {detection.mask.shape} does not match "
                        f"the ({height}, {width}) canvas"
                    )
                mask = detection.mask.reshape(1, height, width).copy()
            else:
                mask = np.zeros((1, height, width), dtype=np.float32)
                region = detection.region
                if region.width > 0 and region.height > 0:
                    # Clamp raw extents to the canvas before floor/ceil:
                    # right/bottom of a finite region can overflow to float
                    # infinity or to an int beyond float range.
                    left = math.floor(min(max(region.x, 0), width))
                    top = math.floor(min(max(region.y, 0), height))
                    right = math.ceil(min(max(region.right, 0), width))
                    bottom = math.ceil(min(max(region.bottom, 0), height))
                    if left < right and top < bottom:
                        mask[:, top:bottom, left:right] = 1.0
            mask = dilate_mask(mask, dilation, tapered_corners)
            masks.append(mask)
            np.maximum(combined, mask, out=combined)
        return cls.outputs(masks=masks, combined=combined)


DETECTION_NODES: tuple[type[Node], ...] = (
    MakeDetection,
    DetectionInfo,
    DetectionFilter,
    DetectionSort,
    DetectionsToMasks,
)


__all__ = [
    "DETECTION_NODES",
    "DetectionFilter",
    "DetectionInfo",
    "DetectionSort",
    "DetectionsToMasks",
    "MakeDetection",
]
