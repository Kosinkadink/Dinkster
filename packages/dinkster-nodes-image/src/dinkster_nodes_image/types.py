"""Spatial value types shared by first-party image operations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from dinkster_api.v1 import (
    DETECTION_TYPE,
    REGION_TYPE,
    Detection,
    Region,
    TypeRegistry,
    coerce_detection,
    coerce_region,
    decode_detection,
    decode_region,
    detection_meta,
    encode_detection,
    encode_region,
    region_meta,
)

from .compositor_types import (
    COMPOSITOR_TYPE,
    LAYERS_TYPE,
)

POSE_TYPE = "dinkster.pose"

MAX_POSE_DIMENSION = 16_384
MAX_POSE_SKELETONS = 1_024
MAX_SKELETON_KEYPOINTS = 4_096
MAX_SKELETON_EDGES = 8_192
MAX_KEYPOINT_COORDINATE = 32_767.0
POSE_KINDS = frozenset(("person", "face", "left_hand", "right_hand", "animal"))

# Reserved for Dinkster#682; this pack intentionally provides no vector runtime.
VECTOR_TYPE = "dinkster.vector"
# Reserved for Dinkster#677; a segmentation result is list<dinkster.detection>
# whose members carry masks, so no separate runtime type exists.
SEGMENTATION_TYPE = "dinkster.segmentation"
RESERVED_TYPE_IDS = frozenset((VECTOR_TYPE, SEGMENTATION_TYPE))


def _finite_number(value: object, subject: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{subject} must be a number")
    try:
        result = float(cast("int | float", value))
    except OverflowError as exc:
        raise ValueError(f"{subject} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{subject} must be finite")
    return result


@dataclass(frozen=True)
class PoseKeypoint:
    """One 2D pose keypoint and its confidence."""

    x: float
    y: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        x = _finite_number(self.x, "keypoint x")
        y = _finite_number(self.y, "keypoint y")
        confidence = _finite_number(self.confidence, "keypoint confidence")
        if abs(x) > MAX_KEYPOINT_COORDINATE or abs(y) > MAX_KEYPOINT_COORDINATE:
            raise ValueError(
                f"keypoint coordinates cannot exceed {MAX_KEYPOINT_COORDINATE:g} in absolute value"
            )
        if not 0.0 < confidence <= 1.0:
            raise ValueError("keypoint confidence must be greater than 0 and at most 1")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "confidence", confidence)

    def to_record(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass(frozen=True)
class PoseSkeleton:
    """An ordered keypoint set with explicit topology and subject identity."""

    kind: str
    layout: str
    subject: int
    normalized: bool
    keypoints: tuple[PoseKeypoint | None, ...]
    edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in POSE_KINDS:
            raise ValueError(f"unknown pose skeleton kind: {self.kind}")
        if type(self.layout) is not str or not self.layout or len(self.layout) > 128:
            raise ValueError("pose skeleton layout must be a string of 1 to 128 characters")
        if type(self.subject) is not int or self.subject < 0:
            raise ValueError("pose skeleton subject must be a non-negative integer")
        if type(self.normalized) is not bool:
            raise TypeError("pose skeleton normalized must be a boolean")
        raw_keypoints = cast("object", self.keypoints)
        if not isinstance(raw_keypoints, Sequence) or isinstance(raw_keypoints, (str, bytes)):
            raise TypeError("pose skeleton keypoints must be a sequence")
        keypoints = tuple(cast("Sequence[object]", raw_keypoints))
        if not 1 <= len(keypoints) <= MAX_SKELETON_KEYPOINTS:
            raise ValueError(
                f"pose skeleton must contain between 1 and {MAX_SKELETON_KEYPOINTS} keypoints"
            )
        if any(point is not None and not isinstance(point, PoseKeypoint) for point in keypoints):
            raise TypeError("pose skeleton keypoints must be PoseKeypoint values or null")

        raw_edges = cast("object", self.edges)
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
            raise TypeError("pose skeleton edges must be a sequence")
        edge_items = cast("Sequence[object]", raw_edges)
        if len(edge_items) > MAX_SKELETON_EDGES:
            raise ValueError(f"pose skeleton cannot exceed {MAX_SKELETON_EDGES} edges")
        edges: list[tuple[int, int]] = []
        for index, raw_edge in enumerate(edge_items):
            if not isinstance(raw_edge, Sequence) or isinstance(raw_edge, (str, bytes)):
                raise TypeError(f"pose skeleton edge {index} must contain two keypoint indexes")
            edge = cast("Sequence[object]", raw_edge)
            if len(edge) != 2:
                raise TypeError(f"pose skeleton edge {index} must contain two keypoint indexes")
            if any(type(value) is not int for value in edge):
                raise TypeError(f"pose skeleton edge {index} indexes must be integers")
            start, end = cast("tuple[int, int]", tuple(edge))
            if start == end or not 0 <= start < len(keypoints) or not 0 <= end < len(keypoints):
                raise ValueError(f"pose skeleton edge {index} has invalid keypoint indexes")
            edges.append((start, end))
        if len(set(edges)) != len(edges):
            raise ValueError("pose skeleton edges must be unique")
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "edges", tuple(edges))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "layout": self.layout,
            "subject": self.subject,
            "normalized": self.normalized,
            "keypoints": [
                point.to_record() if point is not None else None for point in self.keypoints
            ],
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class Pose:
    """One frame of editable pose annotations in a declared canvas space."""

    canvas_width: int
    canvas_height: int
    skeletons: tuple[PoseSkeleton, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("canvas width", self.canvas_width),
            ("canvas height", self.canvas_height),
        ):
            if type(value) is not int or not 1 <= value <= MAX_POSE_DIMENSION:
                raise ValueError(f"pose {name} must be between 1 and {MAX_POSE_DIMENSION}")
        raw_skeletons = cast("object", self.skeletons)
        if not isinstance(raw_skeletons, Sequence) or isinstance(raw_skeletons, (str, bytes)):
            raise TypeError("pose skeletons must be a sequence")
        skeletons = tuple(cast("Sequence[object]", raw_skeletons))
        if len(skeletons) > MAX_POSE_SKELETONS:
            raise ValueError(f"pose cannot exceed {MAX_POSE_SKELETONS} skeletons")
        if any(not isinstance(skeleton, PoseSkeleton) for skeleton in skeletons):
            raise TypeError("pose skeletons must be PoseSkeleton values")
        object.__setattr__(self, "skeletons", skeletons)

    def to_record(self) -> dict[str, object]:
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "skeletons": [skeleton.to_record() for skeleton in self.skeletons],
        }


def _coerce_pose_keypoint(obj: object, subject: str) -> PoseKeypoint | None:
    if obj is None or isinstance(obj, PoseKeypoint):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"{subject} must be an object or null")
    record = cast("Mapping[str, object]", obj)
    if set(record) != {"x", "y", "confidence"}:
        raise ValueError(f"{subject} requires exactly x, y, and confidence")
    return PoseKeypoint(
        x=cast("float", record["x"]),
        y=cast("float", record["y"]),
        confidence=cast("float", record["confidence"]),
    )


def _coerce_pose_skeleton(obj: object, index: int) -> PoseSkeleton:
    if isinstance(obj, PoseSkeleton):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"{POSE_TYPE} skeleton {index} must be an object")
    record = cast("Mapping[str, object]", obj)
    required = {"kind", "layout", "subject", "normalized", "keypoints", "edges"}
    if set(record) != required:
        raise ValueError(
            f"{POSE_TYPE} skeleton {index} requires exactly {', '.join(sorted(required))}"
        )
    raw_keypoints = record["keypoints"]
    if not isinstance(raw_keypoints, Sequence) or isinstance(raw_keypoints, (str, bytes)):
        raise TypeError(f"{POSE_TYPE} skeleton {index} keypoints must be an array")
    raw_edges = record["edges"]
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise TypeError(f"{POSE_TYPE} skeleton {index} edges must be an array")
    return PoseSkeleton(
        kind=cast("str", record["kind"]),
        layout=cast("str", record["layout"]),
        subject=cast("int", record["subject"]),
        normalized=cast("bool", record["normalized"]),
        keypoints=tuple(
            _coerce_pose_keypoint(point, f"{POSE_TYPE} skeleton {index} keypoint {point_index}")
            for point_index, point in enumerate(cast("Sequence[object]", raw_keypoints))
        ),
        edges=tuple(cast("Sequence[tuple[int, int]]", raw_edges)),
    )


def _coerce_pose(obj: object) -> Pose:
    if isinstance(obj, Pose):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"{POSE_TYPE} expects an object, got {type(obj).__name__}")
    record = cast("Mapping[str, object]", obj)
    if set(record) != {"canvas_width", "canvas_height", "skeletons"}:
        raise ValueError(f"{POSE_TYPE} requires exactly canvas_width, canvas_height, and skeletons")
    raw_skeletons = record["skeletons"]
    if not isinstance(raw_skeletons, Sequence) or isinstance(raw_skeletons, (str, bytes)):
        raise TypeError(f"{POSE_TYPE} skeletons must be an array")
    return Pose(
        canvas_width=cast("int", record["canvas_width"]),
        canvas_height=cast("int", record["canvas_height"]),
        skeletons=tuple(
            _coerce_pose_skeleton(skeleton, index)
            for index, skeleton in enumerate(cast("Sequence[object]", raw_skeletons))
        ),
    )


def _encode_pose(obj: object) -> bytes:
    return json.dumps(
        _coerce_pose(obj).to_record(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _decode_pose(data: bytes) -> object:
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid {POSE_TYPE} payload") from exc
    return _coerce_pose(record)


def _pose_meta(obj: object) -> Mapping[str, object]:
    pose = _coerce_pose(obj)
    return {
        "canvas_width": pose.canvas_width,
        "canvas_height": pose.canvas_height,
        "skeletons": len(pose.skeletons),
        "keypoints": sum(len(skeleton.keypoints) for skeleton in pose.skeletons),
    }


def register_image_types(registry: TypeRegistry) -> None:
    from dinkster_api.v1 import (
        ASSET_TYPE,
        PNG_CONTAINER_VERSION,
        SAVE_TARGET_TYPE,
        AssetRef,
        decode_image_array,
        encode_image_array,
        image_array_fingerprint,
        image_array_meta,
        mask_array_meta,
        prepare_image_array_encoding,
        register_asset_type,
        register_save_target_type,
        render_image_png,
        render_mask_png,
        resolver_from_env,
        validate_image_encoded,
    )
    from dinkster_image_document.compat import register_comfy_compositor, register_comfy_layers
    from dinkster_image_document.document import ImageDocument
    from dinkster_image_document.format import IMAGE_DOCUMENT_MEDIA_TYPE

    from .layer_document import decode_layers, layer_meta, migrate_layers
    from .layer_recipe import coerce_delta, decode_delta, encode_delta

    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    for type_id, renderer in (
        ("dinkster.image", render_image_png),
        ("dinkster.mask", render_mask_png),
    ):
        if type_id not in registry:
            registry.register(
                type_id,
                encode=encode_image_array,
                decode=decode_image_array,
                prepare_buffer_encoding=prepare_image_array_encoding,
                fingerprint=image_array_fingerprint(type_id),
                meta=mask_array_meta if type_id == "dinkster.mask" else image_array_meta,
                validate_encoded=validate_image_encoded,
                validate_encoded_buffer=validate_image_encoded,
            )
            registry.register_rendition(
                type_id,
                "png",
                mime="image/png",
                render=renderer,
                version=PNG_CONTAINER_VERSION,
            )

    def bind(wire: dict[str, object]) -> AssetRef:
        coerce = registry.spec("dinkster.asset").coerce if "dinkster.asset" in registry else None
        return (
            cast(AssetRef, coerce(wire))
            if coerce
            else AssetRef.from_wire(wire, resolver_from_env())
        )

    def coerce_document(obj: object) -> ImageDocument:
        document = migrate_layers(obj)
        return ImageDocument(document.data, document.bind or bind)

    if REGION_TYPE not in registry:
        registry.register(
            REGION_TYPE,
            encode=encode_region,
            decode=decode_region,
            coerce=coerce_region,
            meta=region_meta,
        )
    if DETECTION_TYPE not in registry:
        registry.register(
            DETECTION_TYPE,
            encode=encode_detection,
            decode=decode_detection,
            coerce=coerce_detection,
            meta=detection_meta,
        )
    if POSE_TYPE not in registry:
        registry.register(
            POSE_TYPE,
            encode=_encode_pose,
            decode=_decode_pose,
            coerce=_coerce_pose,
            meta=_pose_meta,
        )
    if LAYERS_TYPE not in registry:
        registry.register(
            LAYERS_TYPE,
            encode=lambda obj: migrate_layers(obj).data,
            decode=lambda data: coerce_document(decode_layers(data)),
            coerce=coerce_document,
            meta=layer_meta,
        )
        registry.register_rendition(
            LAYERS_TYPE,
            "image",
            mime="image/png",
            render=lambda obj: migrate_layers(obj).render().png,
        )
        registry.register_rendition(
            LAYERS_TYPE,
            "document",
            mime=IMAGE_DOCUMENT_MEDIA_TYPE,
            render=lambda obj: migrate_layers(obj).data,
        )
    if COMPOSITOR_TYPE not in registry:
        registry.register(
            COMPOSITOR_TYPE,
            encode=encode_delta,
            decode=decode_delta,
            coerce=coerce_delta,
            meta=lambda obj: {"version": coerce_delta(obj)["version"]},
        )
    register_comfy_layers(registry)
    register_comfy_compositor(registry)
    for type_id in ("comfy.BOUNDING_BOX", "comfy.ARRAY"):
        if type_id not in registry:
            registry.register(type_id)


__all__ = [
    "DETECTION_TYPE",
    "COMPOSITOR_TYPE",
    "LAYERS_TYPE",
    "POSE_KINDS",
    "POSE_TYPE",
    "REGION_TYPE",
    "RESERVED_TYPE_IDS",
    "SEGMENTATION_TYPE",
    "VECTOR_TYPE",
    "Detection",
    "Pose",
    "PoseKeypoint",
    "PoseSkeleton",
    "Region",
    "register_image_types",
]
