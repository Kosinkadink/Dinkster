"""Portable source-shaped pose frames without coordinate or score normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_values import (
    POSE_KEYPOINT_TYPE,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

_KEYPOINT_FIELDS = (
    "pose_keypoints_2d",
    "foot_keypoints_2d",
    "face_keypoints_2d",
    "hand_right_keypoints_2d",
    "hand_left_keypoints_2d",
)


def _validate_frames(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise ValueError("POSE_KEYPOINT must be a list of frame mappings")
    frames = cast("list[dict[str, object]]", value)
    for frame in frames:
        if type(frame) is not dict or any(
            type(frame.get(field)) is not int for field in ("canvas_width", "canvas_height")
        ):
            raise ValueError("POSE_KEYPOINT frames require integer canvas dimensions")
        people = frame.get("people")
        if type(people) is not list:
            raise ValueError("POSE_KEYPOINT frames require a people list")
        for person in cast("list[dict[str, object]]", people):
            if type(person) is not dict:
                raise ValueError("POSE_KEYPOINT people must be mappings")
            for field in _KEYPOINT_FIELDS:
                flat = person.get(field)
                if flat is None:
                    continue
                if (
                    type(flat) is not list
                    or len(cast("list[object]", flat)) % 3
                    or any(
                        type(number) not in (int, float) for number in cast("list[object]", flat)
                    )
                ):
                    raise ValueError(f"POSE_KEYPOINT {field} must contain flat numeric triples")
    return frames


def _decode(data: bytes) -> list[dict[str, object]]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError("POSE_KEYPOINT frame must contain only samples")
    return _validate_frames(frame["samples"])


def _encode(value: object) -> bytes:
    _validate_frames(value)
    return encode_latent({"samples": value})


def _fingerprint(value: object) -> str:
    return stable_hash([POSE_KEYPOINT_TYPE.encode("utf-8"), _encode(value)])


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def register_pose_keypoint_type(registry: TypeRegistry) -> TypeSpec:
    return registry.register(
        POSE_KEYPOINT_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=_fingerprint,
        validate_encoded=_validate,
    )
