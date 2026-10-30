"""Portable 3D cameras and model transforms in the structural latent envelope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_values import (
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


def _validate_vector(
    value: object, fields: tuple[str, ...], name: str, type_id: str = LOAD3D_CAMERA_TYPE
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{type_id} {name} must be a numeric mapping")
    mapping = cast("Mapping[str, object]", value)
    if any(not _is_number(mapping.get(field)) for field in fields):
        raise ValueError(f"{type_id} {name} must be a numeric mapping")


def _validate_camera(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("LOAD3D_CAMERA must be a mapping")
    camera = cast("dict[str, object]", value)
    _validate_vector(camera.get("position"), ("x", "y", "z"), "position")
    _validate_vector(camera.get("target"), ("x", "y", "z"), "target")
    if not _is_number(camera.get("zoom")):
        raise ValueError("LOAD3D_CAMERA zoom must be numeric")
    if type(camera.get("cameraType")) is not str:
        raise ValueError("LOAD3D_CAMERA cameraType must be a string")
    if "quaternion" in camera:
        _validate_vector(camera["quaternion"], ("x", "y", "z", "w"), "quaternion")
    if "useCustomUp" in camera and type(camera["useCustomUp"]) is not bool:
        raise ValueError("LOAD3D_CAMERA useCustomUp must be boolean")
    for field in ("fov", "aspect", "near", "far"):
        if field in camera and not _is_number(camera[field]):
            raise ValueError(f"LOAD3D_CAMERA {field} must be numeric")
    if "frustum" in camera:
        _validate_vector(camera["frustum"], ("left", "right", "top", "bottom"), "frustum")
    return camera


def _decode(data: bytes) -> dict[str, object]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError("LOAD3D_CAMERA frame must contain only samples")
    return _validate_camera(frame["samples"])


def _encode(value: object) -> bytes:
    data = encode_latent({"samples": value})
    _decode(data)
    return data


def _fingerprint(value: object) -> str:
    return stable_hash([LOAD3D_CAMERA_TYPE.encode("utf-8"), _encode(value)])


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def register_load3d_camera_type(registry: TypeRegistry) -> TypeSpec:
    return registry.register(
        LOAD3D_CAMERA_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=_fingerprint,
        validate_encoded=_validate,
    )


def _decode_model_info(data: bytes) -> list[dict[str, object]]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError("LOAD3D_MODEL_INFO frame must contain only samples")
    models = frame["samples"]
    if type(models) is not list:
        raise ValueError("LOAD3D_MODEL_INFO must be a list of transforms")
    for model in cast("list[object]", models):
        if type(model) is not dict:
            raise ValueError("LOAD3D_MODEL_INFO transform must be a mapping")
        transform = cast("dict[str, object]", model)
        for field, axes in (
            ("position", ("x", "y", "z")),
            ("quaternion", ("x", "y", "z", "w")),
            ("scale", ("x", "y", "z")),
        ):
            _validate_vector(transform.get(field), axes, field, LOAD3D_MODEL_INFO_TYPE)
    return cast("list[dict[str, object]]", models)


def _encode_model_info(value: object) -> bytes:
    data = encode_latent({"samples": value})
    _decode_model_info(data)
    return data


def _model_info_fingerprint(value: object) -> str:
    return stable_hash([LOAD3D_MODEL_INFO_TYPE.encode("utf-8"), _encode_model_info(value)])


def _validate_model_info(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode_model_info(data)


def register_load3d_model_info_type(registry: TypeRegistry) -> TypeSpec:
    return registry.register(
        LOAD3D_MODEL_INFO_TYPE,
        encode=_encode_model_info,
        decode=_decode_model_info,
        fingerprint=_model_info_fingerprint,
        validate_encoded=_validate_model_info,
    )
