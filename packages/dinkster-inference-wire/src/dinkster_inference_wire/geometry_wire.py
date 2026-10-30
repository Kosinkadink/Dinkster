"""Portable mesh values using the structural tensor envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, cast

from dinkster_values import (
    MESH_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

from .media_wire import materialize_latent
from .sparse import TriangleMeshBatch


def _decode(data: bytes) -> TriangleMeshBatch[Any]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"} or not isinstance(frame["samples"], dict):
        raise ValueError("MESH frame must contain a field mapping")
    values = cast("dict[str, Any]", frame["samples"])
    if set(values) != {field.name for field in fields(TriangleMeshBatch)}:
        raise ValueError("MESH fields do not match its runtime value")
    tensor_fields = tuple(
        field.name
        for field in fields(TriangleMeshBatch)
        if field.name not in {"unlit", "occlusion_in_mr", "material"}
    )
    required = {"vertices", "faces"}
    for name in tensor_fields:
        value = values[name]
        if value is None and name not in required:
            continue
        if not isinstance(value, EncodedLatentTensor):
            raise ValueError(f"MESH {name} must be a tensor")
    if any(type(values[name]) is not bool for name in ("unlit", "occlusion_in_mr")):
        raise ValueError("MESH material flags must be booleans")
    return TriangleMeshBatch(**values)


def _encode(value: object) -> bytes:
    if type(value) is not TriangleMeshBatch:
        raise TypeError("MESH requires TriangleMeshBatch")
    value = cast("TriangleMeshBatch[Any]", value)
    data = encode_latent(
        {"samples": {field.name: getattr(value, field.name) for field in fields(value)}}
    )
    _decode(data)
    return data


def register_mesh_type(registry: TypeRegistry) -> TypeSpec:
    if MESH_TYPE in registry:
        return registry.spec(MESH_TYPE)

    def validate(data: bytes, _metadata: Mapping[str, object]) -> None:
        _decode(data)

    return registry.register(
        MESH_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=lambda value: stable_hash([MESH_TYPE.encode("ascii"), _encode(value)]),
        validate_encoded=validate,
    )


def materialize_mesh(value: object) -> TriangleMeshBatch[Any]:
    """Reconstruct mesh tensors at a consumer, preserving live storage and metadata."""
    if not isinstance(value, TriangleMeshBatch):
        raise TypeError("mesh value must be a TriangleMeshBatch")
    mesh = cast("TriangleMeshBatch[Any]", value)
    values = {field.name: getattr(mesh, field.name) for field in fields(mesh)}
    converted = {name: materialize_latent(item) for name, item in values.items()}
    if all(converted[name] is item for name, item in values.items()):
        return mesh
    return TriangleMeshBatch(**converted)
