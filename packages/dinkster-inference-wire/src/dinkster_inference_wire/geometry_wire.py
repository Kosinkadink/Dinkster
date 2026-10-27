"""Portable mesh and voxel values using the structural tensor envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, cast

from dinkster_values import (
    MESH_TYPE,
    VOXEL_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

from .media_wire import materialize_latent
from .sparse import TriangleMeshBatch, Voxel


def _decode(data: bytes, type_id: str) -> TriangleMeshBatch[Any] | Voxel[Any]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"} or not isinstance(frame["samples"], dict):
        raise ValueError(f"{type_id} frame must contain a field mapping")
    values = cast("dict[str, Any]", frame["samples"])
    runtime = TriangleMeshBatch if type_id == MESH_TYPE else Voxel
    if set(values) != {field.name for field in fields(runtime)}:
        raise ValueError(f"{type_id} fields do not match its runtime value")
    tensor_fields = (
        tuple(
            field.name
            for field in fields(TriangleMeshBatch)
            if field.name not in {"unlit", "occlusion_in_mr", "material"}
        )
        if type_id == MESH_TYPE
        else ("data", "voxel_colors")
    )
    required = {"vertices", "faces"} if type_id == MESH_TYPE else {"data"}
    for name in tensor_fields:
        value = values[name]
        if value is None and name not in required:
            continue
        if not isinstance(value, EncodedLatentTensor):
            raise ValueError(f"{type_id} {name} must be a tensor")
    if type_id == MESH_TYPE:
        if any(type(values[name]) is not bool for name in ("unlit", "occlusion_in_mr")):
            raise ValueError("MESH material flags must be booleans")
        return TriangleMeshBatch(**values)
    return Voxel(**values)


def _encode(value: object, type_id: str) -> bytes:
    runtime = TriangleMeshBatch if type_id == MESH_TYPE else Voxel
    if type(value) is not runtime:
        raise TypeError(f"{type_id} requires {runtime.__name__}")
    value = cast("TriangleMeshBatch[Any] | Voxel[Any]", value)
    data = encode_latent(
        {"samples": {field.name: getattr(value, field.name) for field in fields(runtime)}}
    )
    _decode(data, type_id)
    return data


def register_geometry_type(registry: TypeRegistry, type_id: str) -> TypeSpec:
    if type_id not in (MESH_TYPE, VOXEL_TYPE):
        raise ValueError(f"unsupported geometry type: {type_id}")

    def encode(value: object) -> bytes:
        return _encode(value, type_id)

    def validate(data: bytes, _metadata: Mapping[str, object]) -> None:
        _decode(data, type_id)

    return registry.register(
        type_id,
        encode=encode,
        decode=lambda data: _decode(data, type_id),
        fingerprint=lambda value: stable_hash([type_id.encode("ascii"), encode(value)]),
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


def materialize_voxel(value: object) -> Voxel[Any]:
    """Reconstruct dense or sparse voxel tensors without imposing a grid layout."""
    if not isinstance(value, Voxel):
        raise TypeError("voxel value must be a Voxel")
    voxel = cast("Voxel[Any]", value)
    values = {field.name: getattr(voxel, field.name) for field in fields(voxel)}
    converted = {name: materialize_latent(item) for name, item in values.items()}
    if all(converted[name] is item for name, item in values.items()):
        return voxel
    return Voxel(**converted)
