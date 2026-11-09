"""Structural vision packets retain NumPy arrays, tensors and face topology."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from dinkster_values import (
    DA3_GEOMETRY_TYPE,
    FACE_LANDMARKS_TYPE,
    KIMODO_POSE_DATA_TYPE,
    MHR_POSE_DATA_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

from .media_wire import materialize_latent

_PACKET_TYPES = (DA3_GEOMETRY_TYPE, FACE_LANDMARKS_TYPE, MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)


@dataclass(frozen=True)
class EncodedNumpyArray(EncodedLatentTensor):
    scalar: bool = False


def _pack(value: Any) -> object:
    if isinstance(value, EncodedNumpyArray):
        return (
            "numpy-scalar" if value.scalar else "numpy",
            EncodedLatentTensor(value.dtype, value.shape, value.data),
        )
    if type(value).__module__.split(".")[0] == "numpy":
        np = importlib.import_module("numpy")
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            record = EncodedLatentTensor(str(array.dtype), tuple(array.shape), array.tobytes())
            return ("numpy-scalar" if isinstance(value, np.generic) else "numpy", record)
    if isinstance(value, EncodedLatentTensor) or hasattr(value, "detach"):
        return ("tensor", value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is dict:
        return (
            "mapping",
            {key: _pack(item) for key, item in cast("dict[str, Any]", value).items()},
        )
    if type(value) in (list, tuple):
        items = cast("list[Any] | tuple[Any, ...]", value)
        return ("list" if type(value) is list else "tuple", [_pack(item) for item in items])
    if type(value) is frozenset:
        edges = cast("frozenset[Any]", value)
        if any(
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(index) is not int or index < 0 for index in edge)
            for edge in edges
        ):
            raise ValueError("vision topology requires nonnegative integer edge pairs")
        return ("topology", sorted(edges))
    raise TypeError(f"unsupported vision packet value: {type(value).__name__}")


def _unpack(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is not tuple or len(cast("tuple[Any, ...]", value)) != 2:
        raise ValueError("vision packet contains an invalid structural record")
    kind, payload = cast("tuple[Any, Any]", value)
    if type(kind) is not str:
        raise ValueError("vision packet contains an invalid structural record")
    if kind == "mapping" and type(payload) is dict:
        return {key: _unpack(item) for key, item in cast("dict[str, Any]", payload).items()}
    if kind in {"list", "tuple"} and type(payload) is list:
        items = [_unpack(item) for item in cast("list[Any]", payload)]
        return items if kind == "list" else tuple(items)
    if kind == "topology" and type(payload) is list:
        if any(
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(index) is not int or index < 0 for index in edge)
            for edge in cast("list[Any]", payload)
        ):
            raise ValueError("vision packet contains invalid topology")
        result = frozenset(cast("list[Any]", payload))
        if len(result) != len(cast("list[Any]", payload)):
            raise ValueError("vision packet contains duplicate topology edges")
        return result
    if kind == "tensor" and isinstance(payload, EncodedLatentTensor):
        return payload
    if kind in {"numpy", "numpy-scalar"} and isinstance(payload, EncodedLatentTensor):
        if payload.dtype == "bfloat16":
            raise ValueError("vision NumPy arrays cannot use bfloat16")
        if kind == "numpy-scalar" and payload.shape:
            raise ValueError("vision NumPy scalar must have scalar shape")
        return EncodedNumpyArray(payload.dtype, payload.shape, payload.data, kind == "numpy-scalar")
    raise ValueError("vision packet contains an invalid structural record")


def _validate_packet(type_id: str, value: Any) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{type_id} requires a packet mapping")
    packet = cast("dict[str, Any]", value)
    if type_id == DA3_GEOMETRY_TYPE:
        if not {"depth", "image", "mode"} <= packet.keys() or packet["mode"] not in {
            "mono",
            "multiview",
        }:
            raise ValueError("DA3_GEOMETRY requires depth, source image and mode")
    else:
        if not {"frames", "image_size"} <= packet.keys() or type(packet["frames"]) is not list:
            raise ValueError(f"{type_id} requires frames and image_size")
        size = packet["image_size"]
        if (
            type(size) is not tuple
            or len(cast("tuple[Any, ...]", size)) != 2
            or any(type(n) is not int or n <= 0 for n in cast("tuple[Any, ...]", size))
        ):
            raise ValueError(f"{type_id} requires positive integer image_size")
        if any(
            type(frame) is not list or any(type(p) is not dict for p in frame)
            for frame in cast("list[Any]", packet["frames"])
        ):
            raise ValueError(f"{type_id} frames must contain ordered detection mappings")
        if type_id == FACE_LANDMARKS_TYPE:
            sets = packet.get("connection_sets")
            if type(sets) is not dict or any(
                type(edges) is not frozenset for edges in cast("dict[str, Any]", sets).values()
            ):
                raise ValueError("FACE_LANDMARKS requires connection_sets topology")
        elif "faces" not in packet:
            raise ValueError(f"{type_id} requires faces")
    return cast("dict[str, object]", value)


def _decode(type_id: str, data: bytes) -> dict[str, object]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError(f"{type_id} frame must contain only samples")
    return _validate_packet(type_id, _unpack(frame["samples"]))


def _encode(type_id: str, value: object) -> bytes:
    _validate_packet(type_id, value)
    data = encode_latent({"samples": _pack(value)})
    _decode(type_id, data)
    return data


def _validate_encoded(type_id: str, data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(type_id, data)


def _fingerprint(type_id: str, value: object) -> str:
    return stable_hash([type_id.encode("ascii"), _encode(type_id, value)])


def register_vision_packet_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    for type_id in _PACKET_TYPES:
        if type_id not in registry:
            registry.register(
                type_id,
                encode=partial(_encode, type_id),
                decode=partial(_decode, type_id),
                fingerprint=partial(_fingerprint, type_id),
                validate_encoded=partial(_validate_encoded, type_id),
            )
    return tuple(registry.spec(type_id) for type_id in _PACKET_TYPES)


def materialize_vision_packet(value: object) -> Any:
    """Restore tensors at the consumer; NumPy arrays and live tensors keep their kinds."""
    if isinstance(value, EncodedNumpyArray):
        np = importlib.import_module("numpy")
        array = np.frombuffer(value.data, dtype=value.dtype).reshape(value.shape).copy()
        return array[()] if value.scalar else array
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        converted = {key: materialize_vision_packet(item) for key, item in mapping.items()}
        return (
            mapping if all(converted[key] is item for key, item in mapping.items()) else converted
        )
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        converted = [materialize_vision_packet(item) for item in sequence]
        if all(left is right for left, right in zip(sequence, converted, strict=True)):
            return sequence
        return tuple(converted) if isinstance(value, tuple) else converted
    return materialize_latent(value)
