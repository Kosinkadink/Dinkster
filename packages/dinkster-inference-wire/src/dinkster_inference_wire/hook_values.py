"""Portable structural values used by native hook nodes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from dinkster_values import (
    FLOATS_TYPE,
    TIMESTEPS_RANGE_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

_TENSOR_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"}
)


def _number(value: object) -> int | float:
    if type(value) not in (int, float):
        raise TypeError("hook structural values require exact int or float values")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("hook structural values must be finite")
    return cast("int | float", value)


def _floats(value: object) -> object:
    if type(value) in (int, float):
        return _number(value)
    if type(value) in (list, tuple, range):
        values = tuple(
            _number(item) for item in cast("list[object] | tuple[object, ...] | range", value)
        )
        return values
    if getattr(value, "shape", None) == ():
        raise TypeError("FLOATS requires a one-dimensional numeric tensor-like value")
    try:
        frame = cast("dict[str, object]", decode_latent(encode_latent({"samples": value})))
    except (TypeError, ValueError) as exc:
        raise TypeError("FLOATS requires a sized numeric sequence") from exc
    tensor = frame.get("samples")
    if (
        type(tensor) is not EncodedLatentTensor
        or len(tensor.shape) != 1
        or tensor.dtype not in _TENSOR_DTYPES
    ):
        raise TypeError("FLOATS requires a one-dimensional numeric tensor-like value")
    return tensor


def _timesteps_range(value: object) -> tuple[int | float, int | float]:
    if type(value) not in (list, tuple):
        raise TypeError("TIMESTEPS_RANGE requires an ordered pair")
    values = cast("list[object] | tuple[object, ...]", value)
    if len(values) != 2:
        raise ValueError("TIMESTEPS_RANGE requires exactly two endpoints")
    return (_number(values[0]), _number(values[1]))


def _normalize(type_id: str, value: object) -> object:
    if type_id == FLOATS_TYPE:
        return _floats(value)
    if type_id == TIMESTEPS_RANGE_TYPE:
        return _timesteps_range(value)
    raise ValueError(f"unknown hook structural type: {type_id}")


def _encode(type_id: str, value: object) -> bytes:
    normalized = _normalize(type_id, value)
    return encode_latent({"samples": normalized})


def _decode(type_id: str, data: bytes) -> object:
    frame = decode_latent(data)
    if type(frame) is not dict:
        raise ValueError(f"{type_id} frame must contain only samples")
    payload = cast("dict[str, object]", frame)
    if set(payload) != {"samples"}:
        raise ValueError(f"{type_id} frame must contain only samples")
    value = _normalize(type_id, payload["samples"])
    if _encode(type_id, value) != data:
        raise ValueError(f"{type_id} frame must use canonical encoding")
    return value


def register_hook_structural_type(registry: TypeRegistry, type_id: str) -> TypeSpec:
    """Register one canonical structural hook value codec."""
    if type_id not in (FLOATS_TYPE, TIMESTEPS_RANGE_TYPE):
        raise ValueError(f"unknown hook structural type: {type_id}")
    if type_id in registry:
        return registry.spec(type_id)

    def encode(value: object) -> bytes:
        return _encode(type_id, value)

    def validate(data: bytes, _metadata: Mapping[str, object]) -> None:
        _decode(type_id, data)

    return registry.register(
        type_id,
        encode=encode,
        decode=lambda data: _decode(type_id, data),
        fingerprint=lambda value: stable_hash([type_id.encode("utf-8"), encode(value)]),
        validate_encoded=validate,
    )


__all__ = ["register_hook_structural_type"]
