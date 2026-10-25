"""Structural values and resident dependencies used by native hook nodes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_values import (
    FLOATS_TYPE,
    TIMESTEPS_RANGE_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    resident_owners,
    stable_hash,
)

_TENSOR_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"}
)


def normalize_hook_owners(group: object) -> object:
    """Declare the keyframe groups actually retained by this hook group."""
    hooks = getattr(group, "hooks", None)
    if hooks is None:
        return group
    refs = getattr(group, "_dinkster_resident_refs", ())
    if not isinstance(refs, tuple):
        raise TypeError("hook resident references must be a tuple")
    previous = cast("tuple[object, ...]", getattr(group, "_dinkster_hook_keyframe_refs", ()))
    # Derived references are replaced, not accumulated across keyframe changes.
    explicit: list[object] = []
    for ref in cast("tuple[object, ...]", refs):
        if not any(ref is old for old in (*previous, *explicit)):
            explicit.append(ref)
    retained: list[object] = []
    for hook in cast("list[Any]", hooks):
        keyframes = cast("object", hook.hook_keyframe)
        if not any(keyframes is ref for ref in (*explicit, *retained)):
            retained.append(keyframes)
    cast("Any", group)._dinkster_resident_refs = (*explicit, *retained)
    cast("Any", group)._dinkster_hook_keyframe_refs = tuple(retained)
    return group


def normalize_clip_hook_owners(clip: object) -> object:
    """Declare only hook groups actually retained by the CLIP and its patcher."""
    groups = (
        getattr(clip, "apply_hooks_to_conds", None),
        getattr(getattr(clip, "patcher", None), "forced_hooks", None),
    )
    previous = cast("tuple[object, ...]", getattr(clip, "_dinkster_clip_hook_refs", ()))
    if all(group is None for group in groups) and not previous:
        return clip
    refs = getattr(clip, "_dinkster_resident_refs", ())
    if not isinstance(refs, tuple):
        raise TypeError("CLIP resident references must be a tuple")
    explicit: list[object] = []
    for ref in cast("tuple[object, ...]", refs):
        if not any(ref is old for old in (*previous, *explicit)):
            explicit.append(ref)
    retained: list[object] = []
    for group in groups:
        if group is None:
            continue
        normalize_hook_owners(group)
        for ref in (group, *resident_owners(group)):
            if not any(ref is old for old in (*explicit, *retained)):
                retained.append(ref)
    cast("Any", clip)._dinkster_resident_refs = (*explicit, *retained)
    cast("Any", clip)._dinkster_clip_hook_refs = tuple(retained)
    return clip


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
