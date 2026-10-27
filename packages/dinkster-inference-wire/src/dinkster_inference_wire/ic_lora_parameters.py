"""Portable IC-LoRA parameter mappings using the latent structural envelope."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import cast

from dinkster_values import (
    IC_LORA_PARAMETERS_TYPE,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)


def _decode(data: bytes, *, type_id: str) -> object:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError(f"{type_id} frame must contain only samples")
    pairs = frame["samples"]
    if not isinstance(pairs, list):
        raise ValueError(f"{type_id} must contain ordered mapping entries")
    result: dict[str, object] = {}
    for item in cast("list[object]", pairs):
        pair = cast("tuple[object, ...]", item)
        if len(pair) != 2 or not isinstance(pair[0], str):
            raise ValueError(f"{type_id} entries must be string-keyed pairs")
        key = pair[0]
        value = pair[1]
        if key in result:
            raise ValueError(f"{type_id} entries must have unique keys")
        result[key] = value
    return result


def _encode(value: object, *, type_id: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError(f"{type_id} must be a mapping")
    mapping = cast("Mapping[str, object]", value)
    data = encode_latent({"samples": list(mapping.items())})
    _decode(data, type_id=type_id)
    return data


def _fingerprint(value: object, *, type_id: str) -> str:
    return stable_hash([type_id.encode("utf-8"), _encode(value, type_id=type_id)])


def _validate(data: bytes, _metadata: Mapping[str, object], *, type_id: str) -> None:
    _decode(data, type_id=type_id)


def register_ic_lora_parameters_type(registry: TypeRegistry) -> TypeSpec:
    """Register the canonical portable IC-LoRA parameter mapping."""
    return registry.register(
        IC_LORA_PARAMETERS_TYPE,
        encode=partial(_encode, type_id=IC_LORA_PARAMETERS_TYPE),
        decode=partial(_decode, type_id=IC_LORA_PARAMETERS_TYPE),
        fingerprint=partial(_fingerprint, type_id=IC_LORA_PARAMETERS_TYPE),
        validate_encoded=partial(_validate, type_id=IC_LORA_PARAMETERS_TYPE),
    )


__all__ = ["register_ic_lora_parameters_type"]
