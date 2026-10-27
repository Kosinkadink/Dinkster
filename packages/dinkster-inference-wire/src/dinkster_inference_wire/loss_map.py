"""Portable training loss records using the shared latent structure framing."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_values import (
    LOSS_MAP_TYPE,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)


def _decode(data: bytes) -> dict[str, object]:
    frame = decode_latent(data)
    if type(frame) is not dict or set(frame) != {"samples"}:
        raise ValueError("LOSS_MAP frame must contain only samples")
    value = frame["samples"]
    if type(value) is not dict:
        raise ValueError("LOSS_MAP samples must be a mapping")
    loss_map = value
    if type(loss_map.get("loss")) is not list:
        raise ValueError("LOSS_MAP loss must be an ordered list")
    return loss_map


def _encode(value: object) -> bytes:
    if type(value) is not dict:
        raise TypeError("LOSS_MAP requires a mapping")
    data = encode_latent({"samples": value})
    _decode(data)
    return data


def _fingerprint(value: object) -> str:
    return stable_hash([LOSS_MAP_TYPE.encode("utf-8"), _encode(value)])


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def register_loss_map_type(registry: TypeRegistry) -> TypeSpec:
    """Register the canonical LOSS_MAP mapping transport."""
    if LOSS_MAP_TYPE in registry:
        return registry.spec(LOSS_MAP_TYPE)
    return registry.register(
        LOSS_MAP_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=_fingerprint,
        validate_encoded=_validate,
    )


__all__ = ["register_loss_map_type"]
