"""Portable packed SAM3 tracks using the existing structural tensor envelope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_values import (
    SAM3_TRACK_DATA_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)


def _decode(data: bytes) -> dict[str, object]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"} or not isinstance(frame["samples"], dict):
        raise ValueError("SAM3_TRACK_DATA frame must contain a track mapping")
    value = cast("dict[str, object]", frame["samples"])
    if "packed_masks" not in value:
        raise ValueError("SAM3_TRACK_DATA requires packed_masks")
    packed = value["packed_masks"]
    if packed is not None and (
        not isinstance(packed, EncodedLatentTensor) or packed.dtype != "uint8"
    ):
        raise ValueError("SAM3_TRACK_DATA packed_masks must be a uint8 tensor or None")
    size = value.get("orig_size")
    if (
        not isinstance(size, (list, tuple))
        or len(cast("list[object] | tuple[object, ...]", size)) != 2
        or any(
            type(dimension) is not int or dimension <= 0
            for dimension in cast("list[object] | tuple[object, ...]", size)
        )
    ):
        raise ValueError("SAM3_TRACK_DATA orig_size must contain positive integer height and width")
    frames = value.get("n_frames")
    if type(frames) is not int or frames < 0:
        raise ValueError("SAM3_TRACK_DATA n_frames must be a nonnegative integer")
    return value


def _encode(value: object) -> bytes:
    data = encode_latent({"samples": value})
    _decode(data)
    return data


def _fingerprint(value: object) -> str:
    return stable_hash([SAM3_TRACK_DATA_TYPE.encode("utf-8"), _encode(value)])


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def register_sam3_track_type(registry: TypeRegistry) -> TypeSpec:
    return registry.register(
        SAM3_TRACK_DATA_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=_fingerprint,
        validate_encoded=_validate,
    )
