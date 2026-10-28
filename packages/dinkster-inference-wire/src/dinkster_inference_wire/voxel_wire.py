"""Source-shaped dense and sparse VOXEL fields on the structural tensor wire."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

from dinkster_values import (
    VOXEL_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

from .media_wire import materialize_latent


def _attributes(value: object) -> dict[str, object]:
    attributes = cast(
        "Mapping[object, object]", value if isinstance(value, Mapping) else vars(value)
    )
    if any(type(key) is not str for key in attributes):
        raise TypeError("VOXEL attribute names must be strings")
    if "data" not in attributes:
        raise ValueError("VOXEL requires data")
    return dict(cast("Mapping[str, object]", attributes))


def _decode(data: bytes) -> dict[str, object]:
    attributes = _attributes(decode_latent(data, require_samples=False))
    if not isinstance(attributes["data"], EncodedLatentTensor):
        raise ValueError("VOXEL data must be a tensor")
    colors = attributes.get("voxel_colors")
    if colors is not None and not isinstance(colors, EncodedLatentTensor):
        raise ValueError("VOXEL voxel_colors must be a tensor or None")
    return attributes


def _encode(value: object) -> bytes:
    data = encode_latent(_attributes(value))
    _decode(data)
    return data


def _fingerprint(value: object) -> str:
    return stable_hash([VOXEL_TYPE.encode("utf-8"), _encode(value)])


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def materialize_voxel(value: object) -> SimpleNamespace:
    """Expose source attributes without changing dense or sparse tensor layout."""
    attributes = materialize_latent(_attributes(value))
    return SimpleNamespace(**{"voxel_colors": None, "resolution": None, **attributes})


def register_voxel_type(registry: TypeRegistry) -> TypeSpec:
    if VOXEL_TYPE in registry:
        return registry.spec(VOXEL_TYPE)
    return registry.register(
        VOXEL_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=_fingerprint,
        validate_encoded=_validate,
    )
