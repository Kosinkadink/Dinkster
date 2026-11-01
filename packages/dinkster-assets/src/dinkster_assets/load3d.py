"""Authored 3D viewport state and granted render captures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from dinkster_values import ASSET_TYPE, LOAD_3D_TYPE, TypeRegistry, TypeSpec, stable_hash

from .identity import AssetError
from .model import AssetRef, LocalAssetRef

_CAPTURES = frozenset({"image", "mask", "normal", "recording"})


def register_load3d_type(registry: TypeRegistry) -> TypeSpec:
    """Reuse the receiving registry's asset authority for captured files."""
    if LOAD_3D_TYPE in registry:
        return registry.spec(LOAD_3D_TYPE)
    assets = registry.spec(ASSET_TYPE)
    asset_coerce = assets.coerce
    assert asset_coerce is not None

    def coerce(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise AssetError("LOAD_3D requires an authored viewport mapping")
        state = dict(cast("Mapping[str, object]", value))
        captures = {}
        for name in _CAPTURES:
            if name not in state:
                continue
            capture = state.pop(name)
            if capture is None or capture == "":
                captures[name] = capture
                continue
            captures[name] = asset_coerce(capture)
        # Viewport metadata is document data, not a pickle or filesystem grant.
        copied = cast("dict[str, object]", json.loads(json.dumps(state, allow_nan=False)))
        return {**copied, **captures}

    def wire(value: object) -> dict[str, object]:
        state = coerce(value)
        return {
            name: json.loads(assets.encode(item))
            if isinstance(item, AssetRef | LocalAssetRef)
            else item
            for name, item in state.items()
        }

    def encode(value: object) -> bytes:
        return json.dumps(wire(value), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def meta(value: object) -> dict[str, object]:
        state = coerce(value)
        return {
            "asset_refs": [
                item.to_wire()
                for name in sorted(_CAPTURES)
                for item in (state.get(name),)
                if isinstance(item, AssetRef)
            ]
        }

    def validate(data: bytes, metadata: Mapping[str, object]) -> None:
        if dict(metadata) != meta(json.loads(data)):
            raise AssetError("LOAD_3D metadata does not match its capture identities")

    return registry.register(
        LOAD_3D_TYPE,
        encode=encode,
        decode=lambda data: coerce(json.loads(data)),
        coerce=coerce,
        fingerprint=lambda value: stable_hash([LOAD_3D_TYPE.encode("utf-8"), encode(value)]),
        meta=meta,
        validate_encoded=validate,
    )
