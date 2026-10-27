"""LOSS_MAP uses the shared structural framing without numeric normalization."""

from __future__ import annotations

import pytest
from dinkster_api.v1 import LATENT_TYPE, LOSS_MAP_TYPE, TypeRegistry, register_inference_types
from dinkster_values import encode_latent


def test_loss_map_preserves_ordered_losses_and_structural_extras() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOSS_MAP_TYPE)
    value = {"loss": [3.25, -1.5, 0.0], "epoch": 7, "details": {"name": "L1"}}

    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != registry.spec(LATENT_TYPE).fingerprint({"samples": value})


def test_loss_map_accepts_empty_losses_and_rejects_invalid_frames() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LOSS_MAP_TYPE)

    assert spec.decode(spec.encode({"loss": [], "tag": ["kept"]})) == {"loss": [], "tag": ["kept"]}
    with pytest.raises(ValueError, match="ordered list"):
        spec.decode(encode_latent({"samples": {"loss": (1.0,)}}))
    with pytest.raises(ValueError, match="only samples"):
        spec.decode(encode_latent({"samples": {"loss": []}, "other": {}}))
