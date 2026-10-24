"""Hook value codecs retain canonical runtime identities without Torch."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from dinkster_api.v1 import (
    FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    TIMESTEPS_RANGE_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor, ResidencyTable, ResidentLookupError


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_inference_types(registry)
    return registry


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ((0.75, -2), (0.75, -2)),
        ([0.75, -2], (0.75, -2)),
        ((-3.0, 1.25), (-3.0, 1.25)),
    ),
)
def test_timesteps_range_preserves_endpoint_order_without_clamping(
    value: object, expected: tuple[int | float, int | float]
) -> None:
    spec = _registry().spec(TIMESTEPS_RANGE_TYPE)
    encoded = spec.encode(value)

    assert spec.decode(encoded) == expected
    assert spec.encode(spec.decode(encoded)) == encoded


@pytest.mark.parametrize(
    "value",
    (True, "0.5", object(), (0.0,), (0.0, 1.0, 2.0), (0.0, float("nan"))),
)
def test_timesteps_range_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _registry().spec(TIMESTEPS_RANGE_TYPE).encode(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, 1),
        (-1.25, -1.25),
        ([1, -2.5], (1, -2.5)),
        ((1, -2.5), (1, -2.5)),
        (range(-2, 3, 2), (-2, 0, 2)),
        ([], ()),
    ),
)
def test_floats_preserves_scalars_or_ordered_sequences(value: object, expected: object) -> None:
    spec = _registry().spec(FLOATS_TYPE)
    encoded = spec.encode(value)

    assert spec.decode(encoded) == expected
    assert spec.encode(spec.decode(encoded)) == encoded


def test_floats_preserves_numpy_tensor_dtype_shape_and_bytes() -> None:
    source = np.array([np.float32(0.5), np.float32(-3.25)], dtype=np.float32)
    spec = _registry().spec(FLOATS_TYPE)
    decoded = spec.decode(spec.encode(source))

    assert type(decoded) is EncodedLatentTensor
    assert decoded.dtype == "float32"
    assert decoded.shape == (2,)
    assert decoded.data == source.tobytes()
    assert spec.fingerprint is not None
    fingerprint = spec.fingerprint
    assert fingerprint(source) == fingerprint(decoded)


@pytest.mark.parametrize(
    "value",
    (
        True,
        "1.0",
        object(),
        (item for item in (1, 2)),
        np.float32(1.0),
        np.array(1.0, dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        [1, float("inf")],
        [False],
    ),
)
def test_floats_rejects_impostors_unsized_and_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _registry().spec(FLOATS_TYPE).encode(value)


def test_hook_values_are_resident_and_release_invalidates_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
    registry = _registry()
    for type_id in (HOOKS_TYPE, HOOK_KEYFRAMES_TYPE):
        spec = registry.spec(type_id)
        source = object()
        encoded = spec.encode(source)

        assert spec.decode(encoded) is source
        assert spec.fingerprint is not None
        fingerprint = spec.fingerprint
        assert fingerprint(source) == fingerprint(spec.decode(encoded))
        resident_id = json.loads(encoded)["residentId"]
        assert pool.remove(resident_id) is source
        with pytest.raises(ResidentLookupError):
            spec.decode(encoded)


def test_hook_values_reject_foreign_resident_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    source_pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: source_pool)
    source_registry = _registry()
    target_pool = ResidencyTable()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: target_pool)
    target_registry = _registry()
    encoded = source_registry.spec(HOOKS_TYPE).encode(object())

    with pytest.raises(ResidentLookupError):
        target_registry.spec(HOOKS_TYPE).decode(encoded)


def test_hook_type_registration_is_torch_free() -> None:
    script = """
import sys
from dinkster_api.v1 import (
    FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE,
    TypeRegistry, register_inference_types,
)

registry = TypeRegistry()
register_inference_types(registry)
type_ids = (FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE)
assert all(type_id in registry for type_id in type_ids)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
