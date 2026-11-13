"""Hook value codecs retain canonical runtime identities without Torch."""

from __future__ import annotations

import numpy as np
import pytest
from dinkster_api.v1 import (
    FLOATS_TYPE,
    TIMESTEPS_RANGE_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor


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
