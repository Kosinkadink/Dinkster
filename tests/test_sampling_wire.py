from __future__ import annotations

from dinkster_values import (
    EncodedLatentTensor,
    TypeRegistry,
)


def _registry() -> TypeRegistry:
    from dinkster_inference_wire import register_inference_types

    registry = TypeRegistry()
    register_inference_types(registry)
    return registry


def test_sigmas_tensor_fingerprint_covers_dtype_shape_and_raw_bytes() -> None:
    spec = _registry().spec("SIGMAS")
    baseline = EncodedLatentTensor("float32", (1,), b"\x00\x00\x80?")
    assert spec.fingerprint is not None
    fingerprint = spec.fingerprint
    assert fingerprint(baseline) != fingerprint(EncodedLatentTensor("float64", (1,), b"\x00" * 8))
    assert fingerprint(baseline) != fingerprint(
        EncodedLatentTensor("float32", (1, 1), b"\x00\x00\x80?")
    )
    assert fingerprint(baseline) != fingerprint(
        EncodedLatentTensor("float32", (1,), b"\x00\x00\x00@")
    )
