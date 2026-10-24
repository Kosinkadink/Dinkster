from __future__ import annotations

import json

import pytest
from dinkster_inference_wire.sampling_wire import NoiseSelection, SamplerSelection, SigmaSchedule
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    EncodedLatentTensor,
    TypeRegistry,
    encode_latent,
)


def _registry() -> TypeRegistry:
    from dinkster_inference_wire import register_inference_types

    registry = TypeRegistry()
    register_inference_types(registry)
    return registry


def test_sigmas_tensor_uses_a_canonical_single_tensor_latent_envelope() -> None:
    record = EncodedLatentTensor("float32", (2,), b"\x00\x00\x80?\x00\x00 @")
    spec = _registry().spec("SIGMAS")

    encoded = spec.encode(record)

    assert spec.fingerprint is not None
    assert encoded == encode_latent({"samples": record})
    assert spec.decode(encoded) == record
    assert spec.fingerprint(record) == spec.fingerprint(spec.decode(encoded))


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


@pytest.mark.parametrize(
    "value",
    (
        {"samples": EncodedLatentTensor("float32", (1,), b"\x00\x00\x80?"), "noise": 1},
        {"samples": (EncodedLatentTensor("float32", (1,), b"\x00\x00\x80?"),)},
    ),
)
def test_sigmas_tensor_rejects_noncanonical_latent_envelopes(value: object) -> None:
    spec = _registry().spec("SIGMAS")
    with pytest.raises(ValueError, match="SIGMAS tensor frame"):
        spec.decode(encode_latent(value))


def test_sigmas_tensor_does_not_accept_legacy_json_as_a_tensor_frame() -> None:
    spec = _registry().spec("SIGMAS")
    assert not spec.encode(SigmaSchedule((1.0, 0.0))).startswith(LATENT_CODEC_MAGIC)
    malformed = LATENT_CODEC_MAGIC + b"\x01" + json.dumps({"$": "tensor"}).encode()
    with pytest.raises(ValueError):
        spec.decode(malformed)


def test_sampling_settings_encodings_remain_unchanged() -> None:
    sampler = SamplerSelection("dinkster.euler", (("eta", 0.0),))
    noise = NoiseSelection(7)
    sigmas = SigmaSchedule((1.0, 0.0), "normal")
    registry = _registry()

    assert registry.spec("SAMPLER").encode(sampler) == (
        b'{"extension_ids":[],"extension_snapshot_digest":null,"format":"SAMPLER.v1",'
        b'"options":[["eta",0.0]],"sampler_id":"dinkster.euler"}'
    )
    assert registry.spec("NOISE").encode(noise) == b'{"format":"NOISE.v1","seed":7}'
    assert registry.spec("SIGMAS").encode(sigmas) == (
        b'{"format":"SIGMAS.v1","source_scheduler_id":"normal","values":[1.0,0.0]}'
    )
