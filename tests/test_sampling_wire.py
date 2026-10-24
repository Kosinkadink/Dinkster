from __future__ import annotations

import json
import subprocess
import sys

import pytest
from dinkster_inference_wire.sampling_wire import (
    BuiltinSamplerSelection,
    NoiseSelection,
    SamplerSelection,
    SigmaSchedule,
    register_sampling_type,
)
from dinkster_memory.resident import ResidentPool
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    EncodedLatentTensor,
    ResidentLookupError,
    TypeRegistry,
    encode_latent,
    process_instance_token,
    stable_hash,
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


@pytest.mark.parametrize(
    ("type_id", "value", "expected"),
    (
        (
            "SAMPLER",
            BuiltinSamplerSelection("euler", (("eta", 0.375),)),
            b'{"format":"SAMPLER.v1","options":[["eta",0.375]],"sampler_id":"euler"}',
        ),
        (
            "SAMPLER",
            SamplerSelection("custom", (("s_noise", 1.25),), "snapshot", ("extension",)),
            b'{"extension_ids":["extension"],"extension_snapshot_digest":"snapshot",'
            b'"format":"SAMPLER.v1","options":[["s_noise",1.25]],"sampler_id":"custom"}',
        ),
        ("NOISE", NoiseSelection(None), b'{"format":"NOISE.v1","seed":null}'),
        ("NOISE", NoiseSelection(739), b'{"format":"NOISE.v1","seed":739}'),
    ),
)
def test_portable_sampling_identity_is_unchanged(
    type_id: str, value: object, expected: bytes
) -> None:
    spec = _registry().spec(type_id)
    assert spec.encode(value) == expected
    assert spec.decode(expected) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == stable_hash([expected])
    assert spec.meta is not None
    assert spec.meta(value) == {}


class RuntimeSampler:
    def sample(self) -> str:
        return "sampled"


class RuntimeNoise:
    def generate_noise(self) -> str:
        return "noise"


@pytest.mark.parametrize(
    ("type_id", "value"), (("SAMPLER", RuntimeSampler()), ("NOISE", RuntimeNoise()))
)
def test_runtime_sampling_residency_validation_and_release(type_id: str, value: object) -> None:
    pool = ResidentPool()
    registry = TypeRegistry()
    spec = register_sampling_type(registry, type_id, resident_table=pool)
    assert registry.resident_tables == (pool,)
    encoded = spec.encode(value)
    assert spec.meta is not None
    metadata = spec.meta(value)
    assert metadata["format"] == type_id + ".resident.v1"
    assert metadata[RESOURCE_OWNER_META_KEY] == process_instance_token()
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == metadata[RESOURCE_ID_META_KEY]
    assert spec.decode(encoded) is value
    assert spec.encode(value) == encoded
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, metadata)

    foreign = register_sampling_type(TypeRegistry(), type_id, resident_table=ResidentPool())
    assert foreign.validate_encoded is not None
    foreign.validate_encoded(encoded, metadata)
    with pytest.raises(ResidentLookupError, match="another worker or an earlier worker lifetime"):
        foreign.decode(encoded)

    invalidated: list[str] = []
    pool.register_invalidator(invalidated.append)
    assert pool.release_resident(value)
    assert invalidated == [metadata[RESOURCE_ID_META_KEY]]
    with pytest.raises(ResidentLookupError, match="another worker or an earlier worker lifetime"):
        spec.decode(encoded)
    spec.validate_encoded(encoded, metadata)

    for payload in (b"{}", b'{"residentId":"foreign"}', b'{"format":"NOISE.v1","seed":-1}'):
        with pytest.raises(ValueError):
            spec.validate_encoded(payload, metadata)
        with pytest.raises(ValueError):
            spec.decode(payload)
    with pytest.raises(ValueError, match="metadata format"):
        spec.validate_encoded(encoded, {})
    with pytest.raises(ResidentLookupError, match="malformed resident stub"):
        spec.validate_encoded(type_id.encode() + b'.resident.v1\x00{"residentId":""}', metadata)


@pytest.mark.parametrize("native_first", (True, False))
def test_sampling_registration_order_preserves_the_shared_pool(native_first: bool) -> None:
    from dinkster_inference_wire import register_inference_types
    from dinkster_memory.resident import default_pool
    from dinkster_native.native import register_native_types

    registry = TypeRegistry()
    register = (register_native_types, register_inference_types)
    if not native_first:
        register = register[::-1]
    for registration in register:
        registration(registry)
    for type_id, value in (("SAMPLER", RuntimeSampler()), ("NOISE", RuntimeNoise())):
        spec = registry.spec(type_id)
        assert spec.decode(spec.encode(value)) is value
        assert default_pool() in registry.resident_tables


def test_cold_registration_and_resident_inspection_need_no_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["torch"] = None
sys.modules["dinkster_inference"] = None
from dinkster_api.v1 import TypeRegistry, register_inference_types
from dinkster_native.native import register_native_types
from dinkster_nodes_comfy import COMFY_NODES
registry = TypeRegistry()
register_inference_types(registry)
register_native_types(registry)
for node in COMFY_NODES:
    node.schema()
for type_id in ("SAMPLER", "NOISE"):
    prefix = type_id + ".resident.v1"
    registry.spec(type_id).validate_encoded(
        prefix.encode() + b'\\x00{"residentId":"foreign"}', {"format": prefix}
    )
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
