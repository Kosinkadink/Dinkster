"""Inference-owned structural and resident boundary value types."""

from __future__ import annotations

import json
import threading
from typing import cast

import numpy as np
import pytest
from dinkster_inference import (
    CLIP_TYPE_ID,
    CONTROL_TYPE_ID,
    CONTROL_WIRE_FORMAT,
    GUIDER_TYPE_ID,
    LATENT_TYPE_ID,
    MODEL_TYPE_ID,
    NOISE_TYPE_ID,
    SAMPLER_TYPE_ID,
    SIGMAS_TYPE_ID,
    VAE_TYPE_ID,
    BuiltinSamplerSelection,
    ControlApplication,
    MultiStreamLatent,
    PayloadReference,
    PercentRange,
    SDControlMode,
    decode_control_application,
    encode_control_application,
    register_inference_types,
)
from dinkster_inference.sampling_wire import NoiseSelection, SamplerSelection, SigmaSchedule
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    LATENT_CODEC_VERSION,
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    ResidentLookupError,
    TypeRegistry,
)
from dinkster_workers import ValueCodec


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_inference_types(registry)
    return registry


# --- dinkster.latent: structural codec -------------------------------------


def test_latent_encode_and_validate_accept_single_stream() -> None:
    spec = _registry().spec(LATENT_TYPE_ID)
    value = {"samples": np.arange(8, dtype=np.float32).reshape(2, 4)}
    encoded = spec.encode(value)
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(
        {"samples": np.arange(8, dtype=np.float32).reshape(2, 4)}
    )


def _multistream_value() -> dict[str, object]:
    return {
        "samples": MultiStreamLatent.from_pairs(
            [
                (
                    "video",
                    {"shape": (1, 4, 2, 2), "values": [1.0, 2.0], "fps": 24.0},
                ),
                (
                    "audio",
                    {"shape": (1, 8, 3), "values": [-1.0, 0.5], "sample_rate": 24_000},
                ),
            ]
        ),
        "noise_mask": MultiStreamLatent.from_pairs(
            [
                ("video", {"shape": (1, 2, 2), "values": [0.0, 1.0]}),
                ("audio", {"shape": (1, 3), "values": [1.0, 0.0]}),
            ]
        ),
        "batch_index": [3],
        "source": {"kind": "prepared", "tags": ("video", "audio")},
    }


def _assert_multistream_value(value: object) -> None:
    decoded = cast("dict[str, object]", value)
    samples = cast("MultiStreamLatent[object]", decoded["samples"])
    mask = cast("MultiStreamLatent[object]", decoded["noise_mask"])
    expected = _multistream_value()
    expected_samples = cast("MultiStreamLatent[object]", expected["samples"])
    expected_mask = cast("MultiStreamLatent[object]", expected["noise_mask"])
    assert samples.roles == ("video", "audio")
    assert mask.roles == samples.roles
    for role in samples.roles:
        assert samples.by_role(role) == expected_samples.by_role(role)
        assert mask.by_role(role) == expected_mask.by_role(role)
    assert decoded["batch_index"] == [3]
    assert decoded["source"] == {"kind": "prepared", "tags": ("video", "audio")}


def test_latent_multistream_round_trip_preserves_roles_payloads_masks_and_metadata() -> None:
    spec = _registry().spec(LATENT_TYPE_ID)
    encoded = spec.encode(_multistream_value())
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    _assert_multistream_value(spec.decode(encoded))


def test_latent_multistream_crosses_the_value_codec_boundary() -> None:
    registry = _registry()
    sender = ValueCodec(registry, use_shm=False, accept_shm=False)
    receiver = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, sent = sender.encode(registry.wrap(LATENT_TYPE_ID, _multistream_value()), blobs, [])
    received, decoded = receiver.decode(wire, blobs, [])

    assert sent.transport == "inline"
    assert decoded.transport == "inline"
    _assert_multistream_value(received.resolve())


def _latent_frame(samples: object) -> bytes:
    body = json.dumps(
        {"$": "map", "items": {"samples": samples}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return LATENT_CODEC_MAGIC + bytes((LATENT_CODEC_VERSION,)) + body


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ({"$": "future", "streams": []}, "unknown tag"),
        ({"$": "multi"}, "invalid multi-stream record"),
        ({"$": "multi", "streams": []}, "invalid multi-stream topology"),
        ({"$": "multi", "streams": [["video"]]}, "invalid stream record"),
        (
            {"$": "multi", "streams": [["video", 1], ["video", 2]]},
            "invalid stream roles",
        ),
        (
            {"$": "multi", "streams": [["video", 1]], "metadata": {}},
            "invalid multi-stream record",
        ),
    ],
)
def test_latent_multistream_decode_fails_closed(samples: object, message: str) -> None:
    spec = _registry().spec(LATENT_TYPE_ID)
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError, match=message):
        spec.validate_encoded(_latent_frame(samples), {})


def test_latent_single_stream_encoding_matches_v1_golden() -> None:
    value = {
        "samples": np.arange(4, dtype=np.float32).reshape(1, 4),
        "batch_index": [3],
    }
    expected = (
        b'DINKSTER-LATENT\x00\x01{"$":"map","items":{"batch_index":[3],'
        b'"samples":{"$":"tensor","data":"AAAAAAAAgD8AAABAAABAQA==",'
        b'"dtype":"float32","shape":[1,4]}}}'
    )
    assert _registry().spec(LATENT_TYPE_ID).encode(value) == expected


def test_latent_encode_requires_samples_mapping() -> None:
    spec = _registry().spec(LATENT_TYPE_ID)
    with pytest.raises(ValueError, match="samples"):
        spec.encode({"noise": np.zeros((1,), dtype=np.float32)})


# --- dinkster.control: canonical ControlApplication wire --------------------


def _control_chain() -> ControlApplication:
    base = ControlApplication(
        child_id="controlnet",
        hint=PayloadReference("hint-digest-a"),
        strength=0.5,
        window=PercentRange(0.0, 1.0),
    )
    return ControlApplication(
        child_id="union",
        hint=PayloadReference("hint-digest-b"),
        strength=1.25,
        window=PercentRange(0.25, 0.75),
        previous=base,
        mode=SDControlMode(provider="sdxl-controlnet-union", token="depth"),
    )


def test_control_round_trip_preserves_chain_and_mode() -> None:
    application = _control_chain()
    assert decode_control_application(encode_control_application(application)) == application


def test_control_wire_is_canonical_and_fingerprint_stable() -> None:
    spec = _registry().spec(CONTROL_TYPE_ID)
    first, second = _control_chain(), _control_chain()
    assert encode_control_application(first) == encode_control_application(second)
    assert spec.fingerprint is not None
    assert spec.fingerprint(first) == spec.fingerprint(second)


def test_control_decode_fails_closed() -> None:
    encoded = encode_control_application(_control_chain())
    wire = json.loads(encoded)
    wrong_format = dict(wire, format="dinkster.control-wire.v0")
    with pytest.raises(ValueError, match="wire format"):
        decode_control_application(json.dumps(wrong_format).encode("utf-8"))
    extra_key = dict(wire, extra=1)
    with pytest.raises(ValueError, match="exactly format and applications"):
        decode_control_application(json.dumps(extra_key).encode("utf-8"))
    empty = dict(wire, applications=[])
    with pytest.raises(ValueError, match="nonempty"):
        decode_control_application(json.dumps(empty).encode("utf-8"))
    unknown_record_key = dict(wire, applications=[dict(wire["applications"][0], rogue=1)])
    with pytest.raises(ValueError, match="childId"):
        decode_control_application(json.dumps(unknown_record_key).encode("utf-8"))


def test_control_decode_refuses_noncanonical_bytes() -> None:
    encoded = encode_control_application(_control_chain())
    padded = json.dumps(json.loads(encoded), sort_keys=True, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        decode_control_application(padded)
    duplicate_key = encoded.replace(
        b'"format":"dinkster.control-wire.v1"',
        b'"format":"bogus","format":"dinkster.control-wire.v1"',
    )
    with pytest.raises(ValueError, match="canonical"):
        decode_control_application(duplicate_key)


def test_control_decode_rebuilds_through_validators() -> None:
    encoded = encode_control_application(_control_chain())
    wire = json.loads(encoded)
    wire["applications"][0]["strength"] = 99.0
    with pytest.raises(ValueError, match="strength"):
        decode_control_application(json.dumps(wire).encode("utf-8"))


def test_control_coerce_requires_exact_application() -> None:
    spec = _registry().spec(CONTROL_TYPE_ID)
    assert spec.coerce is not None
    with pytest.raises(TypeError, match="ControlApplication"):
        spec.coerce("not-a-control-application")
    assert spec.meta is not None
    assert spec.meta(_control_chain()) == {"format": CONTROL_WIRE_FORMAT}


# --- resident handle types: objects stay put, stubs cross ----------------


@pytest.mark.parametrize(
    ("type_id", "settings"),
    [
        (NOISE_TYPE_ID, NoiseSelection(None)),
        (NOISE_TYPE_ID, NoiseSelection(2**64 - 1)),
        (SIGMAS_TYPE_ID, SigmaSchedule((1.0, 0.5, 0.0))),
        (SIGMAS_TYPE_ID, SigmaSchedule((1.0, 0.0), "dinkster.simple")),
        (SIGMAS_TYPE_ID, SigmaSchedule(())),
        (SAMPLER_TYPE_ID, BuiltinSamplerSelection("dinkster.euler", ())),
        (
            SAMPLER_TYPE_ID,
            SamplerSelection("extension.euler", (("eta", 0.5),), "sha256:" + "a" * 64, ("ext",)),
        ),
    ],
)
def test_sampling_settings_cross_independent_registries_without_residency(
    type_id: str, settings: object
) -> None:
    sender, receiver = _registry().spec(type_id), _registry().spec(type_id)
    encoded = sender.encode(settings)
    assert "residentId" not in json.loads(encoded)
    assert sender.meta is None
    assert receiver.decode(encoded) == settings
    assert receiver.encode(receiver.decode(encoded)) == encoded
    assert sender.fingerprint is not None and receiver.fingerprint is not None
    assert sender.fingerprint(settings) == receiver.fingerprint(receiver.decode(encoded))
    assert receiver.validate_encoded is not None
    receiver.validate_encoded(encoded, {})
    with pytest.raises(ValueError, match="canonical"):
        receiver.decode(b" " + encoded)
    with pytest.raises(ValueError):
        receiver.decode(encoded[:-1] + b',"extra":true}')
    with pytest.raises(ValueError):
        receiver.decode(b'{"residentId":"old-worker"}')


@pytest.mark.parametrize(
    ("type_id", "payload"),
    [
        (NOISE_TYPE_ID, {"seed": True}),
        (NOISE_TYPE_ID, {"seed": -1}),
        (NOISE_TYPE_ID, {"seed": 2**64}),
        (SIGMAS_TYPE_ID, {"values": [float("nan")]}),
        (SIGMAS_TYPE_ID, {"values": [True]}),
        (SIGMAS_TYPE_ID, {"values": "abc"}),
        (SIGMAS_TYPE_ID, {"values": [1.0, 0.0], "source_scheduler_id": ""}),
        (SAMPLER_TYPE_ID, {"sampler_id": "dinkster.euler", "options": [["eta", {}]]}),
        (SAMPLER_TYPE_ID, {"sampler_id": "dinkster.euler", "options": [["eta", 1, 2]]}),
        (SAMPLER_TYPE_ID, {"sampler_id": "dinkster.euler", "options": [["eta", 1], ["eta", 2]]}),
        (SAMPLER_TYPE_ID, {"sampler_id": "", "options": []}),
    ],
)
def test_sampling_settings_reject_malformed_payloads(
    type_id: str, payload: dict[str, object]
) -> None:
    data = json.dumps({"format": type_id + ".v1", **payload}, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError):
        _registry().spec(type_id).decode(data.encode())


class _Unpicklable:
    def __init__(self) -> None:
        self.lock = threading.Lock()


@pytest.mark.parametrize(
    "type_id",
    [
        MODEL_TYPE_ID,
        CLIP_TYPE_ID,
        VAE_TYPE_ID,
        GUIDER_TYPE_ID,
    ],
)
def test_resident_types_send_stubs_and_resolve_locally(type_id: str) -> None:
    spec = _registry().spec(type_id)
    handle = _Unpicklable()
    encoded = spec.encode(handle)
    wire = json.loads(encoded)
    assert set(wire) == {"residentId"}
    assert spec.decode(encoded) is handle
    assert spec.fingerprint is not None
    assert spec.fingerprint(handle).startswith("resident:")
    assert spec.meta is not None
    meta = spec.meta(handle)
    assert meta[RESOURCE_ID_META_KEY] == spec.fingerprint(handle)
    assert RESOURCE_OWNER_META_KEY in meta


def test_resident_stub_for_unknown_handle_fails_clearly() -> None:
    spec = _registry().spec(MODEL_TYPE_ID)
    stub = json.dumps({"residentId": "not-held-here"}).encode("utf-8")
    with pytest.raises(ResidentLookupError, match="not held by this process"):
        spec.decode(stub)
