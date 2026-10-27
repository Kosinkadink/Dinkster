"""Cold portable Wan values preserve tensor records and structural cache identity."""

from __future__ import annotations

import subprocess
import sys

import pytest
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    TRACKS_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_values import EncodedLatentTensor, encode_latent

HALF = EncodedLatentTensor("float16", (1, 2), b"\x00\x3c\x00\xc0")
BFLOAT = EncodedLatentTensor("bfloat16", (2, 1), b"\x80\x3f\x00\xc0")
EMPTY = EncodedLatentTensor("float32", (2, 0, 3), b"")
VISIBILITY = EncodedLatentTensor("bool", (1, 2), b"\x01\x00")
ENCODER = {
    "encoded_audio": HALF,
    "encoded_audio_all_layers": (BFLOAT, EMPTY),
    "audio_samples": 17,
    "metadata": {"flag": True, "shape": [1, 2], "tag": ("layer", 3)},
}


@pytest.mark.parametrize(
    ("type_id", "value", "changed"),
    (
        (WAN_CAMERA_EMBEDDING_TYPE, HALF, BFLOAT),
        (
            WAN_CAMERA_EMBEDDING_TYPE,
            HALF,
            EncodedLatentTensor("float16", (1, 2), b"\x00\x40\x00\xc0"),
        ),
        (WAN_CAMERA_EMBEDDING_TYPE, EMPTY, HALF),
        (TRACKS_TYPE, {"track_path": HALF}, {"track_path": HALF, "track_visibility": VISIBILITY}),
        (TRACKS_TYPE, {"track_path": HALF, "track_visibility": VISIBILITY}, {"track_path": BFLOAT}),
        (AUDIO_ENCODER_OUTPUT_TYPE, ENCODER, {**ENCODER, "audio_samples": 18}),
        (
            AUDIO_ENCODER_OUTPUT_TYPE,
            ENCODER,
            {**ENCODER, "encoded_audio_all_layers": (EMPTY, BFLOAT)},
        ),
        (
            AUDIO_ENCODER_OUTPUT_TYPE,
            ENCODER,
            {**ENCODER, "encoded_audio_all_layers": [BFLOAT, EMPTY]},
        ),
    ),
)
def test_public_codec_roundtrip_and_fingerprints(
    type_id: str, value: object, changed: object
) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    original_spec = registry.spec(type_id)
    register_inference_types(registry)
    spec = registry.spec(type_id)
    assert spec is original_spec
    encoded = spec.encode(value)
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    assert spec.decode(encoded) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != spec.fingerprint(changed)
    if isinstance(value, dict):
        assert spec.fingerprint(value) == spec.fingerprint(dict(reversed(tuple(value.items()))))


@pytest.mark.parametrize(
    ("type_id", "value"),
    (
        (WAN_CAMERA_EMBEDDING_TYPE, {"tensor": HALF}),
        (WAN_CAMERA_EMBEDDING_TYPE, EncodedLatentTensor("float16", (2, 3), b"\x00\x00")),
        (TRACKS_TYPE, {}),
        (TRACKS_TYPE, {"track_path": [1, 2]}),
        (TRACKS_TYPE, {"track_path": HALF, "track_visibility": False}),
        (AUDIO_ENCODER_OUTPUT_TYPE, {**ENCODER, "encoded_audio": None}),
        (AUDIO_ENCODER_OUTPUT_TYPE, {**ENCODER, "encoded_audio_all_layers": {"0": HALF}}),
        (AUDIO_ENCODER_OUTPUT_TYPE, {**ENCODER, "encoded_audio_all_layers": [1]}),
        (AUDIO_ENCODER_OUTPUT_TYPE, {**ENCODER, "audio_samples": True}),
        (AUDIO_ENCODER_OUTPUT_TYPE, {**ENCODER, "audio_samples": 1.5}),
    ),
)
def test_public_codec_rejects_invalid_fields_on_encode_and_decode(
    type_id: str, value: object
) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(type_id)
    assert spec.validate_encoded is not None
    payload = encode_latent({"samples": value})
    with pytest.raises(ValueError):
        spec.encode(value)
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


@pytest.mark.parametrize(
    "type_id", (WAN_CAMERA_EMBEDDING_TYPE, TRACKS_TYPE, AUDIO_ENCODER_OUTPUT_TYPE)
)
@pytest.mark.parametrize("payload", (b"not-a-tensor", encode_latent({"samples": HALF, "extra": 1})))
def test_public_codec_rejects_invalid_envelopes(type_id: str, payload: bytes) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(type_id)
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


def test_public_registration_remains_torch_and_model_runtime_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import TypeRegistry, register_inference_types\n"
            "from dinkster_values import EncodedLatentTensor\n"
            "r = TypeRegistry(); register_inference_types(r); register_inference_types(r)\n"
            "t = EncodedLatentTensor('bool', (0, 3), b'')\n"
            "for name, value in [('WAN_CAMERA_EMBEDDING', t), ('TRACKS', {'track_path': t}), "
            "('AUDIO_ENCODER_OUTPUT', {'encoded_audio': t, "
            "'encoded_audio_all_layers': [t], 'audio_samples': 0})]:\n"
            " s = r.spec(name); data = s.encode(value); s.validate_encoded(data, {})\n"
            " assert s.decode(data) == value\n"
            "assert not any(n == 'torch' or n.startswith('torch.') or "
            "n == 'dinkster_inference' or n.startswith('dinkster_inference.') "
            "for n in sys.modules)\n",
        ],
        check=True,
    )
