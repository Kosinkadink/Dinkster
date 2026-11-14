"""Cold portable Wan values preserve tensor records and structural cache identity."""

from __future__ import annotations

import pytest
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    TRACKS_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    TypeRegistry,
)
from dinkster_inference_wire.wan_wire import register_wan_type
from dinkster_values import EncodedLatentTensor

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
WANDANCER_ENCODER = {
    "audio_feature": EncodedLatentTensor("float32", (1, 1, 2), b"\x00\x00\x00\x40\x00\x00\x80\xbf"),
    "fps": 15.0,
    "audio_inject_scale": 0.375,
    "metadata": {"source": "wandancer"},
}


@pytest.fixture
def registry():
    values = TypeRegistry()
    with values.registering_pack(
        "wan-codec-fixture", (WAN_CAMERA_EMBEDDING_TYPE, TRACKS_TYPE), allow_bare=True
    ):
        for type_id in (WAN_CAMERA_EMBEDDING_TYPE, TRACKS_TYPE, AUDIO_ENCODER_OUTPUT_TYPE):
            register_wan_type(values, type_id)
        yield values


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
        (
            AUDIO_ENCODER_OUTPUT_TYPE,
            WANDANCER_ENCODER,
            {**WANDANCER_ENCODER, "fps": 30.0},
        ),
        (
            AUDIO_ENCODER_OUTPUT_TYPE,
            {**WANDANCER_ENCODER, "fps": 0.0, "audio_inject_scale": 11.0},
            {**WANDANCER_ENCODER, "fps": 0.0, "audio_inject_scale": 12.0},
        ),
    ),
)
def test_public_codec_roundtrip_and_fingerprints(
    registry: TypeRegistry, type_id: str, value: object, changed: object
) -> None:
    spec = registry.spec(type_id)
    encoded = spec.encode(value)
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    assert spec.decode(encoded) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != spec.fingerprint(changed)
    if isinstance(value, dict):
        assert spec.fingerprint(value) == spec.fingerprint(dict(reversed(tuple(value.items()))))
