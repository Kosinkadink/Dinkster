"""Cold structural SAM3 tracks retain packed bytes and independent value identity."""

from __future__ import annotations

import subprocess
import sys

import pytest
from dinkster_api.v1 import SAM3_TRACK_DATA_TYPE, TypeRegistry, register_inference_types
from dinkster_values import LATENT_TYPE, EncodedLatentTensor, encode_latent

PACKED = EncodedLatentTensor("uint8", (2, 1, 1, 3), bytes((1, 128, 255, 85, 42, 16)))
EMPTY = EncodedLatentTensor("uint8", (3, 0, 2, 4), b"")
TRACKS = {
    "packed_masks": PACKED,
    "orig_size": (9, 17),
    "n_frames": 2,
    "scores": [0.75],
    "metadata": {"tag": ("source", 4), "mask": EncodedLatentTensor("bool", (1,), b"\x01")},
}


@pytest.mark.parametrize(
    "value",
    (
        TRACKS,
        {**TRACKS, "scores": None},
        {**TRACKS, "scores": EncodedLatentTensor("float32", (1,), b"\x00\x00\x40\x3f")},
        {**TRACKS, "packed_masks": EMPTY, "n_frames": 3, "scores": ()},
        {"packed_masks": None, "orig_size": [13, 7], "n_frames": 0},
        {"packed_masks": None, "orig_size": (13, 7), "n_frames": 4, "scores": []},
    ),
)
def test_sam3_public_codec_preserves_complete_structure(value: dict[str, object]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)
    register_inference_types(registry)
    assert registry.spec(SAM3_TRACK_DATA_TYPE) is spec
    assert spec.validate_encoded is not None
    payload = spec.encode(value)
    spec.validate_encoded(payload, {})
    assert spec.decode(payload) == value
    assert spec.fingerprint is not None
    assert spec.fingerprint(spec.decode(payload)) == spec.fingerprint(value)
    assert spec.fingerprint(dict(reversed(tuple(value.items())))) == spec.fingerprint(value)
    latent_fingerprint = registry.spec(LATENT_TYPE).fingerprint
    assert latent_fingerprint is not None
    assert latent_fingerprint({"samples": value}) != spec.fingerprint(value)


@pytest.mark.parametrize(
    "changed",
    (
        {
            **TRACKS,
            "packed_masks": EncodedLatentTensor("uint8", PACKED.shape, b"\x02\x80\xffU*\x10"),
        },
        {**TRACKS, "packed_masks": EncodedLatentTensor("uint8", (1, 2, 3, 1), PACKED.data)},
        {**TRACKS, "orig_size": [9, 17]},
        {**TRACKS, "orig_size": (17, 9)},
        {**TRACKS, "n_frames": 3},
        {**TRACKS, "scores": (0.75,)},
        {**TRACKS, "scores": [0.25]},
        {key: value for key, value in TRACKS.items() if key != "scores"},
    ),
)
def test_sam3_fingerprint_includes_bytes_shape_dimensions_frames_and_scores(
    changed: dict[str, object],
) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)
    assert spec.fingerprint is not None
    assert spec.fingerprint(TRACKS) != spec.fingerprint(changed)


@pytest.mark.parametrize(
    "value",
    (
        {},
        {**TRACKS, "packed_masks": False},
        {**TRACKS, "packed_masks": EncodedLatentTensor("float32", (1,), b"\0\0\0\0")},
        {**TRACKS, "packed_masks": EncodedLatentTensor("uint8", (2, 3), b"\0")},
        {**TRACKS, "orig_size": (9, True)},
        {**TRACKS, "orig_size": (9,)},
        {**TRACKS, "orig_size": (0, 3)},
        {**TRACKS, "n_frames": True},
        {**TRACKS, "n_frames": 1.5},
        {**TRACKS, "n_frames": -1},
    ),
)
def test_sam3_rejects_malformed_payloads_in_every_codec_entry(value: dict[str, object]) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)
    assert spec.validate_encoded is not None
    payload = encode_latent({"samples": value})
    with pytest.raises(ValueError):
        spec.encode(value)
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


def test_sam3_codec_registration_is_cold_and_public_identity_is_shared() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import (SAM3_TRACK_DATA_TYPE, TypeRegistry, "
            "register_inference_types)\n"
            "from dinkster_values.builtin_types import SAM3_TRACK_DATA_TYPE as canonical\n"
            "assert SAM3_TRACK_DATA_TYPE is canonical\n"
            "r = TypeRegistry(); register_inference_types(r); register_inference_types(r)\n"
            "s = r.spec(canonical)\n"
            "v = {'packed_masks': None, 'orig_size': (2, 5), 'n_frames': 0}\n"
            "b = s.encode(v); s.validate_encoded(b, {}); assert s.decode(b) == v\n"
            "assert not any(n == 'torch' or n.startswith('torch.') or "
            "n == 'dinkster_inference' or n.startswith('dinkster_inference.') "
            "for n in sys.modules)\n",
        ],
        check=True,
    )
