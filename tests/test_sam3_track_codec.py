"""Cold structural SAM3 tracks retain packed bytes and independent value identity."""

from __future__ import annotations

import pytest
from dinkster_api.v1 import SAM3_TRACK_DATA_TYPE, TypeRegistry, register_media_types
from dinkster_inference_wire.sam3_wire import register_sam3_track_type
from dinkster_values import EncodedLatentTensor

PACKED = EncodedLatentTensor("uint8", (2, 1, 1, 3), bytes((1, 128, 255, 85, 42, 16)))
EMPTY = EncodedLatentTensor("uint8", (3, 0, 2, 4), b"")
TRACKS = {
    "packed_masks": PACKED,
    "orig_size": (9, 17),
    "n_frames": 2,
    "scores": [0.75],
    "metadata": {"tag": ("source", 4), "mask": EncodedLatentTensor("bool", (1,), b"\x01")},
}


@pytest.fixture
def registry() -> TypeRegistry:
    values = TypeRegistry()
    register_media_types(values)
    register_sam3_track_type(values)
    return values


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
    registry: TypeRegistry,
    changed: dict[str, object],
) -> None:
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)
    assert spec.fingerprint is not None
    assert spec.fingerprint(TRACKS) != spec.fingerprint(changed)
