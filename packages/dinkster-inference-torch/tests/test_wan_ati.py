"""Wan ATI trajectory parity with the executed ComfyUI reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference_torch.wan_ati import (
    WanAtiError,
    patch_wan_ati_motion,
    prepare_wan_ati_tracks,
)
from golden_files import load_platform_golden

GOLDEN = load_platform_golden(Path(__file__).parent / "goldens" / "wan_ati_goldens.json")

# Fresh pinned ComfyUI tracks were bit-identical to Dinkster on the failing Linux
# host, isolating issue #564 to cross-host stored-golden drift. The measured
# maximum was 5.97e-8; 8e-8 leaves 1.34x headroom. Downstream mask and feature
# values remain bit-exact and retain strict comparisons below.
TRACK_GOLDEN_ATOL = 8e-8


def _tensor(value: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, str(value["dtype"]))
    return torch.tensor(value["data"], dtype=dtype).reshape(value["shape"])


def test_wan_ati_golden_pins_executed_comfyui_reference() -> None:
    assert GOLDEN["reference"]["repository"] == "https://github.com/Comfy-Org/ComfyUI"
    assert GOLDEN["reference"]["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"


def test_track_preparation_and_motion_projection_match_comfyui_golden() -> None:
    case = GOLDEN["case"]
    tracks = prepare_wan_ati_tracks(
        case["tracks"],
        width=case["width"],
        height=case["height"],
        length=case["length"],
        batch_size=case["batch_size"],
    )
    expected_tracks = tuple(_tensor(value) for value in case["processed_tracks"])

    assert len(tracks) == len(expected_tracks) == 1
    torch.testing.assert_close(
        tracks[0],
        expected_tracks[0],
        rtol=0.0,
        atol=TRACK_GOLDEN_ATOL,
    )

    mask, feature = patch_wan_ati_motion(
        tracks,
        _tensor(case["video"]),
        temperature=case["temperature"],
        topk=case["topk"],
    )

    assert torch.equal(mask, _tensor(case["mask"]))
    assert torch.equal(feature, _tensor(case["feature"]))
    assert torch.equal(mask[:, :, :1], torch.ones_like(mask[:, :, :1]))


def test_track_batches_follow_comfyui_distributed_selection() -> None:
    batch = [
        [{"x": 1, "y": 2}],
        [{"x": 3, "y": 4}],
    ]
    tracks = prepare_wan_ati_tracks(
        json.dumps([[batch[0]], [batch[1]]]),
        width=16,
        height=16,
        length=5,
        batch_size=5,
    )

    assert len(tracks) == 5
    assert [float(value[0, 0, 0, 1]) for value in tracks] == [
        -0.875,
        -0.875,
        -0.625,
        -0.625,
        -0.625,
    ]


@pytest.mark.parametrize("tracks", ("[]", "not json"))
@pytest.mark.parametrize("batch_size", (1, 3))
def test_empty_or_invalid_json_uses_the_image_to_video_fallback(
    tracks: str,
    batch_size: int,
) -> None:
    assert (
        prepare_wan_ati_tracks(
            tracks,
            width=16,
            height=16,
            length=5,
            batch_size=batch_size,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("tracks", "message"),
    (
        ('{"x": 1, "y": 2}', "must contain a list"),
        ('[[{"x": 1}]]', "unsupported batch or track structure"),
        ('[[{"x": 1, "y": "bad"}]]', "numeric x and y"),
        ('[[{"x": 1, "y": NaN}]]', "finite"),
    ),
)
def test_track_preparation_rejects_malformed_values(tracks: str, message: str) -> None:
    with pytest.raises(WanAtiError, match=message):
        prepare_wan_ati_tracks(
            tracks,
            width=16,
            height=16,
            length=5,
            batch_size=1,
        )


def test_motion_projection_requires_matching_wan_video_geometry() -> None:
    tracks = prepare_wan_ati_tracks(
        '[[{"x": 1, "y": 2}]]',
        width=16,
        height=16,
        length=5,
        batch_size=1,
    )
    with pytest.raises(WanAtiError, match="batches must match"):
        patch_wan_ati_motion(tracks, torch.zeros((2, 16, 2, 2, 2)))
    with pytest.raises(WanAtiError, match=r"\[B,16,T,H,W\]"):
        patch_wan_ati_motion(tracks, torch.zeros((1, 15, 2, 2, 2)))


def test_single_frame_motion_keeps_the_conditioned_mask_frame() -> None:
    tracks = prepare_wan_ati_tracks(
        '[[{"x": 1, "y": 2}]]',
        width=16,
        height=16,
        length=1,
        batch_size=1,
    )

    mask, feature = patch_wan_ati_motion(tracks, torch.zeros((1, 16, 1, 2, 2)))

    assert torch.equal(mask, torch.ones((1, 4, 1, 2, 2)))
    assert feature.shape == (1, 16, 1, 2, 2)
