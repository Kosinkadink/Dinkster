"""Portable VIDEO behavior for ComfyUI video component nodes."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from dinkster_api.v1 import coerce_audio, effective_video_facts
from dinkster_nodes_comfy.video_components import CreateVideo, GetVideoComponents, VideoSlice


def _images(*, channels: int = 3) -> np.ndarray:
    pixels = np.arange(3 * 2 * 4 * channels, dtype=np.float32)
    return pixels.reshape(3, 2, 4, channels) / pixels.size


def _audio() -> object:
    return {"waveform": np.arange(12, dtype=np.float32).reshape(1, 1, 12), "sample_rate": 6}


def test_create_and_get_components_preserve_asymmetric_pixels_and_unedited_audio() -> None:
    images = _images(channels=4)
    audio = _audio()
    video = CreateVideo.execute(images=images, fps=3.0, audio=audio)["VIDEO"]

    assert video["components"]["images"] is images
    assert video["components"]["audio"] == coerce_audio(audio)
    components = GetVideoComponents.execute(video=video)

    np.testing.assert_array_equal(components["images"], images)
    assert components["images"].shape[-1] == 4
    assert components["fps"] == 3.0
    assert components["bit_depth"] == 8
    assert components["color_space"] == "sRGB"
    assert components["audio"] == coerce_audio(audio)


@pytest.mark.parametrize(
    ("color_space", "bit_depth", "expected_depth"),
    (("sRGB", "auto", 8), ("HDR", "auto", 10), ("HDR PQ", 8, 8), ("sRGB", 10, 10)),
)
def test_create_video_resolves_color_and_depth_defaults(
    color_space: str, bit_depth: str | int, expected_depth: int
) -> None:
    video = CreateVideo.execute(
        images=_images(), fps=30.0, color_space=color_space, bit_depth=bit_depth
    )["VIDEO"]
    components = GetVideoComponents.execute(video=video)

    assert components["bit_depth"] == expected_depth
    assert components["color_space"] == color_space


@pytest.mark.parametrize("codec", ("auto", "h264"))
def test_create_video_encodes_requested_codec_to_mkv(codec: str) -> None:
    components = CreateVideo.execute(images=_images(), fps=2.0)["VIDEO"]
    encoded = CreateVideo.execute(images=_images(), fps=2.0, codec=codec)["VIDEO"]

    assert "components" in components
    assert "source" in encoded
    assert encoded["probe"]["container"] == "mkv"
    assert encoded["probe"]["video_codec"] == "h264"


def test_slice_preserves_negative_start_and_unlimited_duration_without_mutating_source() -> None:
    video = CreateVideo.execute(images=_images(), fps=2.0)["VIDEO"]
    trimmed = VideoSlice.execute(video=video, start_time=-1.0, duration=0.0, strict_duration=False)[
        "VIDEO"
    ]
    facts = effective_video_facts(trimmed)

    assert video["edits"] == []
    assert facts["duration"] == Fraction(1)
    np.testing.assert_array_equal(
        GetVideoComponents.execute(video=trimmed)["images"], _images()[-2:]
    )


def test_get_components_trims_audio_only_after_a_video_edit() -> None:
    audio = _audio()
    video = CreateVideo.execute(images=_images(), fps=3.0, audio=audio)["VIDEO"]
    edited = VideoSlice.execute(video=video, start_time=0.0, duration=0.5, strict_duration=False)[
        "VIDEO"
    ]

    assert GetVideoComponents.execute(video=video)["audio"] == coerce_audio(audio)
    assert GetVideoComponents.execute(video=edited)["audio"]["edits"] == [
        {"trim": {"start_sample": 0, "sample_count": 3}}
    ]


def test_slice_rejects_strict_duration_past_video_end() -> None:
    video = CreateVideo.execute(images=_images(), fps=2.0)["VIDEO"]

    with pytest.raises(ValueError, match="requested duration"):
        VideoSlice.execute(video=video, start_time=1.0, duration=1.0, strict_duration=True)
