"""Portable VIDEO behavior for ComfyUI video component nodes.

Hand-derived expectations from ComfyUI comfy_extras/nodes_video.py,
CreateVideo/GetVideoComponents/VideoSlice at immutable revision
bc5f880dbdd43e426077f42ccec53bb577ffee9f. No captured F constants.
The exact source fixture is arange(3*2*4*C)/(3*2*4*C), shape (3,2,4,C),
C=3 or 4; audio is arange(12) at 6 Hz. Create/get preserve these inputs.
Auto depth is 10 for HDR/HDR PQ and 8 for sRGB; explicit depth wins.
At 2 FPS, start=-1/duration=0 retains the last two of three frames.
At 3 FPS, a 0.5-second edit retains round(0.5*6)=3 audio samples.
The start=1/duration=1 request exceeds a three-frame 2-FPS clip.
Requested auto/h264 encoding uses the source constructor's MKV/H.264 path.
"""

from __future__ import annotations

from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from dinkster_api.v1 import (
    AssetRef,
    AssetVault,
    AssetWidget,
    coerce_audio,
    digest_bytes,
    effective_video_facts,
    save_video_stream,
)
from dinkster_nodes_comfy.video_components import CreateVideo, GetVideoComponents, VideoSlice
from dinkster_nodes_comfy.video_load import LoadVideo


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


def test_load_video_uses_bound_file_without_changing_source(tmp_path: Path) -> None:
    # LoadVideo at ComfyUI comfy_extras/nodes_video.py, immutable
    # bc5f880dbdd43e426077f42ccec53bb577ffee9f, preserves its source video.
    # Hand-derived fixture facts: three 2x4 frames at 2 FPS last 1.5 seconds.
    # Bound ASSET input and source-filename annotation are Dinkster contracts.
    buffer = BytesIO()
    save_video_stream(
        CreateVideo.execute(images=_images(), fps=2.0)["VIDEO"],
        buffer,
        container="mkv",
        codec="ffv1",
    )
    data = buffer.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    source = AssetRef(digest, "input.mkv", len(data), resolver=vault)
    video = LoadVideo.execute(file=source)["VIDEO"]
    facts = effective_video_facts(video)
    assert (facts["width"], facts["height"], facts["frame_count"]) == (4, 2, 3)
    assert facts["fps"] == Fraction(2)
    assert facts["duration"] == Fraction(3, 2)
    assert video["edits"] == []
    with source.open() as handle:
        assert handle.read() == data
    port = LoadVideo.schema().inputs[0]
    assert port.id == "file" and port.source_filename is not None
    assert isinstance(port.widget, AssetWidget) and port.widget.allow_upload
    with pytest.raises(TypeError, match="AssetRef"):
        LoadVideo.execute(file="unbound.mkv")
