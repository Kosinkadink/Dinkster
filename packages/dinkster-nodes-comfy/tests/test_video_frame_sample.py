"""Native VIDEO selection preserves order, timing and bounded source ownership."""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
from dinkster_api.v1 import (
    AssetRef,
    AssetVault,
    assemble_video,
    digest_bytes,
    disassemble_video,
    effective_video_facts,
    save_video_stream,
    video_from_source,
)
from dinkster_nodes_comfy.video_frame_sample import VideoFrameSample


def _pixels() -> np.ndarray:
    return (np.arange(7 * 4 * 6 * 3, dtype=np.uint16).reshape(7, 4, 6, 3) % 251).astype(np.uint8)


@pytest.mark.parametrize(
    ("strategy", "count", "seed", "indices"),
    [
        ("uniform", 3, 0, [0, 3, 6]),
        ("uniform", 1, 0, [3]),
        ("uniform", 99, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("random", 3, 19, [1, 3, 4]),
        ("random", 3, 2**63 + 7, [1, 2, 6]),
    ],
)
def test_noncontiguous_selection_preserves_exact_pixels_source_rate_and_drops_audio(
    strategy: str, count: int, seed: int, indices: list[int]
) -> None:
    rate = Fraction(30000, 1001)
    video = assemble_video(
        _pixels(),
        fps=rate,
        audio={"waveform": np.ones((1, 1, 1120), np.float32), "sample_rate": 4800},
    )
    result = VideoFrameSample.execute(video=video, num_frames=count, strategy=strategy, seed=seed)[
        "video"
    ]
    facts = effective_video_facts(result)
    assert facts["fps"] == rate
    assert facts["frame_count"] == len(indices)
    decoded = disassemble_video(result)
    assert decoded["audio"] is None
    np.testing.assert_array_equal(decoded["images"], _pixels()[indices].astype(np.float32) / 255)


@pytest.mark.parametrize(("strategy", "start"), [("head", 0), ("tail", 4)])
def test_contiguous_selection_retains_lazy_source_and_audio_without_decoding(
    strategy: str, start: int
) -> None:
    audio = {"waveform": np.ones((1, 1, 1120), np.float32), "sample_rate": 4800}
    video = assemble_video(_pixels(), fps=Fraction(5), audio=audio)
    result = VideoFrameSample.execute(video=video, num_frames=3, strategy=strategy, seed=0)["video"]
    assert result["components"]["images"] is video["components"]["images"]
    retained_audio = result["components"]["audio"]
    assert np.shares_memory(retained_audio["waveform"], audio["waveform"])
    np.testing.assert_array_equal(retained_audio["waveform"], audio["waveform"])
    assert retained_audio["sample_rate"] == audio["sample_rate"]
    assert result["edits"] == [
        {
            "trim": {"start_time": Fraction(start, 5), "duration": Fraction(3, 5)},
            "strict_duration": False,
        }
    ]


def test_selected_encoded_frames_release_asset_and_match_requested_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = io.BytesIO()
    save_video_stream(
        assemble_video(_pixels(), fps=Fraction(5)), encoded, codec="ffv1", container="mkv"
    )
    data = encoded.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    video = video_from_source(AssetRef(digest, "sample.mkv", len(data), resolver=vault))
    opened: list[BinaryIO] = []
    original = AssetRef.open

    def track_open(ref: AssetRef) -> BinaryIO:
        handle = original(ref)
        opened.append(handle)
        return handle

    monkeypatch.setattr(AssetRef, "open", track_open)
    result = VideoFrameSample.execute(video=video, num_frames=3, strategy="random", seed=19)[
        "video"
    ]
    assert opened and all(handle.closed for handle in opened)
    np.testing.assert_array_equal(
        disassemble_video(result)["images"], _pixels()[[1, 3, 4]].astype(np.float32) / 255
    )


@pytest.mark.parametrize(("strategy", "indices"), [("head", [0, 1, 2]), ("tail", [4, 5, 6])])
def test_fractional_rate_encoded_windows_select_exact_requested_frames(
    tmp_path: Path, strategy: str, indices: list[int]
) -> None:
    pixels = (np.arange(7 * 4 * 32 * 3).reshape(7, 4, 32, 3) % 251).astype(np.uint8)
    encoded = io.BytesIO()
    save_video_stream(
        assemble_video(pixels, fps=Fraction(30000, 1001)), encoded, codec="ffv1", container="mkv"
    )
    data = encoded.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    video = video_from_source(AssetRef(digest, "fractional.mkv", len(data), resolver=vault))
    result = VideoFrameSample.execute(video=video, num_frames=3, strategy=strategy, seed=0)["video"]
    assert effective_video_facts(result)["fps"] == Fraction(30000, 1001)
    np.testing.assert_array_equal(disassemble_video(result)["images"], pixels[indices])
