"""Value-level video operation contracts: window, rate, assemble, disassemble."""

from __future__ import annotations

import io
from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_media_io import (
    AssembleVideo,
    DisassembleVideo,
    TrimVideo,
    VideoFrameRate,
    VideoFrameWindow,
)
from dinkster_nodes_media_io.video import decode_video_frames
from dinkster_values import video_from_source
from dinkster_video import save_video_stream


def _frames(count: int = 10, *, alpha: bool = False) -> np.ndarray:
    channels = 4 if alpha else 3
    result = np.zeros((count, 64, 64, channels), dtype=np.float32)
    for index in range(count):
        result[index, ..., 0] = index / max(1, count - 1)
        result[index, ..., 1] = 0.25
        result[index, ..., 2] = 0.75
        if alpha:
            result[index, ..., 3] = (index + 1) / count
    return result


def _audio(samples: int = 16_000, sample_rate: int = 16_000) -> dict[str, object]:
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * timeline, dtype=np.float32)[None, None, :]
    return {"waveform": waveform, "sample_rate": sample_rate}


def _frame_ids(images: object, count: int = 10) -> list[int]:
    batch = cast("np.ndarray", images)
    return [int(round(float(frame[0, 0, 0]) * (count - 1))) for frame in batch]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, list(range(10))),
        ({"skip_first_frames": 3}, [3, 4, 5, 6, 7, 8, 9]),
        ({"select_every_nth": 4}, [0, 4, 8]),
        ({"frame_load_cap": 2}, [0, 1]),
        ({"skip_first_frames": 2, "select_every_nth": 3}, [2, 5, 8]),
        ({"skip_first_frames": 2, "select_every_nth": 3, "frame_load_cap": 2}, [2, 5]),
        ({"skip_first_frames": 9, "frame_load_cap": 5}, [9]),
    ],
)
def test_window_applies_skip_then_stride_then_cap(
    kwargs: dict[str, Any], expected: list[int]
) -> None:
    result = VideoFrameWindow.execute(images=_frames(), **kwargs)
    assert _frame_ids(result["images"]) == expected
    assert result["frame_count"] == len(expected)
    assert cast("np.ndarray", result["images"]).flags["C_CONTIGUOUS"]


def test_window_empty_selection_is_an_error() -> None:
    with pytest.raises(ValueError, match="contains no frames"):
        VideoFrameWindow.execute(images=_frames(4), skip_first_frames=4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"skip_first_frames": -1}, "skip_first_frames"),
        ({"skip_first_frames": 1.0}, "skip_first_frames"),
        ({"skip_first_frames": True}, "skip_first_frames"),
        ({"select_every_nth": 0}, "select_every_nth"),
        ({"select_every_nth": "2"}, "select_every_nth"),
        ({"select_every_nth": True}, "select_every_nth"),
        ({"frame_load_cap": -1}, "frame_load_cap"),
        ({"frame_load_cap": 2.5}, "frame_load_cap"),
        ({"frame_load_cap": False}, "frame_load_cap"),
    ],
)
def test_window_rejects_non_integer_frame_controls(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        VideoFrameWindow.execute(images=_frames(), **kwargs)


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((0, 8, 8, 3), dtype=np.float32),
        np.zeros((2, 8, 8, 2), dtype=np.float32),
        np.zeros((8, 8, 3), dtype=np.float32),
        "frames",
    ],
)
def test_window_refuses_invalid_frame_batches(images: object) -> None:
    with pytest.raises(ValueError, match="frame batch"):
        VideoFrameWindow.execute(images=images)


@pytest.mark.parametrize(
    ("count", "fps", "target", "expected"),
    [
        # Downsample: 10 frames at 10 fps -> 5 fps keeps every other frame.
        (10, 10.0, 5.0, [0, 2, 4, 6, 8]),
        # Upsample: 5 frames at 5 fps -> 10 fps duplicates each frame.
        (5, 5.0, 10.0, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        # Tie at tick 0.5 s between frames 1 and 2 rounds to the earlier frame.
        (3, 3.0, 2.0, [0, 1]),
        # Non-integer tick count rounds up: ticks strictly before 5/4 s.
        (5, 4.0, 3.0, [0, 1, 3, 4]),
    ],
)
def test_rate_resamples_by_nearest_tick_with_earlier_frame_ties(
    count: int, fps: float, target: float, expected: list[int]
) -> None:
    result = VideoFrameRate.execute(images=_frames(count), fps=fps, target_fps=target)
    assert _frame_ids(result["images"], count) == expected
    assert result["frame_count"] == len(expected)
    assert result["fps"] == target
    assert result["duration"] == len(expected) / target
    assert cast("np.ndarray", result["images"]).flags["C_CONTIGUOUS"]


def test_rate_zero_target_passes_frames_through() -> None:
    frames = _frames(6)
    result = VideoFrameRate.execute(images=frames, fps=12.0, target_fps=0.0)
    assert np.array_equal(cast("np.ndarray", result["images"]), frames)
    assert result["frame_count"] == 6
    assert result["fps"] == 12.0
    assert result["duration"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0.0},
        {"fps": -24.0},
        {"fps": float("inf")},
        {"fps": 24.0, "target_fps": -1.0},
        {"fps": 24.0, "target_fps": 2000.0},
        {"fps": 24.0, "target_fps": float("nan")},
    ],
)
def test_rate_rejects_out_of_domain_rates(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        VideoFrameRate.execute(images=_frames(4), **kwargs)


def test_rate_matches_load_video_force_rate_decode() -> None:
    # The rate node promises the exact frame selection dinkster.load_video
    # performs at decode time with force_rate: same nearest-tick rule, same
    # tie direction, same trailing fill before the source end.
    frames = _frames(12)
    components = AssembleVideo.execute(images=frames, fps=6.0)["video"]
    encoded = io.BytesIO()
    save_video_stream(components, encoded)

    class Source:
        def open(self) -> io.BytesIO:
            return io.BytesIO(encoded.getvalue())

    source = Source()
    forced, forced_fps, forced_duration = decode_video_frames(
        source,
        force_rate=2.5,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        start_time=0.0,
        select_every_nth=1,
    )
    decoded, source_fps, _ = decode_video_frames(
        source,
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        start_time=0.0,
        select_every_nth=1,
    )
    resampled = VideoFrameRate.execute(
        images=decoded,
        fps=source_fps,
        target_fps=2.5,
    )
    assert np.array_equal(forced, cast("np.ndarray", resampled["images"]))
    assert resampled["fps"] == forced_fps
    assert resampled["duration"] == forced_duration


def test_assemble_disassemble_round_trip_preserves_frames_fps_and_audio() -> None:
    frames = _frames(10)
    audio = _audio()
    video = cast(
        "dict[str, Any]",
        AssembleVideo.execute(images=frames, fps=24.0, audio=audio)["video"],
    )
    assert video["probe"]["container"] is None
    assert video["components"]["images"] is frames
    result = DisassembleVideo.execute(video=video)
    images = cast("np.ndarray", result["images"])
    assert images.shape == frames.shape
    assert result["frame_count"] == 10
    assert result["fps"] == 24.0
    assert result["duration"] == 10 / 24.0
    assert np.array_equal(images, frames)
    decoded_audio = cast("dict[str, Any]", result["audio"])
    waveform = cast("np.ndarray", decoded_audio["waveform"])
    assert decoded_audio["sample_rate"] == 16_000
    assert waveform.shape[0] == 1 and waveform.shape[1] == 1
    assert waveform.shape[2] == 6667


def test_round_trip_preserves_fractional_frame_rates_exactly() -> None:
    video = cast(
        "dict[str, Any]",
        AssembleVideo.execute(images=_frames(12), fps=29.97)["video"],
    )
    result = DisassembleVideo.execute(video=video)
    assert result["fps"] == 29.97
    assert result["duration"] == 12 / 29.97


def test_disassemble_reports_absent_audio_for_silent_video() -> None:
    video = AssembleVideo.execute(images=_frames(4), fps=8.0)["video"]
    result = DisassembleVideo.execute(video=video)
    assert not isinstance(result["audio"], dict)


def test_trim_zero_window_preserves_components_without_encoding() -> None:
    video = cast("dict[str, Any]", AssembleVideo.execute(images=_frames(4), fps=8.0)["video"])
    result = cast("dict[str, Any]", TrimVideo.execute(video=video)["video"])
    assert result["probe"] == video["probe"]
    assert result["components"]["images"] is video["components"]["images"]
    assert video["edits"] == []


def test_trim_zero_window_preserves_container_bytes() -> None:
    encoded = io.BytesIO()
    save_video_stream(AssembleVideo.execute(images=_frames(4), fps=8.0)["video"], encoded)
    value = video_from_source(encoded.getvalue())
    trimmed = TrimVideo.execute(video=value)["video"]
    saved = io.BytesIO()
    save_video_stream(trimmed, saved)
    assert saved.getvalue() == encoded.getvalue()


def test_trim_selects_requested_frame_window() -> None:
    video = AssembleVideo.execute(images=_frames(10), fps=10.0)["video"]
    trimmed = TrimVideo.execute(video=video, start_time=0.2, duration=0.3)["video"]
    result = DisassembleVideo.execute(video=trimmed)
    assert result["frame_count"] == 3
    assert result["fps"] == 10.0
    assert result["duration"] == pytest.approx(0.3)


def test_trim_strict_duration_refuses_short_source() -> None:
    video = AssembleVideo.execute(images=_frames(4), fps=8.0)["video"]
    with pytest.raises(ValueError, match="requested duration"):
        TrimVideo.execute(video=video, start_time=0.25, duration=1.0, strict_duration=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fps": 0.0}, "fps"),
        ({"bit_depth": "12"}, "bit_depth"),
        ({"color_space": "invalid"}, "color_space"),
    ],
)
def test_assemble_rejects_out_of_domain_controls(kwargs: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {"images": _frames(4), **kwargs}
    with pytest.raises(ValueError, match=match):
        AssembleVideo.execute(**arguments)


@pytest.mark.parametrize(
    "video",
    [
        None,
        b"raw bytes",
        {"container": "mp4"},
        {"container": "avi", "bytes": b"x"},
        {"container": "webm", "bytes": b"\x00\x00\x00\x18ftypisom"},
        {"container": "mp4", "bytes": b"not a container"},
    ],
)
def test_disassemble_rejects_malformed_video_values(video: object) -> None:
    with pytest.raises(ValueError, match="not a valid comfy.VIDEO value"):
        DisassembleVideo.execute(video=video)


def test_video_ops_schemas_preserve_value_and_preview_contracts() -> None:
    window = VideoFrameWindow.schema()
    rate = VideoFrameRate.schema()
    assemble = AssembleVideo.schema()
    disassemble = DisassembleVideo.schema()
    trim = TrimVideo.schema()
    assert window.node_type == "dinkster.video.window"
    assert rate.node_type == "dinkster.video.rate"
    assert assemble.node_type == "dinkster.video.assemble"
    assert disassemble.node_type == "dinkster.video.disassemble"
    assert trim.node_type == "dinkster.video.trim"
    for schema in (window, rate, assemble, disassemble, trim):
        assert schema.category == "video"
        assert schema.output_node is False
        assert schema.outputs[0].preview is True
    assert assemble.aliases == ("CreateVideo",)
    for schema in (window, rate, disassemble, trim):
        assert schema.aliases == ()
    # Assemble and disassemble exchange plain comfy.VIDEO values, not assets.
    assert assemble.outputs[0].type.kind == "concrete"
    assert assemble.outputs[0].type.types == ("comfy.VIDEO",)
    assert disassemble.inputs[0].type.kind == "concrete"
    assert disassemble.inputs[0].type.types == ("comfy.VIDEO",)
    assert [(output.id, output.preview, output.optional) for output in disassemble.outputs] == [
        ("images", True, False),
        ("frame_count", False, False),
        ("audio", True, True),
        ("fps", False, False),
        ("duration", False, False),
        ("bit_depth", False, False),
        ("color_space", False, False),
    ]
