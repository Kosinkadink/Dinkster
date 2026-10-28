"""Value-level video operation contracts: window, rate, assemble, disassemble."""

from __future__ import annotations

import asyncio
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    VIDEO_EDIT_TYPE,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    assemble_video,
    disassemble_video,
    edit_video,
)
from dinkster_graph import Graph, GraphNode
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

from dinkster.compose import PackSpec, ServingComposer


def test_native_video_registration_is_torch_free_and_preserves_first_authority() -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_api.v1 import TypeRegistry, VIDEO_TYPE, VIDEO_EDIT_TYPE; "
            "from dinkster_nodes_comfy import register_inference_types; "
            "r=TypeRegistry(); register_inference_types(r); "
            "assert VIDEO_TYPE in r and VIDEO_EDIT_TYPE in r; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
    from dinkster_api.v1 import TypeRegistry, register_video_edit_type
    from dinkster_nodes_comfy import register_inference_types
    from dinkster_nodes_media_io import register_media_types

    for first, second in (
        (register_inference_types, register_media_types),
        (register_media_types, register_inference_types),
    ):
        registry = TypeRegistry()
        first(registry)
        video_spec = registry.spec(VIDEO_TYPE)
        edit_spec = registry.spec(VIDEO_EDIT_TYPE)
        second(registry)
        register_video_edit_type(registry)
        assert registry.spec(VIDEO_TYPE) is video_spec
        assert registry.spec(VIDEO_EDIT_TYPE) is edit_spec

    registry = TypeRegistry()
    registry.register(VIDEO_TYPE, encode=lambda _: b"video", decode=lambda _: "video")
    registry.register(VIDEO_EDIT_TYPE, encode=lambda _: b"edit", decode=lambda _: "edit")
    video_spec = registry.spec(VIDEO_TYPE)
    edit_spec = registry.spec(VIDEO_EDIT_TYPE)
    register_inference_types(registry)
    assert registry.spec(VIDEO_TYPE) is video_spec
    assert registry.spec(VIDEO_EDIT_TYPE) is edit_spec


def test_shared_video_edit_registration_is_torch_free_and_preserves_first_authority() -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_api.v1 import TypeRegistry, VIDEO_EDIT_TYPE; "
            "from dinkster_inference_wire.types import register_inference_types; "
            "r=TypeRegistry(); specs=register_inference_types(r); "
            "assert VIDEO_EDIT_TYPE in r; assert r.spec(VIDEO_EDIT_TYPE) in specs; "
            "assert not any(n in sys.modules for n in ('torch','av','dinkster_inference'))",
        ],
        check=True,
    )
    from dinkster_api.v1 import TypeRegistry, register_video_edit_type
    from dinkster_inference_wire.types import register_inference_types

    value = {"crop": {"x": 5, "y": 3, "width": 11, "height": 7}}
    registry = TypeRegistry()
    first = register_inference_types(registry)
    spec = registry.spec(VIDEO_EDIT_TYPE)
    assert spec.declared_codec
    assert spec.encode(value) == b'{"crop": {"height": 7, "width": 11, "x": 5, "y": 3}}'
    assert spec.decode(spec.encode(value)) == value
    repeated = register_inference_types(registry)
    register_video_edit_type(registry)
    assert len(first) == len({item.type_id for item in first})
    assert [item.type_id for item in first] == [item.type_id for item in repeated]
    assert all(before is after for before, after in zip(first, repeated, strict=True))
    assert registry.spec(VIDEO_EDIT_TYPE) is spec

    registry = TypeRegistry()
    original = registry.register(VIDEO_EDIT_TYPE, encode=lambda _: b"first", decode=lambda _: value)
    specs = register_inference_types(registry)
    assert registry.spec(VIDEO_EDIT_TYPE) is original
    assert next(item for item in specs if item.type_id == VIDEO_EDIT_TYPE) is original
    assert original.encode(value) == b"first"


class VideoEditLiteralFixture(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-edit-public.apply",
            inputs=(InputSpec("edit", TypeExpr.concrete(VIDEO_EDIT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, edit: object) -> Mapping[str, object]:
        assert edit == {"crop": {"x": 5, "y": 3, "width": 11, "height": 7}}
        images = np.arange(2 * 12 * 18 * 3, dtype=np.float32).reshape(2, 12, 18, 3) / 2000
        result = edit_video(assemble_video(images, fps=4), cast(dict[str, object], edit))
        np.testing.assert_array_equal(disassemble_video(result)["images"], images[:, 2:8, 4:14])
        return cls.outputs(summary=json.dumps(edit, sort_keys=True))


@pytest.mark.parametrize("in_process", (True, False))
def test_video_edit_literal_crosses_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        checkout = Path(__file__).resolve().parents[1]
        native = checkout / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        native_bytes = native.read_bytes()
        module = "video_edit_public_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_video_ops import VideoEditLiteralFixture\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (VideoEditLiteralFixture,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "video-edit-public-dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "video-edit-public-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["video-edit-public"]\n\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native, fixture)
        paths = (
            tmp_path,
            checkout / "tests",
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
        )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            },
        )
        try:
            for manifest in manifests:
                pack = PackSpec(manifest=manifest, in_process=True)
                if not in_process:
                    pack = PackSpec(
                        manifest=manifest,
                        worker_group="video-edit-public",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            value = {"crop": {"x": 5, "y": 3, "width": 11, "height": 7}}
            graph = Graph(nodes={"apply": GraphNode("video-edit-public.apply", {"edit": value})})
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["apply"])
                assert result.outputs["apply"]["summary"].resolve() == json.dumps(
                    value, sort_keys=True
                )
                assert set(result.cached) == (set() if attempt == 0 else {"apply"})
        finally:
            await composer.close()
        assert native.read_bytes() == native_bytes

    asyncio.run(scenario())


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
    with pytest.raises(ValueError, match="not a valid VIDEO value"):
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
    # Assemble and disassemble exchange plain VIDEO values, not assets.
    assert assemble.outputs[0].type.kind == "concrete"
    assert assemble.outputs[0].type.types == (VIDEO_TYPE,)
    assert disassemble.inputs[0].type.kind == "concrete"
    assert disassemble.inputs[0].type.types == (VIDEO_TYPE,)
    assert trim.inputs[1].type.types == (VIDEO_EDIT_TYPE,)
    assert [(output.id, output.preview, output.optional) for output in disassemble.outputs] == [
        ("images", True, False),
        ("frame_count", False, False),
        ("audio", True, True),
        ("fps", False, False),
        ("duration", False, False),
        ("bit_depth", False, False),
        ("color_space", False, False),
    ]
