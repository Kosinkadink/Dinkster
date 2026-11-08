"""Retained VIDEO codecs, edit transport, assembly and disassembly contracts."""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import (
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
from dinkster_values import video_from_source
from dinkster_video_runtime import save_video_stream

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario


def test_native_video_registration_is_torch_free_and_preserves_first_authority() -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_api.v1 import TypeRegistry, VIDEO_TYPE, VIDEO_EDIT_TYPE; "
            "from dinkster_video import register_types; "
            "r=TypeRegistry(); register_types(r); "
            "assert VIDEO_TYPE in r and VIDEO_EDIT_TYPE in r; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
    from dinkster_api.v1 import TypeRegistry, register_video_edit_type
    from dinkster_inference_wire.types import register_inference_types
    from dinkster_video import register_types

    for first, second in (
        (register_inference_types, register_types),
        (register_types, register_inference_types),
    ):
        registry = TypeRegistry()
        first(registry)
        edit_spec = registry.spec(VIDEO_EDIT_TYPE)
        second(registry)
        video_spec = registry.spec(VIDEO_TYPE)
        register_types(registry)
        register_video_edit_type(registry)
        assert registry.spec(VIDEO_TYPE) is video_spec
        assert registry.spec(VIDEO_EDIT_TYPE) is edit_spec

    registry = TypeRegistry()
    registry.register(VIDEO_TYPE, encode=lambda _: b"video", decode=lambda _: "video")
    registry.register(VIDEO_EDIT_TYPE, encode=lambda _: b"edit", decode=lambda _: "edit")
    video_spec = registry.spec(VIDEO_TYPE)
    edit_spec = registry.spec(VIDEO_EDIT_TYPE)
    register_types(registry)
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

    value = {
        "crop": {"x": 5, "y": 3, "width": 11, "height": 7},
        "trim": {"start_time": 0.25, "duration": 0.25},
        "vendor": {"tag": "retain", "enabled": False},
    }
    registry = TypeRegistry()
    first = register_inference_types(registry)
    spec = registry.spec(VIDEO_EDIT_TYPE)
    assert spec.declared_codec
    assert spec.encode(value) == (
        b'{"crop": {"height": 7, "width": 11, "x": 5, "y": 3}, '
        b'"trim": {"duration": 0.25, "start_time": 0.25}, '
        b'"vendor": {"enabled": false, "tag": "retain"}}'
    )
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
            outputs=(OutputSpec("edit", TypeExpr.concrete(VIDEO_EDIT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, edit: object) -> Mapping[str, object]:
        assert edit == {
            "crop": {"x": 5, "y": 3, "width": 11, "height": 7},
            "trim": {"start_time": 0.25, "duration": 0.25},
            "vendor": {"tag": "retain", "enabled": False},
        }
        fields = cast(dict[str, object], edit)
        images = np.arange(2 * 12 * 18 * 3, dtype=np.float32).reshape(2, 12, 18, 3) / 2000
        result = edit_video(assemble_video(images, fps=4), {"crop": fields["crop"]})
        np.testing.assert_array_equal(disassemble_video(result)["images"], images[:, 2:8, 4:14])
        trimmed = edit_video(result, {"trim": fields["trim"]})
        np.testing.assert_array_equal(disassemble_video(trimmed)["images"], images[1:, 2:8, 4:14])
        return cls.outputs(edit=edit)


@pytest.mark.parametrize("in_process", (True, False))
def test_video_edit_literal_crosses_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        checkout = Path(__file__).resolve().parents[1]
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
        manifests = (fixture,)
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
            value = {
                "crop": {"x": 5, "y": 3, "width": 11, "height": 7},
                "trim": {"start_time": 0.25, "duration": 0.25},
                "vendor": {"tag": "retain", "enabled": False},
            }
            graph = Graph(nodes={"apply": GraphNode("video-edit-public.apply", {"edit": value})})
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["apply"])
                assert result.outputs["apply"]["edit"].resolve() == value
                assert set(result.cached) == (set() if attempt == 0 else {"apply"})
        finally:
            await composer.close()

    run_scenario(scenario())


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


def test_assemble_disassemble_round_trip_preserves_frames_fps_and_audio() -> None:
    frames = _frames(10)
    audio = _audio()
    video = cast(
        "dict[str, Any]",
        assemble_video(frames, fps=24.0, audio=audio),
    )
    assert video["probe"]["container"] is None
    assert video["components"]["images"] is frames
    result = disassemble_video(video)
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
        assemble_video(_frames(12), fps=29.97),
    )
    result = disassemble_video(video)
    assert result["fps"] == 29.97
    assert result["duration"] == 12 / 29.97


def test_disassemble_reports_absent_audio_for_silent_video() -> None:
    video = assemble_video(_frames(4), fps=8.0)
    result = disassemble_video(video)
    assert result["audio"] is None


def test_trim_zero_window_preserves_components_without_encoding() -> None:
    video = cast("dict[str, Any]", assemble_video(_frames(4), fps=8.0))
    result = cast("dict[str, Any]", edit_video(video, {"trim": {"start_time": 0, "duration": 0}}))
    assert result["probe"] == video["probe"]
    assert result["components"]["images"] is video["components"]["images"]
    assert video["edits"] == []


def test_trim_zero_window_preserves_container_bytes() -> None:
    encoded = io.BytesIO()
    save_video_stream(assemble_video(_frames(4), fps=8.0), encoded)
    value = video_from_source(encoded.getvalue())
    trimmed = edit_video(value, {"trim": {"start_time": 0, "duration": 0}})
    saved = io.BytesIO()
    save_video_stream(trimmed, saved)
    assert saved.getvalue() == encoded.getvalue()


def test_trim_selects_requested_frame_window() -> None:
    frames = _frames(10)
    video = assemble_video(frames, fps=10.0)
    trimmed = edit_video(video, {"trim": {"start_time": 0.2, "duration": 0.3}})
    result = disassemble_video(trimmed)
    np.testing.assert_array_equal(result["images"], frames[2:5])
    assert result["frame_count"] == 3
    assert result["fps"] == 10.0
    assert result["duration"] == pytest.approx(0.3)


def test_trim_strict_duration_refuses_short_source() -> None:
    video = assemble_video(_frames(4), fps=8.0)
    with pytest.raises(ValueError, match="requested duration"):
        edit_video(video, {"trim": {"start_time": 0.25, "duration": 1.0}, "strict_duration": True})


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
        assemble_video(**arguments)


@pytest.mark.parametrize(
    "video,match",
    [
        (None, "VIDEO must be an object"),
        (b"raw bytes", "VIDEO must be an object"),
        ({"container": "mp4"}, "VIDEO requires exactly one of source and components"),
        ({"container": "avi", "bytes": b"x"}, "Invalid data found when processing input"),
        (
            {"container": "webm", "bytes": b"\x00\x00\x00\x18ftypisom"},
            "Invalid data found when processing input",
        ),
        (
            {"container": "mp4", "bytes": b"not a container"},
            "Invalid data found when processing input",
        ),
    ],
)
def test_disassemble_rejects_malformed_video_values(video: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        disassemble_video(video)
