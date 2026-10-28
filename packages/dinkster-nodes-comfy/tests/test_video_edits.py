"""Observable component and source video editing contracts."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    assemble_video,
    audio_window,
    disassemble_video,
    effective_audio_facts,
    effective_video_facts,
    save_video_stream,
    video_from_source,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.video_edits import VideoCrop, VideoSlice, VideoTrim

from dinkster.compose import PackSpec, ServingComposer


def test_crop_schema_and_cold_import() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from dinkster_api.v1 import (
    CORE_BOOLEAN, CORE_FLOAT, VIDEO_TYPE, VIDEO_EDIT_TYPE, NumberWidget, TypeExpr,
)
from dinkster_nodes_comfy.video_edits import VideoCrop, VideoSlice, VideoTrim
s = VideoCrop.schema()
assert (s.node_type, s.display_name, s.category) == ('VideoCrop', 'Crop Video', 'video')
assert s.output_node and s.editor_role == 'video-crop'
assert [(i.id, i.type) for i in s.inputs] == [
    ('video', TypeExpr.concrete(VIDEO_TYPE)), ('crop', TypeExpr.concrete(VIDEO_EDIT_TYPE))
]
assert s.inputs[1].widget.widget_type == 'VIDEO_EDIT'
assert s.inputs[1].widget.params == {'features': ('crop',)}
assert s.inputs[1].default == {'crop': {'x': 0, 'y': 0, 'width': 0, 'height': 0}}
assert [(o.id, o.type, o.preview) for o in s.outputs] == [
    ('video', TypeExpr.concrete(VIDEO_TYPE), True)
]
s = VideoSlice.schema()
assert (s.node_type, s.display_name, s.category) == ('Video Slice', 'Trim Video', 'video')
assert not s.output_node
assert [(i.id, i.type) for i in s.inputs] == [
    ('video', TypeExpr.concrete(VIDEO_TYPE)),
    ('start_time', TypeExpr.concrete(CORE_FLOAT)),
    ('duration', TypeExpr.concrete(CORE_FLOAT)),
    ('strict_duration', TypeExpr.concrete(CORE_BOOLEAN)),
]
assert [i.default for i in s.inputs[1:]] == [0.0, 0.0, False]
assert s.inputs[1].widget == NumberWidget(min=-1e5, max=1e5, step=0.001)
assert s.inputs[2].widget == NumberWidget(min=0.0, step=0.001)
assert [(o.id, o.type) for o in s.outputs] == [('video', TypeExpr.concrete(VIDEO_TYPE))]
s = VideoTrim.schema()
assert (s.node_type, s.display_name, s.category) == ('VideoTrim', 'Trim Video (Advanced)', 'video')
assert s.output_node and s.editor_role == 'video-trim'
assert [(i.id, i.type) for i in s.inputs] == [
    ('video', TypeExpr.concrete(VIDEO_TYPE)),
    ('trim', TypeExpr.concrete(VIDEO_EDIT_TYPE)),
    ('strict_duration', TypeExpr.concrete(CORE_BOOLEAN)),
]
assert s.inputs[1].default == {'trim': {'start_time': 0.0, 'duration': 0.0}}
assert s.inputs[1].widget.widget_type == 'VIDEO_EDIT'
assert s.inputs[1].widget.params == {'features': ('trim',)}
assert s.inputs[2].default is False and s.inputs[2].advanced
assert [(o.id, o.type, o.preview) for o in s.outputs] == [
    ('video', TypeExpr.concrete(VIDEO_TYPE), True)
]
assert not any(n in sys.modules for n in ('torch', 'torchvision', 'av', 'dinkster_inference'))
""",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("rectangle", "expected"),
    [
        ({"x": 3, "y": 1, "width": 5, "height": 5}, (2, 0, 4, 4)),
        ({"x": -9, "y": 3, "width": 99, "height": 99}, (0, 2, 8, 4)),
        ({"x": 99, "y": 99, "width": 3, "height": 3}, (6, 4, 2, 2)),
        ({"x": 3, "width": 0, "height": 4}, (0, 0, 8, 6)),
        ({"width": 1, "height": 1}, (0, 0, 8, 6)),
        ({"width": 8, "height": 6}, (0, 0, 8, 6)),
    ],
)
def test_component_crop_keeps_exact_pixels_full_audio_and_source(
    rectangle: dict[str, int], expected: tuple[int, int, int, int]
) -> None:
    images = np.arange(2 * 6 * 8 * 3, dtype=np.float64).reshape(2, 6, 8, 3) / 301
    original = images.copy()
    waveform = np.array([[[0.2, -0.7, 0.9, 0.1, -0.4, 0.8, -0.2, 0.6]]], np.float32)
    video = assemble_video(
        images,
        fps=Fraction(30000, 1001),
        audio={"waveform": waveform, "sample_rate": 8},
        bit_depth="10",
        color_space="HDR PQ",
    )
    result = VideoCrop.execute(video=video, crop={"crop": rectangle})["video"]
    x, y, width, height = expected
    components = result["components"]
    np.testing.assert_array_equal(components["images"], original[:, y : y + height, x : x + width])
    assert components["images"].dtype == np.float64
    assert components["fps"] == Fraction(30000, 1001)
    assert components["bit_depth"] == 10
    assert components["color_space"] == ("HDR PQ" if (width, height) == (8, 6) else "sRGB")
    assert result["edits"] == []
    assert effective_audio_facts(components["audio"])["frames"] == 8
    np.testing.assert_array_equal(audio_window(components["audio"], 0, 8)["waveform"], waveform)
    np.testing.assert_array_equal(images, original)
    if (width, height) != (8, 6):
        assert not np.shares_memory(components["images"], images)


def test_encoded_crop_is_lazy_and_composes_on_display_pixels() -> None:
    images = (np.arange(3 * 16 * 24 * 3).reshape(3, 16, 24, 3) % 251).astype(np.uint8)
    encoded = io.BytesIO()
    save_video_stream(assemble_video(images, fps=4), encoded, container="mkv", codec="h264")
    video = video_from_source(encoded.getvalue())
    original = disassemble_video(video)["images"]
    first = VideoCrop.execute(
        video=video, crop={"crop": {"x": 3, "y": 5, "width": 12, "height": 10}}
    )["video"]
    result = VideoCrop.execute(
        video=first, crop={"crop": {"x": 3, "y": 1, "width": 6, "height": 6}}
    )["video"]
    assert result["source"] == video["source"]
    assert "components" not in result
    assert len(result["edits"]) == 2
    facts = effective_video_facts(result)
    assert (facts["width"], facts["height"], facts["frame_count"]) == (6, 6, 3)
    np.testing.assert_array_equal(disassemble_video(result)["images"], original[:, 4:10, 4:10])


def _source_components_encoded(images: np.ndarray) -> bytes:
    encoded = io.BytesIO()
    with av.open(encoded, mode="w", format="mp4", options={"movflags": "use_metadata_tags"}) as mux:
        stream = mux.add_stream("libx264", rate=4)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        stream.codec_context.colorspace = stream.codec_context.color_range = 1
        for samples in images:
            frame = av.VideoFrame.from_ndarray(
                np.clip(samples * 255, 0, 255).astype(np.uint8), format="rgb24"
            ).reformat(format="yuv420p", dst_colorspace=1)
            frame.color_primaries, frame.color_trc = 1, 13
            frame.colorspace = frame.color_range = 1
            mux.mux(stream.encode(frame))
        mux.mux(stream.encode())
    return encoded.getvalue()


def test_component_trim_encodes_before_selection_but_advanced_zero_window_is_noop() -> None:
    images = np.linspace(0, 1, 8 * 16 * 24 * 3, dtype=np.float32).reshape(8, 16, 24, 3)
    before = images.copy()
    video = assemble_video(images, fps=4)
    untouched = VideoTrim.execute(video=video, trim=None)["video"]
    assert untouched["components"]["images"] is images
    sliced = VideoSlice.execute(video=video, start_time=0.5, duration=0.75)["video"]
    advanced = VideoTrim.execute(video=video, trim={"trim": {"start_time": 0.5, "duration": 0.75}})[
        "video"
    ]
    decoded = disassemble_video(video_from_source(_source_components_encoded(images)))["images"]
    for result in (sliced, advanced):
        assert "components" not in result
        assert result["probe"]["container"] == "mp4"
        np.testing.assert_array_equal(disassemble_video(result)["images"], decoded[2:5])
    assert "source" in VideoSlice.execute(video=video)["video"]
    np.testing.assert_array_equal(images, before)


@pytest.mark.parametrize("strict", (False, True))
def test_component_trim_rejects_source_overrun_even_without_strict_duration(strict: bool) -> None:
    video = assemble_video(np.zeros((8, 16, 24, 3), np.float32), fps=4)
    with pytest.raises(ValueError, match="Failed to slice video"):
        VideoSlice.execute(video=video, start_time=0.75, duration=1.5, strict_duration=strict)
    with pytest.raises(ValueError, match="Failed to trim video"):
        VideoTrim.execute(
            video=video,
            trim={"trim": {"start_time": 0.75, "duration": 1.5}},
            strict_duration=strict,
        )


def test_encoded_trim_is_tail_relative_and_replaces_previous_duration() -> None:
    images = np.linspace(0, 1, 8 * 16 * 24 * 3, dtype=np.float32).reshape(8, 16, 24, 3)
    encoded = io.BytesIO()
    save_video_stream(
        assemble_video(images, fps=4), encoded, container="mp4", source_components=True
    )
    video = video_from_source(encoded.getvalue())
    original = disassemble_video(video)["images"]
    tail = VideoSlice.execute(video=video, start_time=-0.75, duration=0.5)["video"]
    np.testing.assert_array_equal(disassemble_video(tail)["images"], original[5:7])
    first = VideoSlice.execute(video=video, start_time=0.25, duration=0.5)["video"]
    result = VideoSlice.execute(video=first, start_time=0.5, duration=0)["video"]
    assert result["source"] == video["source"]
    np.testing.assert_array_equal(disassemble_video(result)["images"], original[3:])
    clipped = VideoSlice.execute(video=video, start_time=1.5, duration=2)["video"]
    np.testing.assert_array_equal(disassemble_video(clipped)["images"], original[6:])
    with pytest.raises(ValueError):
        VideoSlice.execute(video=video, start_time=1.5, duration=2, strict_duration=True)
    with pytest.raises(ValueError):
        VideoSlice.execute(video=video, start_time=2, duration=0)
    with pytest.raises(ValueError, match="Trim duration must be >= 0"):
        VideoTrim.execute(video=video, trim={"trim": {"duration": -0.5}})


class CropSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-crop-fixture.source",
            outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            video=assemble_video(
                np.arange(2 * 6 * 8 * 3, dtype=np.float32).reshape(2, 6, 8, 3) / 301,
                fps=Fraction(30000, 1001),
                audio={"waveform": np.array([[[0.25, -0.75, 0.5]]]), "sample_rate": 8},
                bit_depth="10",
                color_space="HDR PQ",
            )
        )


class CropConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-crop-fixture.consumer",
            inputs=(InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, video: object) -> dict[str, object]:
        parts = disassemble_video(video)
        original = np.arange(2 * 6 * 8 * 3, dtype=np.float32).reshape(2, 6, 8, 3) / 301
        np.testing.assert_array_equal(parts["images"], original[:, :4, 2:6])
        assert parts["fps"] == float(Fraction(30000, 1001))
        assert parts["bit_depth"] == 10
        assert parts["color_space"] == "sRGB"
        assert video["components"]["fps"] == Fraction(30000, 1001)
        audio = video["components"]["audio"]
        assert effective_audio_facts(audio)["frames"] == 3
        np.testing.assert_array_equal(
            audio_window(audio, 0, 3)["waveform"], np.array([[[0.25, -0.75, 0.5]]], np.float32)
        )
        assert effective_video_facts(video)["frame_count"] == 2
        return cls.outputs(summary="crop-preserved")


class TrimSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-trim-fixture.source",
            outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            video=assemble_video(
                np.linspace(0, 1, 8 * 16 * 24 * 3, dtype=np.float32).reshape(8, 16, 24, 3),
                fps=4,
            )
        )


class TrimConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-trim-fixture.consumer",
            inputs=(InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, video: object) -> dict[str, object]:
        original = np.linspace(0, 1, 8 * 16 * 24 * 3, dtype=np.float32).reshape(8, 16, 24, 3)
        encoded = _source_components_encoded(original)
        expected = disassemble_video(video_from_source(encoded))["images"][2:5]
        np.testing.assert_array_equal(disassemble_video(video)["images"], expected)
        facts = effective_video_facts(video)
        assert facts["duration"] == Fraction(3, 4)
        assert facts["frame_count"] == 3
        return cls.outputs(summary="trim-preserved")


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("operation", ("VideoCrop", "Video Slice", "VideoTrim"))
def test_edits_cross_public_hosts_twice(tmp_path: Path, in_process: bool, operation: str) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        original = native.read_bytes()
        module = f"video_edit_fixture_{operation.replace(' ', '_')}_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_nodes_comfy import register_inference_types\n"
            "from test_video_edits import CropSource, CropConsumer, TrimSource, TrimConsumer\n"
            "NODES = (CropSource, CropConsumer, TrimSource, TrimConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "video-crop-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["video-crop-fixture", "video-trim-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native, fixture)
        paths = (tmp_path, package / "tests", checkout / "src", *checkout.glob("packages/*/src"))
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
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="video-crop", group_manifests=manifests
                    )
                )
            crop = operation == "VideoCrop"
            prefix = "video-crop-fixture" if crop else "video-trim-fixture"
            edit_inputs: dict[str, object] = {"video": Link("source", "video")}
            if crop:
                edit_inputs["crop"] = {"crop": {"x": 3, "y": 1, "width": 5, "height": 5}}
            elif operation == "VideoTrim":
                edit_inputs["trim"] = {"trim": {"start_time": 0.5, "duration": 0.75}}
            else:
                edit_inputs.update(start_time=0.5, duration=0.75)
            graph = Graph(
                nodes={
                    "source": GraphNode(f"{prefix}.source"),
                    "edit": GraphNode(operation, edit_inputs),
                    "consumer": GraphNode(f"{prefix}.consumer", {"video": Link("edit", "video")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                expected = "crop-preserved" if crop else "trim-preserved"
                assert result.outputs["consumer"]["summary"].resolve() == expected
                assert set(result.cached) == (set() if attempt == 0 else set(graph.nodes))
        finally:
            await composer.close()
        assert native.read_bytes() == original

    asyncio.run(scenario())
