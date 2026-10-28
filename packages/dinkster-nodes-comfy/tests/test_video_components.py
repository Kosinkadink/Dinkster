"""CreateVideo and GetVideoComponents preserve public VIDEO components."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    IMAGE_TYPE,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    audio_window,
    concatenate_video,
    disassemble_video,
    effective_audio_facts,
    effective_video_facts,
    video_from_source,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.video_components import CreateVideo, GetVideoComponents

from dinkster.compose import PackSpec, ServingComposer

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_STRING = TypeExpr.concrete(CORE_STRING)
_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_FLOAT64 = np.dtype(np.float64)


def _images() -> np.ndarray:
    return np.array(
        [
            [
                [[0.03125, 0.125, 0.875], [0.15625, 0.25, 0.75]],
                [[0.28125, 0.375, 0.625], [0.40625, 0.5, 0.5]],
            ],
            [
                [[0.90625, 0.0625, 0.1875], [0.78125, 0.1875, 0.3125]],
                [[0.65625, 0.3125, 0.4375], [0.53125, 0.4375, 0.5625]],
            ],
        ],
        dtype=np.float64,
    )


def _audio() -> dict[str, object]:
    return {
        "waveform": np.array([[[0.125, -0.25, 0.375, -0.5, 0.625]]], dtype=np.float64),
        "sample_rate": 48_000,
    }


def _assert_components(
    images: object,
    audio: object,
    fps: object,
    bit_depth: object,
    color_space: object,
    *,
    expected_depth: int = 8,
    expected_color_space: str = "sRGB",
    image_dtype: np.dtype[Any] = _FLOAT64,
) -> None:
    assert isinstance(images, np.ndarray)
    assert images.dtype == image_dtype
    np.testing.assert_array_equal(images, _images().astype(image_dtype))
    assert isinstance(audio, Mapping)
    source = audio["source"]
    assert isinstance(source, Mapping)
    waveform = source["pcm"]
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, np.asarray(_audio()["waveform"], dtype=np.float32))
    assert audio["sample_rate"] == 48_000
    assert type(fps) is float
    assert fps == float(Fraction(30_000, 1_001))
    assert type(bit_depth) is int
    assert bit_depth == expected_depth
    assert type(color_space) is str
    assert color_space == expected_color_space


def test_video_component_schema_and_cold_import_use_only_public_types() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/video_components.py"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from dinkster_api.v1 import AUDIO_TYPE, CORE_COMBO, CORE_FLOAT, IMAGE_TYPE, VIDEO_TYPE, TypeExpr
import dinkster_nodes_comfy.video_components as module
from dinkster_nodes_comfy.video_components import ConcatenateVideo, CreateVideo, GetVideoComponents

assert Path(module.__file__).resolve() == Path(sys.argv[1])
create = CreateVideo.schema()
assert create.node_type == "CreateVideo"
assert create.display_name == "Create Video"
assert create.category == "video"
assert create.description == "Create a video from images."
assert create.search_terms == ("images to video",)
assert tuple(item.id for item in create.inputs) == (
    "images", "fps", "audio", "bit_depth", "color_space", "codec"
)
assert [item.type for item in create.inputs] == [
    TypeExpr.concrete(IMAGE_TYPE), TypeExpr.concrete(CORE_FLOAT), TypeExpr.concrete(AUDIO_TYPE),
    TypeExpr.concrete(CORE_COMBO), TypeExpr.concrete(CORE_COMBO), TypeExpr.concrete(CORE_COMBO),
]
assert create.inputs[1].default == 30.0
assert create.inputs[2].required is False
assert create.inputs[3].default == "auto"
assert tuple(option.value for option in create.inputs[3].widget.options) == ("auto", 8, 10)
assert create.inputs[4].default == "sRGB"
assert create.inputs[4].widget.options == ("sRGB", "HDR", "HDR PQ")
assert create.inputs[5].default == "none"
assert create.inputs[5].widget.options == ("none", "auto", "h264", "av1")
assert [(item.id, item.type) for item in create.outputs] == [
    ("video", TypeExpr.concrete(VIDEO_TYPE))
]
get = GetVideoComponents.schema()
assert get.node_type == "GetVideoComponents"
assert get.display_name == "Get Video Components"
assert get.category == "video"
assert get.description == "Extracts video frames, audio, frame rate, bit depth, and color space."
assert get.search_terms == ("extract frames", "split video", "video to images", "demux")
assert [(item.id, item.type) for item in get.inputs] == [
    ("video", TypeExpr.concrete(VIDEO_TYPE))
]
assert [(item.id, item.type) for item in get.outputs] == [
    ("images", TypeExpr.concrete(IMAGE_TYPE)),
    ("audio", TypeExpr.concrete(AUDIO_TYPE)),
    ("fps", TypeExpr.concrete(CORE_FLOAT)),
    ("bit_depth", TypeExpr.concrete(CORE_COMBO)),
    ("color_space", TypeExpr.concrete(CORE_COMBO)),
]
concat = ConcatenateVideo.schema()
assert concat.node_type == "ConcatenateVideo"
assert concat.display_name == "Concatenate Video"
assert concat.category == "video"
assert (
    concat.description
    == "Concatenates videos in order without decoding compatible encoded inputs."
)
assert len(concat.input_families) == 1
videos = concat.input_families[0]
assert videos.id == "videos"
assert videos.member_prefix == "video"
assert (videos.min_members, videos.max_members) == (1, 100)
assert [(item.id, item.type, item.doc) for item in videos.template] == [
    (
        "video",
        TypeExpr.list_of(TypeExpr.concrete(VIDEO_TYPE)),
        "VIDEO lists to concatenate in input order.",
    )
]
assert [(item.id, item.default, item.advanced) for item in concat.inputs] == [
    ("codec", "auto", True),
    ("complete_audio", None, True),
]
assert concat.inputs[0].widget.options == ("auto", "h264", "av1")
assert concat.inputs[1].type == TypeExpr.concrete(AUDIO_TYPE)
assert [(item.id, item.type) for item in concat.outputs] == [
    ("video", TypeExpr.concrete(VIDEO_TYPE))
]
assert "torch" not in sys.modules
assert "torchvision" not in sys.modules
assert "av" not in sys.modules
assert not any(
    name == "dinkster_inference" or name.startswith("dinkster_inference.")
    for name in sys.modules
)
""",
            str(source),
        ],
        check=True,
        cwd=root,
    )


@pytest.mark.parametrize(
    ("bit_depth", "color_space", "expected_depth"),
    (
        ("auto", "sRGB", 8),
        ("auto", "HDR", 10),
        ("auto", "HDR PQ", 10),
        (8, "HDR", 8),
        (10, "sRGB", 10),
    ),
)
def test_create_and_get_components_preserve_inputs_without_encoding(
    bit_depth: str | int, color_space: str, expected_depth: int
) -> None:
    images = _images()
    audio = _audio()
    images_before = images.copy()
    audio_before = np.asarray(audio["waveform"]).copy()
    video = CreateVideo.execute(
        images=images,
        fps=Fraction(30_000, 1_001),
        audio=audio,
        bit_depth=bit_depth,
        color_space=color_space,
        codec="none",
    )["video"]
    np.testing.assert_array_equal(video["components"]["images"], images)
    np.testing.assert_array_equal(
        video["components"]["audio"]["source"]["pcm"],
        np.asarray(audio["waveform"], dtype=np.float32),
    )
    assert video["components"]["fps"] == Fraction(30_000, 1_001)
    assert video["components"]["bit_depth"] == expected_depth
    assert video["components"]["color_space"] == color_space
    assert video["edits"] == []
    np.testing.assert_array_equal(images, images_before)
    np.testing.assert_array_equal(audio["waveform"], audio_before)

    result = GetVideoComponents.execute(video=video)
    _assert_components(
        result["images"],
        result["audio"],
        result["fps"],
        result["bit_depth"],
        result["color_space"],
        expected_depth=expected_depth,
        expected_color_space=color_space,
    )
    assert result["bit_depth"] == expected_depth
    assert result["color_space"] == color_space


def test_create_and_get_components_allow_no_audio_without_mutating_images() -> None:
    images = _images()
    before = images.copy()
    video = CreateVideo.execute(images=images, fps=24.0)["video"]
    result = GetVideoComponents.execute(video=video)
    np.testing.assert_array_equal(result["images"], images)
    assert result["audio"] is None
    assert result["fps"] == 24.0
    assert result["bit_depth"] == 8
    assert result["color_space"] == "sRGB"
    np.testing.assert_array_equal(images, before)


def test_get_components_keeps_component_audio_longer_than_video() -> None:
    images = np.array(
        [
            [[[0.125, 0.25, 0.375]]],
            [[[0.625, 0.75, 0.875]]],
        ],
        dtype=np.float32,
    )
    waveform = np.array(
        [[[0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0]]], dtype=np.float32
    )
    video = CreateVideo.execute(
        images=images,
        fps=4.0,
        audio={"waveform": waveform, "sample_rate": 8},
    )["video"]
    result = GetVideoComponents.execute(video=video)
    audio = result["audio"]
    assert audio is not None
    facts = effective_audio_facts(audio)
    assert facts["sample_rate"] == 8
    assert facts["frames"] == 8
    assert facts["duration"] == 1.0
    np.testing.assert_array_equal(audio_window(audio, 0, 8)["waveform"], waveform)


def test_h264_create_encodes_then_extracts_real_components() -> None:
    images = np.zeros((3, 16, 16, 3), dtype=np.float32)
    images[0, 2:6, 2:6, 0] = 1.0
    images[1, 6:10, 6:10, 1] = 1.0
    images[2, 10:14, 10:14, 2] = 1.0
    video = CreateVideo.execute(images=images, fps=Fraction(30_000, 1_001), codec="h264")["video"]
    assert "components" not in video
    assert video["probe"]["container"] == "mkv"
    assert video["probe"]["video_codec"] == "h264"
    result = GetVideoComponents.execute(video=video)
    decoded = result["images"]
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (3, 16, 16, 3)
    assert result["audio"] is None
    assert result["fps"] == float(Fraction(2997, 100))
    assert result["bit_depth"] == 8
    assert result["color_space"] == "sRGB"
    expected = io.BytesIO()
    with av.open(expected, mode="w", format="matroska") as output:
        stream = output.add_stream("libx264", rate=Fraction(2997, 100))
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 1
        for samples in images:
            frame = av.VideoFrame.from_ndarray(
                np.clip(samples * 255, 0, 255).astype(np.uint8), format="rgb24"
            ).reformat(format="yuv420p", dst_colorspace=1)
            frame.color_primaries = 1
            frame.color_trc = 13
            frame.colorspace = 1
            frame.color_range = 1
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    np.testing.assert_array_equal(
        decoded, disassemble_video(video_from_source(expected.getvalue()))["images"]
    )
    for index, channel in enumerate((0, 1, 2)):
        means = decoded[index].mean(axis=(0, 1))
        assert int(np.argmax(means)) == channel


class ComponentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-components-public.source",
            outputs=(
                OutputSpec("images", _IMAGE),
                OutputSpec("audio", _AUDIO),
                OutputSpec("fps", _FLOAT),
                OutputSpec("bit_depth", _COMBO),
                OutputSpec("color_space", _COMBO),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            images=_images(),
            audio=_audio(),
            fps=float(Fraction(30_000, 1_001)),
            bit_depth="auto",
            color_space="sRGB",
        )


class ComponentConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-components-public.consumer",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec("audio", _AUDIO),
                InputSpec("fps", _FLOAT),
                InputSpec("bit_depth", _COMBO),
                InputSpec("color_space", _COMBO),
            ),
            outputs=(OutputSpec("summary", _STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        audio: object,
        fps: object,
        bit_depth: object,
        color_space: object,
    ) -> dict[str, object]:
        _assert_components(
            images,
            audio,
            fps,
            bit_depth,
            color_space,
            image_dtype=np.dtype(np.float32),
        )
        return cls.outputs(summary="components-preserved")


class ConcatenateSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-components-public.concatenate-source",
            outputs=(
                OutputSpec("first", TypeExpr.list_of(_VIDEO)),
                OutputSpec("second", TypeExpr.list_of(_VIDEO)),
                OutputSpec("replacement", _AUDIO),
                OutputSpec("expected_with_audio", _VIDEO),
                OutputSpec("expected_without_audio", _VIDEO),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        replacement = {
            "waveform": np.tile(
                np.array([[[0.125, -0.25, 0.375, -0.5]]], dtype=np.float64), (1, 1, 6_000)
            ),
            "sample_rate": 48_000,
        }

        def video(colors: list[tuple[float, float, float]]) -> object:
            images = np.asarray([np.full((2, 2, 3), color, dtype=np.float32) for color in colors])
            return CreateVideo.execute(images=images, fps=12.0, codec="none")["video"]

        first = [
            video([(1.0, 0.0, 0.0)]),
            video([(0.0, 1.0, 0.0)] * 2),
        ]
        second = [video([(0.0, 0.0, 1.0)] * 3)]
        ordered = [*first, *second]
        return cls.outputs(
            first=first,
            second=second,
            replacement=replacement,
            expected_with_audio=concatenate_video(ordered, complete_audio=replacement),
            expected_without_audio=concatenate_video(ordered),
        )


class ConcatenateConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-components-public.concatenate-consumer",
            inputs=(InputSpec("video", _VIDEO), InputSpec("expected", _VIDEO)),
            outputs=(OutputSpec("summary", _STRING),),
        )

    @classmethod
    def execute(cls, *, video: object, expected: object) -> dict[str, object]:
        components = disassemble_video(video)
        expected_components = disassemble_video(expected)
        images = components["images"]
        expected_images = expected_components["images"]
        assert isinstance(images, np.ndarray)
        assert isinstance(expected_images, np.ndarray)
        assert images.shape == (6, 2, 2, 3)
        np.testing.assert_array_equal(images, expected_images)
        assert components["fps"] == 12.0
        facts = effective_video_facts(video)
        assert facts["frame_count"] == effective_video_facts(expected)["frame_count"]
        assert facts["duration"] == effective_video_facts(expected)["duration"]
        audio = components["audio"]
        expected_audio = expected_components["audio"]
        if expected_audio is None:
            assert audio is None
            return cls.outputs(summary="concatenated-in-document-order")
        assert isinstance(audio, Mapping)
        assert isinstance(expected_audio, Mapping)
        assert audio["sample_rate"] == expected_audio["sample_rate"] == 48_000
        source = audio["source"]
        expected_source = expected_audio["source"]
        assert isinstance(source, Mapping)
        assert isinstance(expected_source, Mapping)
        np.testing.assert_array_equal(source["pcm"], expected_source["pcm"])
        return cls.outputs(summary="concatenated-in-document-order")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_concatenate_video_cross_public_hosts_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        native_bytes = native.read_bytes()
        module = "video_components_concat_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from pathlib import Path\n"
            "import dinkster_nodes_comfy.video_components as native\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "from test_video_components import (\n"
            "    ComponentConsumer, ComponentSource, ConcatenateConsumer, ConcatenateSource,\n"
            ")\n"
            "assert Path(native.__file__).resolve() == "
            f"Path({str(package / 'src/dinkster_nodes_comfy/video_components.py')!r})\n"
            "NODES = (ComponentSource, ComponentConsumer, "
            "ConcatenateSource, ConcatenateConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "video-components-public-dinkster-pack.toml"
        fixture.write_text(
            "[pack]\n"
            'name = "video-components-public-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["video-components-public"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native, fixture)
        paths = (
            tmp_path,
            package / "tests",
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
        )
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_PACK_SCRATCH": str(scratch),
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            },
        )
        try:
            for manifest in manifests:
                pack = PackSpec(manifest=manifest, in_process=True)
                if not in_process:
                    pack = PackSpec(
                        manifest=manifest,
                        worker_group="video-components-public",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("video-components-public.concatenate-source"),
                    "concatenate_with_audio": GraphNode(
                        "ConcatenateVideo",
                        {
                            "videos.video_z": Link("source", "first"),
                            "videos.video_a": Link("source", "second"),
                            "complete_audio": Link("source", "replacement"),
                        },
                    ),
                    "concatenate_without_audio": GraphNode(
                        "ConcatenateVideo",
                        {
                            "videos.video_z": Link("source", "first"),
                            "videos.video_a": Link("source", "second"),
                        },
                    ),
                    "with_audio": GraphNode(
                        "video-components-public.concatenate-consumer",
                        {
                            "video": Link("concatenate_with_audio", "video"),
                            "expected": Link("source", "expected_with_audio"),
                        },
                    ),
                    "without_audio": GraphNode(
                        "video-components-public.concatenate-consumer",
                        {
                            "video": Link("concatenate_without_audio", "video"),
                            "expected": Link("source", "expected_without_audio"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["with_audio", "without_audio"])
                assert (
                    result.outputs["with_audio"]["summary"].resolve()
                    == "concatenated-in-document-order"
                )
                assert (
                    result.outputs["without_audio"]["summary"].resolve()
                    == "concatenated-in-document-order"
                )
                assert set(result.cached) == (set() if attempt == 0 else set(graph.nodes))
        finally:
            await composer.close()
        assert native.read_bytes() == native_bytes

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_native_video_components_cross_public_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        native_bytes = native.read_bytes()
        module = "video_components_public_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from pathlib import Path\n"
            "import dinkster_nodes_comfy.video_components as native\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "from test_video_components import ComponentConsumer, ComponentSource\n"
            "assert Path(native.__file__).resolve() == "
            f"Path({str(package / 'src/dinkster_nodes_comfy/video_components.py')!r})\n"
            "NODES = (ComponentSource, ComponentConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "video-components-public-dinkster-pack.toml"
        fixture.write_text(
            "[pack]\n"
            'name = "video-components-public-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["video-components-public"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native, fixture)
        paths = (
            tmp_path,
            package / "tests",
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
                        worker_group="video-components-public",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("video-components-public.source"),
                    "create": GraphNode(
                        "CreateVideo",
                        {
                            "images": Link("source", "images"),
                            "fps": Link("source", "fps"),
                            "audio": Link("source", "audio"),
                            "bit_depth": Link("source", "bit_depth"),
                            "color_space": Link("source", "color_space"),
                            "codec": "none",
                        },
                    ),
                    "get": GraphNode("GetVideoComponents", {"video": Link("create", "video")}),
                    "consumer": GraphNode(
                        "video-components-public.consumer",
                        {
                            "images": Link("get", "images"),
                            "audio": Link("get", "audio"),
                            "fps": Link("get", "fps"),
                            "bit_depth": Link("get", "bit_depth"),
                            "color_space": Link("get", "color_space"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "components-preserved"
                assert set(result.cached) == (set() if attempt == 0 else set(graph.nodes))
        finally:
            await composer.close()
        assert native.read_bytes() == native_bytes

    asyncio.run(scenario())
