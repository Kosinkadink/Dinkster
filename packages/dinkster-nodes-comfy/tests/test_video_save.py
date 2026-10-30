"""CPU behavior for the native SaveVideo output node."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    CORE_COMBO,
    VIDEO_TYPE,
    ComboWidget,
    ExportSnapshot,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_assets.identity import AssetError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.video_components import CreateVideo
from dinkster_nodes_comfy.video_save import SaveVideo
from dinkster_schema import elaborate

from dinkster.compose import PackSpec, ServingComposer

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


def _mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    default, alternate, readonly = (
        tmp_path / name for name in ("default", "alternate", "readonly")
    )
    for root in (default, alternate, readonly):
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "default",
                "mounts": [
                    {"id": "default", "root": str(default), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                    {"id": "readonly", "root": str(readonly), "mode": "read"},
                ],
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return default, alternate, readonly


def _images() -> np.ndarray[Any, Any]:
    images = np.zeros((3, 16, 24, 3), dtype=np.float32)
    images[:, :, :12, 0] = 1
    images[:, :, 12:, 1] = 1
    return images


def _video() -> object:
    return CreateVideo.execute(images=_images(), fps=4.0)["video"]


def _tags(path: Path) -> tuple[dict[str, object], Any, int]:
    with av.open(path) as container:
        tags = {
            name: json.loads(container.metadata[name])
            for name in ("prompt", "workflow")
            if name in container.metadata
        }
        stream = container.streams.video[0]
        return tags, stream, len(list(container.decode(stream)))


def test_save_video_schema_exposes_native_format_controls() -> None:
    schema = SaveVideo.define_schema()
    inputs = {input.id: input for input in schema.inputs}
    (format_combo,) = schema.combos
    assert schema.node_type == "SaveVideo"
    assert inputs["filename_prefix"].default == "video/ComfyUI"
    assert [option.key for option in format_combo.options] == ["auto", "mp4", "mkv", "webm"]
    top_codec = inputs["codec"]
    assert top_codec.type == TypeExpr.concrete(CORE_COMBO)
    assert not top_codec.required and top_codec.hidden and top_codec.default == "auto"
    assert isinstance(top_codec.widget, ComboWidget)
    assert top_codec.widget.options == ("auto", "h264", "av1")
    nested_codecs = [option.inputs[0] for option in format_combo.options]
    assert all(codec.id == "codec" and codec.required for codec in nested_codecs)
    assert all(codec.default is None for codec in nested_codecs)
    wire = schema_to_wire(schema)
    codec_wire = next(entry for entry in wire["interface"] if entry["id"] == "codec")
    assert codec_wire["role"] == "input"
    assert codec_wire["hidden"] is True
    assert codec_wire["default"] == "auto"
    assert schema_from_wire(wire) == schema
    effective = elaborate(
        schema,
        {"video": object(), "filename_prefix": "video/ComfyUI", "format.codec.encoding.crf": 18.0},
        slot_variants={
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "re-encode",
        },
    )
    assert [input.id for input in effective.inputs] == [
        "video",
        "filename_prefix",
        "format.codec.encoding.crf",
        "codec",
        "mount",
    ]
    mount = inputs["mount"]
    assert not mount.required and mount.default is None and mount.adjacent_to == "filename_prefix"
    assert isinstance(mount.widget, MountWidget)
    assert schema.outputs[0].id == "video" and schema.outputs[0].preview
    assert schema.output_node and not schema.idempotent


def test_save_video_writes_selected_mount_metadata_and_original_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_nodes_comfy import video_save

    default, alternate, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    prompt = {"4": {"class_type": "SaveVideo", "inputs": {"format": "mp4"}}}
    workflow = {"nodes": [{"id": 4, "type": "SaveVideo"}]}
    monkeypatch.setattr(
        video_save,
        "current_export_snapshot",
        lambda: ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow}),
    )

    assert (
        SaveVideo.execute(
            video=video,
            filename_prefix="video/%width%x%height%/clip",
            format={
                "format": "mp4",
                "codec": {"codec": "h264", "encoding": {"crf": 18.0}},
            },
            mount="alternate",
        )["video"]
        is video
    )
    assert not list(default.rglob("*"))
    files = list(alternate.glob("video/24x16/clip_*.mp4"))
    assert len(files) == 1
    tags, stream, frame_count = _tags(files[0])
    assert {name: tags[name] for name in ("prompt", "workflow")} == {
        "prompt": prompt,
        "workflow": workflow,
    }
    assert stream.codec_context.name == "h264"
    assert (stream.width, stream.height) == (24, 16)
    assert float(stream.average_rate) == 4.0
    assert frame_count == 3


@pytest.mark.parametrize(
    ("format", "codec", "crf"),
    (
        ("auto", "h264", 18),
        ("mkv", "h264", 18),
    ),
)
def test_save_video_accepts_flat_dynamic_native_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    codec: str,
    crf: int,
) -> None:
    default, _, _ = _mounts(tmp_path, monkeypatch)
    video = _video()
    extension = "mp4" if format == "auto" else format
    result = SaveVideo.execute(
        video=video,
        filename_prefix=f"flat/{extension}",
        format=format,
        **{
            "format.codec": codec,
            "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": crf,
        },
    )
    assert result["video"] is video
    path = next(default.glob(f"flat/{extension}_*.{extension}"))
    _, stream, frame_count = _tags(path)
    assert stream.codec_context.name == "h264"
    assert (stream.width, stream.height) == (24, 16)
    assert frame_count == 3


@pytest.mark.parametrize(
    ("format", "codec", "prefix"),
    (
        ("gif", "auto", "video/clip"),
        ("webm", "h264", "video/clip"),
        ("mp4", "vp9", "video/clip"),
        ("mp4", "auto", "../escape"),
    ),
)
def test_save_video_refuses_unsafe_values_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str, codec: str, prefix: str
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises((ValueError, AssetError)):
        SaveVideo.execute(
            video=_video(), filename_prefix=prefix, format=format, **{"format.codec": codec}
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))


def test_save_video_refuses_readonly_mount_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises(AssetError, match="mount"):
        SaveVideo.execute(
            video=_video(), filename_prefix="video/clip", format="mp4", mount="readonly"
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))


class VideoSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "video-save-fixture.source",
            outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(video=_video())


@pytest.mark.parametrize("in_process", (True, False))
def test_save_video_ordinary_manifest_runs_twice_in_both_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
) -> None:
    output, _, _ = _mounts(tmp_path, monkeypatch)

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        paths = (checkout / "src", *sorted(checkout.glob("packages/*/src")))
        module = f"video_save_fixture_{int(in_process)}"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": os.environ["DINKSTER_MOUNTS_SNAPSHOT"],
                "PYTHONPATH": os.pathsep.join(
                    (*map(str, paths), str(Path(__file__).parent), *sys.path)
                ),
            }
        )
        try:
            (tmp_path / f"{module}.py").write_text(
                "from test_video_save import VideoSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (VideoSource,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "fixture.toml"
            fixture.write_text(
                '[pack]\nname = "video-save-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["video-save-fixture"]\n[pack.entry]\n'
                f'nodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            native = package / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="video-save-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("video-save-fixture.source"),
                    "save": GraphNode(
                        "SaveVideo",
                        {
                            "video": Link("source", "video"),
                            "filename_prefix": "graph/mp4",
                            "format.codec.encoding.crf": 18.0,
                        },
                        slot_variants={
                            "format": "mp4",
                            "format.codec": "h264",
                            "format.codec.encoding": "re-encode",
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["save"])
                saved = result.outputs["save"]["video"]
                if in_process:
                    assert saved.resolve() is not None
                else:
                    assert saved.payload.type_id == "VIDEO"
                assert ("source" in result.cached) is bool(attempt)
                assert "save" not in result.cached
            files = sorted(output.glob("graph/mp4_*.mp4"))
            assert len(files) == 2
            for path in files:
                _, stream, frame_count = _tags(path)
                assert stream.codec_context.name == "h264"
                assert (stream.width, stream.height) == (24, 16)
                assert float(stream.average_rate) == 4.0
                assert frame_count == 3
        finally:
            await composer.close()

    asyncio.run(scenario())
