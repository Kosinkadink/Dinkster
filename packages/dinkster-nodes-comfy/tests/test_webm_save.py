"""CPU behavior for the native SaveWEBM output node."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from dinkster_api.v1 import (
    IMAGE_TYPE,
    ExportSnapshot,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.webm_save import SaveWEBM

from dinkster.compose import PackSpec, ServingComposer


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


def _images(*, channels: int = 3, width: int = 24, height: int = 16) -> np.ndarray[Any, Any]:
    images = np.zeros((3, height, width, channels), dtype=np.float32)
    images[:, :, : width // 2, 0] = 1
    images[:, :, width // 2 :, 1] = 1
    if channels == 4:
        images[0, ..., 3] = 0
        images[1, ..., 3] = 0.5
        images[2, ..., 3] = 1
    return images


def test_save_webm_schema_is_discoverable_without_loading_av() -> None:
    from dinkster_nodes_comfy import webm_save

    assert Path(webm_save.__file__).resolve() == (
        Path(__file__).resolve().parents[1] / "src/dinkster_nodes_comfy/webm_save.py"
    )
    schema = SaveWEBM.define_schema()
    assert "av" not in vars(webm_save)
    assert schema.node_type == "SaveWEBM"
    assert tuple(input.id for input in schema.inputs) == (
        "images",
        "filename_prefix",
        "codec",
        "fps",
        "crf",
        "mount",
    )
    assert schema.inputs[1].default == "ComfyUI"
    assert tuple(schema.inputs[2].widget.options) == ("vp9", "av1")
    assert schema.inputs[3].default == 24.0
    assert schema.inputs[4].default == 32.0
    mount = schema.inputs[-1]
    assert mount.id == "mount"
    assert not mount.required and mount.default is None
    assert mount.adjacent_to == "filename_prefix"
    assert isinstance(mount.widget, MountWidget)
    assert schema.outputs[0].id == "images" and schema.outputs[0].preview
    assert schema.output_node and not schema.idempotent
    checkout = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.webm_save import SaveWEBM; "
            "assert SaveWEBM.define_schema().node_type == 'SaveWEBM'; "
            "assert not any(name in sys.modules for name in "
            "('av', 'torch', 'dinkster_inference'))",
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                str(path) for path in (checkout / "src", *sorted(checkout.glob("packages/*/src")))
            ),
        },
        check=True,
    )


@pytest.mark.parametrize("channels", (3, 4))
def test_save_webm_vp9_preserves_frames_metadata_source_and_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    from dinkster_nodes_comfy import webm_save

    default, alternate, _ = _mounts(tmp_path, monkeypatch)
    images = _images(channels=channels)
    original = images.copy()
    prompt = {"4": {"class_type": "SaveWEBM", "inputs": {"fps": 3.125}}}
    workflow = {"nodes": [{"id": 4, "type": "SaveWEBM"}], "opaque": [False, {"n": 2}]}
    monkeypatch.setattr(
        webm_save,
        "current_export_snapshot",
        lambda: ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow, "flag": False}),
    )

    result = SaveWEBM.execute(
        images=images,
        filename_prefix="web/%width%x%height%/clip",
        codec="vp9",
        fps=3.125,
        crf=24.0,
        mount="alternate",
    )

    assert result["images"] is images
    np.testing.assert_array_equal(images, original)
    assert not list(default.rglob("*"))
    paths = list(alternate.glob("web/24x16/clip_*.webm"))
    assert len(paths) == 1
    with av.open(paths[0]) as container:
        tags = {name.lower(): value for name, value in container.metadata.items()}
        assert {name: json.loads(tags[name]) for name in ("prompt", "workflow", "flag")} == {
            "prompt": prompt,
            "workflow": workflow,
            "flag": False,
        }
        stream = container.streams.video[0]
        assert stream.codec_context.name == "vp9"
        assert (stream.width, stream.height) == (24, 16)
        assert float(stream.average_rate) == 3.125
        if channels == 4:
            assert stream.metadata["alpha_mode"] == "1"
            decoder = av.CodecContext.create("libvpx-vp9", "r")
            decoder.thread_count = 1
            decoder.extradata = stream.codec_context.extradata
            decoded = [
                frame
                for packet in container.demux(stream)
                if packet.dts is not None
                for frame in decoder.decode(packet)
            ]
            decoded.extend(decoder.decode(None))
        else:
            decoded = list(container.decode(stream))
        frames = [frame.to_ndarray(format="rgba") for frame in decoded]
    assert len(frames) == 3
    assert frames[0][:, :10, 0].mean() > frames[0][:, 14:, 0].mean() + 100
    assert frames[0][:, 14:, 1].mean() > frames[0][:, :10, 1].mean() + 100
    if channels == 4:
        assert all(frame.format.name.startswith("yuva") for frame in decoded)
        alpha = [frame[..., 3].mean() for frame in frames]
        assert alpha[0] < 30
        assert 90 < alpha[1] < 165
        assert alpha[2] > 220


def test_save_webm_av1_writes_a_10_bit_video_at_supported_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, _, _ = _mounts(tmp_path, monkeypatch)
    images = _images(width=64, height=64)
    original = images.copy()

    assert (
        SaveWEBM.execute(
            images=images,
            filename_prefix="av1/clip",
            codec="av1",
            fps=12.5,
            crf=30.0,
        )["images"]
        is images
    )
    np.testing.assert_array_equal(images, original)
    path = next(default.glob("av1/clip_*.webm"))
    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.codec.id == av.Codec("av1", "r").id
        assert stream.codec_context.format.name == "yuv420p10le"
        assert (stream.width, stream.height) == (64, 64)
        assert float(stream.average_rate) == 12.5
        assert len(list(container.decode(stream))) == 3


@pytest.mark.parametrize("mount", (None, "alternate"))
def test_save_webm_uses_default_or_explicit_writable_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str | None
) -> None:
    default, alternate, _ = _mounts(tmp_path, monkeypatch)
    SaveWEBM.execute(
        images=_images(), filename_prefix="video/clip", codec="vp9", fps=24.0, crf=32.0, mount=mount
    )
    selected, untouched = (default, alternate) if mount is None else (alternate, default)
    assert len(list(selected.glob("video/clip_*.webm"))) == 1
    assert not list(untouched.rglob("*"))


@pytest.mark.parametrize("mount", ("outside", "readonly"))
def test_save_webm_refuses_unknown_or_readonly_mount_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str
) -> None:
    default, alternate, readonly = _mounts(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="mount"):
        SaveWEBM.execute(
            images=_images(),
            filename_prefix="video/clip",
            codec="vp9",
            fps=24.0,
            crf=32.0,
            mount=mount,
        )
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))
    assert not list(readonly.rglob("*"))


class ImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "webm-fixture.source", outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(image=_images(width=64, height=64))


@pytest.mark.parametrize("in_process", (True, False))
def test_save_webm_ordinary_manifest_runs_twice_with_a_cached_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    output, _, _ = _mounts(tmp_path, monkeypatch)

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        paths = (checkout / "src", *sorted(checkout.glob("packages/*/src")))
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
            manifest = package / "dinkster-pack.toml"
            (tmp_path / "webm_fixture.py").write_text(
                "from test_webm_save import ImageSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (ImageSource,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "fixture.toml"
            fixture.write_text(
                '[pack]\nname = "webm-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["webm-fixture"]\n[pack.entry]\n'
                'nodes = "webm_fixture:NODES"\ntypes = "webm_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for path in (manifest, fixture):
                await composer.add_pack(
                    PackSpec(manifest=path, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=path,
                        worker_group="webm-fixture",
                        group_manifests=(manifest, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("webm-fixture.source"),
                    "save": GraphNode(
                        "SaveWEBM",
                        {
                            "images": Link("source", "image"),
                            "filename_prefix": "graph/clip",
                            "codec": "av1",
                            "fps": 12.5,
                            "crf": 30.0,
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["save"])
            second = await engine.run(graph, ["save"])
            assert "source" not in first.cached and "save" not in first.cached
            assert "source" in second.cached and "save" not in second.cached
            for result in (first, second):
                pixels = result.outputs["save"]["images"].resolve()
                np.testing.assert_array_equal(pixels, _images(width=64, height=64))
            files = sorted(output.glob("graph/clip_*.webm"))
            assert len(files) == 2
            for path in files:
                with av.open(path) as container:
                    stream = container.streams.video[0]
                    assert (stream.width, stream.height) == (64, 64)
                    assert float(stream.average_rate) == 12.5
                    frames = [
                        frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
                    ]
                assert len(frames) == 3
                for frame in frames:
                    assert frame[:, :24, 0].mean() > frame[:, 40:, 0].mean() + 100
                    assert frame[:, 40:, 1].mean() > frame[:, :24, 1].mean() + 100
        finally:
            await composer.close()

    asyncio.run(scenario())
