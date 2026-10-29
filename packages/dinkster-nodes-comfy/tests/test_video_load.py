"""LoadVideo admits indexed video assets without copying their source bytes."""

from __future__ import annotations

import asyncio
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
    AssetError,
    AssetIntegrityError,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import LocalAssetRef, MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.video_load import LoadVideo
from dinkster_values import video_source

from dinkster.compose import PackSpec, ServingComposer


def _frames() -> np.ndarray:
    return np.array(
        [
            [
                [[3, 17, 29], [41, 53, 67], [71, 83, 97]],
                [[101, 113, 127], [131, 149, 157], [163, 173, 181]],
            ],
            [
                [[191, 199, 211], [223, 227, 229], [233, 239, 241]],
                [[251, 2, 5], [7, 11, 13], [19, 23, 31]],
            ],
        ],
        dtype=np.uint8,
    )


def _write_clip(path: Path) -> tuple[bytes, np.ndarray]:
    frames = _frames()
    with av.open(path, "w") as container:
        stream = container.add_stream("huffyuv", rate=Fraction(30_000, 1_001))
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "rgb24"
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes(), frames


def _asset(
    tmp_path: Path, *, indexed: bool
) -> tuple[MountTable, AssetRef | LocalAssetRef, bytes, np.ndarray]:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "asymmetric.avi"
    original, frames = _write_clip(path)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="video", path=source), kind="media/video")
    if indexed:
        table.scan("video")
    asset = table.bind_name(path.name, "media/video")
    assert isinstance(asset, AssetRef if indexed else LocalAssetRef)
    return table, asset, original, frames


def _decoded_pixels(video: object) -> np.ndarray:
    source = video_source(video)
    with source.open() as handle, av.open(handle) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def test_load_video_schema_is_the_canonical_single_file_surface_without_runtime_dependencies() -> (
    None
):
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from dinkster_api.v1 import ASSET_TYPE, VIDEO_TYPE, TypeExpr
import dinkster_nodes_comfy.video_load as module
from dinkster_nodes_comfy.video_load import LoadVideo

assert Path(module.__file__).resolve() == Path(sys.argv[1])
schema = LoadVideo.schema()
assert schema.node_type == "LoadVideo"
assert schema.display_name == "Load Video"
assert schema.category == "video"
assert schema.search_terms == ("import video", "open video", "video file")
assert tuple(item.id for item in schema.inputs) == ("file",)
assert schema.inputs[0].type == TypeExpr.concrete(ASSET_TYPE)
assert schema.inputs[0].widget.kind == "media/video"
assert schema.inputs[0].widget.accept == ("video/*",)
assert schema.inputs[0].widget.allow_upload is True
assert schema.inputs[0].source_filename.kind == "media/video"
assert schema.inputs[0].source_filename.category == "input"
assert [(item.id, item.type, item.preview) for item in schema.outputs] == [
    ("VIDEO", TypeExpr.concrete(VIDEO_TYPE), True)
]
assert "av" not in sys.modules
assert "torch" not in sys.modules
assert not any(
    name == "dinkster_inference" or name.startswith("dinkster_inference.")
    for name in sys.modules
)
""",
            str(root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/video_load.py"),
        ],
        check=True,
        cwd=root,
    )


def test_load_video_admits_granted_sources_without_copying_and_rejects_untrusted_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, local, original, expected = _asset(tmp_path, indexed=False)
    path = tmp_path / "source" / "asymmetric.avi"
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))

    with pytest.raises(AssetError, match="must be indexed"):
        LoadVideo.execute(file=local)
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadVideo.execute(file=path)
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadVideo.execute(file=str(path))

    table.scan("video")
    selected = table.bind_name(path.name, "media/video")
    assert isinstance(selected, AssetRef)
    loaded_from_local = LoadVideo.execute(file=local)["VIDEO"]
    loaded = LoadVideo.execute(file=selected)["VIDEO"]
    assert video_source(loaded_from_local).digest == selected.digest
    source = video_source(loaded)
    assert isinstance(source, AssetRef)
    assert source.digest == digest_file(path)
    assert source.local_path() == path
    assert source.read_bytes() == original
    np.testing.assert_array_equal(_decoded_pixels(loaded), expected)
    assert loaded["probe"]["fps"] == Fraction(30_000, 1_001)
    assert loaded["probe"]["width"] == 3
    assert loaded["probe"]["height"] == 2
    assert loaded["probe"]["frame_count"] == 2
    assert not (tmp_path / "vault").exists()
    assert not (tmp_path / "scratch").exists()
    assert path.read_bytes() == original

    path.write_bytes(original + b"changed")
    with pytest.raises(AssetIntegrityError):
        LoadVideo.execute(file=selected)


class VideoLoadConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-load-fixture.consumer",
            inputs=(InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, video: object) -> dict[str, object]:
        source = video_source(video)
        with source.open() as handle, av.open(handle) as container:
            pixels = np.stack(
                [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            )
        np.testing.assert_array_equal(pixels, _frames())
        probe = video["probe"]
        assert probe["fps"] == Fraction(30_000, 1_001)
        assert (probe["width"], probe["height"], probe["frame_count"]) == (3, 2, 2)
        return cls.outputs(summary=source.digest)


@pytest.mark.parametrize("in_process", (True, False))
def test_load_video_executes_through_native_and_fixture_packs_with_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    _table, asset, original, _frames_value = _asset(tmp_path, indexed=True)
    path = tmp_path / "source" / "asymmetric.avi"
    snapshot = tmp_path / "mounts.json"
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        fixture_module = tmp_path / "video_load_fixture.py"
        fixture_module.write_text(
            "from pathlib import Path\n"
            "from test_video_load import VideoLoadConsumer\n"
            "import dinkster_nodes_comfy.video_load as module\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "assert Path(module.__file__).resolve() == "
            f"Path({str(package / 'src/dinkster_nodes_comfy/video_load.py')!r})\n"
            "NODES = (VideoLoadConsumer,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="video-load-fixture"\npublisher="dinkster"\n'
            'namespaces=["video-load-fixture"]\n[pack.entry]\n'
            'nodes="video_load_fixture:NODES"\n'
            'types="video_load_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (
            tmp_path,
            package / "tests",
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        )
        native = package / "dinkster-pack.toml"
        try:
            for manifest in (native, fixture):
                pack = PackSpec(manifest=manifest, in_process=True)
                if not in_process:
                    pack = PackSpec(
                        manifest=manifest,
                        worker_group="video-load-fixture",
                        group_manifests=(native, fixture),
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "load": GraphNode("LoadVideo", {"file": asset.to_wire()}),
                    "consume": GraphNode(
                        "video-load-fixture.consumer", {"video": Link("load", "VIDEO")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consume"])
                assert result.outputs["consume"]["summary"].resolve() == digest_file(path)
                assert set(result.cached) == (set() if attempt == 0 else {"load", "consume"})
                assert path.read_bytes() == original
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert path.read_bytes() == original
