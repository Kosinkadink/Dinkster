"""Native asset-backed 3D model I/O contracts."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import dinkster_nodes_media_io.model3d as model3d_module
import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    FILE_3D_GLB_TYPE,
    FILE_3D_GLTF_TYPE,
    FILE_3D_PLY_TYPE,
    FILE_3D_TYPE,
    LOAD_3D_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    LocalAssetRef,
    MountDef,
    MountSnapshotResolver,
    MountTable,
    digest_bytes,
    load_file3d,
    register_asset_type,
    register_file3d_types,
    register_load3d_type,
    resolver_from_env,
)
from dinkster_assets.declared import produced_output_scratch
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_nodes_comfy import register_inference_types
from dinkster_nodes_comfy.preview3d import (
    Preview3DAdvanced,
    PreviewGaussianSplat,
    PreviewPointCloud,
)
from dinkster_nodes_comfy.save3d import Save3DAdvanced, SaveGaussianSplat, SavePointCloud
from dinkster_nodes_media_io import LoadModel3D, PreviewModel3D, SaveModel3D, register_media_types
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, decode_model3d, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.compose import PackSpec, ServingComposer

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


def test_load_file3d_binds_only_declared_adjacent_files_and_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / "data").mkdir()
    primary = (
        b'{"asset":{"version":"2.0"},"buffers":[{"uri":"data/mesh.bin"}],'
        b'"images":[{"uri":"albedo.png"},{"uri":"data:image/png;base64,AA=="}]}'
    )
    (root / "scene.gltf").write_bytes(primary)
    (root / "data/mesh.bin").write_bytes(b"\x01\x03\x09")
    (root / "albedo.png").write_bytes(b"different texture bytes")
    (root / "unused.bin").write_bytes(b"not declared")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef(id="models", path=root), kind="media/model3d")
    table.scan("models")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    source = table.bind_name("scene.gltf", "media/model3d")
    assert isinstance(source, AssetRef)
    packet = load_file3d(source)
    assert packet is not None and packet["source"] is source and packet["format"] == "gltf"
    refs = cast("dict[str, AssetRef]", packet["resources"])
    assert {name: ref.read_bytes() for name, ref in refs.items()} == {
        "data/mesh.bin": b"\x01\x03\x09",
        "albedo.png": b"different texture bytes",
    }
    registry = TypeRegistry()
    register_file3d_types(registry, MountSnapshotResolver(snapshot))
    value = registry.wrap(FILE_3D_GLTF_TYPE, packet)
    manifest = json.loads(registry.render(value, "manifest").data)
    assert set(manifest["resources"]) == {"data/mesh.bin", "albedo.png"}
    assert source.read_bytes() == primary
    payload = registry.spec(FILE_3D_GLTF_TYPE).encode(packet)
    table.remove("models")
    restored = cast("dict[str, object]", registry.spec(FILE_3D_GLTF_TYPE).decode(payload))
    with pytest.raises(AssetError):
        cast(AssetRef, restored["source"]).read_bytes()
    with pytest.raises(AssetError):
        cast("dict[str, AssetRef]", restored["resources"])["data/mesh.bin"].read_bytes()
    with pytest.raises(AssetError):
        load_file3d(source)


def test_load_file3d_adjacent_bindings_require_indexed_contained_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / "scene.obj").write_bytes(b"mtllib surface.mtl\nv 1 3 7\n")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef(id="models", path=root), kind="media/model3d")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    source = table.bind_name("scene.obj", "media/model3d")
    assert isinstance(source, LocalAssetRef)
    with pytest.raises(AssetError, match="requires an explicit"):
        load_file3d(source)
    table.scan("models")
    (root / "surface.mtl").write_bytes(b"newmtl material\nKd 0.1 0.3 0.7\n")
    with pytest.raises(AssetError, match="must be indexed"):
        load_file3d(source)
    table.scan("models")
    packet = load_file3d(source)
    assert packet is not None
    material = cast("dict[str, AssetRef]", packet["resources"])["surface.mtl"]
    assert material.read_bytes() == b"newmtl material\nKd 0.1 0.3 0.7\n"
    if os.name != "nt":
        outside = tmp_path / "outside.mtl"
        outside.write_bytes(b"not granted")
        (root / "surface.mtl").unlink()
        (root / "surface.mtl").symlink_to(outside)
        with pytest.raises(AssetError, match="outside its grant"):
            load_file3d(source)


def test_load_file3d_explicit_obj_dependencies_and_missing_extra_controls(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "assets")

    def publish(data: bytes, name: str) -> AssetRef:
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
        return AssetRef(digest_bytes(data), name, len(data), resolver=vault)

    source = publish(b"mtllib materials/model.mtl\nv 2 3 5\n", "source.obj")
    material = publish(b"newmtl surface\nmap_Kd -s 1 2 3 textures/color.png\n", "model.mtl")
    texture = publish(b"texture contents", "color.png")
    resources = {"materials/model.mtl": material, "materials/textures/color.png": texture}
    assert load_file3d(source, resources=resources) == {
        "format": "obj",
        "source": source,
        "resources": resources,
    }
    for incomplete in ({}, {"materials/model.mtl": material}):
        with pytest.raises(AssetError, match="requires an explicit"):
            load_file3d(source, resources=incomplete)
    with pytest.raises(AssetError, match="undeclared"):
        load_file3d(source, resources={**resources, "unused.png": texture})
    assert source.read_bytes() == b"mtllib materials/model.mtl\nv 2 3 5\n"


@pytest.mark.parametrize("format_id", ["gltf", "glb"])
def test_load_file3d_explicit_gltf_resources_decode_uri_names(
    tmp_path: Path, format_id: str
) -> None:
    vault = AssetVault(tmp_path / "assets")

    def publish(data: bytes, name: str) -> AssetRef:
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
        return AssetRef(digest_bytes(data), name, len(data), resolver=vault)

    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "data/mesh%20data.bin"}],
        "images": [{"uri": "textures/albedo%20map.png"}],
    }
    primary = json.dumps(document).encode()
    if format_id == "glb":
        primary += b" " * (-len(primary) % 4)
        primary = (
            struct.pack("<4sIIII", b"glTF", 2, 20 + len(primary), len(primary), 0x4E4F534A)
            + primary
        )
    source = publish(primary, "scene." + format_id)
    mesh = publish(b"asymmetric mesh bytes", "mesh data.bin")
    texture = publish(b"different texture bytes", "albedo map.png")
    resources = {"data/mesh data.bin": mesh, "textures/albedo map.png": texture}
    packet = load_file3d(source, resources=resources)
    assert packet == {"format": format_id, "source": source, "resources": resources}
    registry = TypeRegistry()
    register_file3d_types(registry, vault)
    codec = registry.spec(FILE_3D_TYPE)
    restored = cast("dict[str, object]", codec.decode(codec.encode(packet)))
    assert cast(AssetRef, restored["source"]).read_bytes() == primary
    assert {
        name: ref.read_bytes()
        for name, ref in cast("dict[str, AssetRef]", restored["resources"]).items()
    } == {
        "data/mesh data.bin": b"asymmetric mesh bytes",
        "textures/albedo map.png": b"different texture bytes",
    }
    for missing in ({}, {"data/mesh data.bin": mesh}):
        with pytest.raises(AssetError, match="requires an explicit"):
            load_file3d(source, resources=missing)
    with pytest.raises(AssetError, match="undeclared"):
        load_file3d(source, resources={**resources, "unused.bin": mesh})


@pytest.mark.parametrize(
    "uri",
    [
        "../secret.bin",
        "/secret.bin",
        "https://host/file.bin",
        "%2e%2e/file.bin",
        "%252e%252e/file.bin",
        "data/%2e%2e/secret.bin",
        "%2fsecret.bin",
        "data%5csecret.bin",
        "data.bin?remote=1",
        "data.bin#fragment",
        "data/%00secret.bin",
        "\tdata.bin",
    ],
)
def test_load_file3d_cannot_follow_resource_paths(tmp_path: Path, uri: str) -> None:
    vault = AssetVault(tmp_path / "assets")
    data = json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": uri}]}).encode()
    with vault.writer(digest_bytes(data)) as writer:
        writer.write(data)
        writer.commit()
    source = AssetRef(digest_bytes(data), "scene.gltf", len(data), resolver=vault)
    with pytest.raises(AssetError, match="contained relative"):
        load_file3d(source, resources={})


def test_load_file3d_empty_and_self_contained_binary_sources(tmp_path: Path) -> None:
    for source in (None, "", "none"):
        assert load_file3d(source) is None
    with pytest.raises(AssetError, match="bound asset"):
        load_file3d("host-file.obj")
    data = b'{"asset":{"version":"2.0"}}  '
    payload = struct.pack("<4sIIII", b"glTF", 2, 20 + len(data), len(data), 0x4E4F534A) + data
    vault = AssetVault(tmp_path / "assets")
    with vault.writer(digest_bytes(payload)) as writer:
        writer.write(payload)
        writer.commit()
    source = AssetRef(digest_bytes(payload), "source.GLB", len(payload), resolver=vault)
    assert load_file3d(source) == {"format": "glb", "source": source, "resources": {}}
    assert source.read_bytes() == payload


@pytest.mark.parametrize("node", [Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud])
def test_file3d_preview_metadata_precedence_and_original_identity(
    node: type[Preview3DAdvanced | PreviewGaussianSplat | PreviewPointCloud],
) -> None:
    source = {"format": "gltf", "source": object(), "resources": {"mesh.bin": object()}}
    camera = {"zoom": 2.5}
    placement = [{"position": {"x": 3.0}}]
    state = {"camera_info": camera, "model_3d_info": placement}
    for viewport, explicit_camera, explicit_placement, expected_camera, expected_placement in (
        (state, None, None, camera, placement),
        (state, {}, [], {}, []),
        ({"model_3d_info": None}, None, None, None, None),
        (None, None, None, None, []),
    ):
        result = node.execute(
            model_3d=source,
            viewport_state=viewport,
            width=371,
            height=209,
            camera_info=explicit_camera,
            model_3d_info=explicit_placement,
        )
        assert result["model_3d"] is source
        assert result["camera_info"] == expected_camera
        assert result["model_3d_info"] == expected_placement
        assert (result["width"], result["height"]) == (371, 209)


class Preview3DFileSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "preview-3d-source.make",
            outputs=tuple(
                OutputSpec(node.__name__, TypeExpr.concrete(node.output_type))
                for node in (Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud)
            )
            + (OutputSpec("viewport", TypeExpr.concrete(LOAD_3D_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        vault = produced_output_scratch()
        assert vault is not None
        refs = []
        for name, data in (
            ("original.ply", b"original point bytes"),
            ("sidecar.bin", b"sidecar bytes"),
        ):
            digest = digest_bytes(data)
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            refs.append(
                AssetRef(digest, name, len(data), "application/octet-stream", resolver=vault)
            )
        value = {"format": "ply", "source": refs[0], "resources": {"sidecar.bin": refs[1]}}
        return cls.outputs(
            **{
                node.__name__: value
                for node in (Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud)
            },
            viewport={
                "camera_info": {
                    "position": {"x": 1.0, "y": 2.0, "z": -3.0},
                    "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "zoom": 2.5,
                    "cameraType": "perspective",
                },
                "model_3d_info": [],
            },
        )


@pytest.mark.parametrize("in_process", [True, False])
def test_file3d_previews_keep_original_assets_through_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    module = f"preview_3d_source_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from test_model3d_io import Preview3DFileSource\n"
        "from dinkster_nodes_comfy import register_inference_types\nNODES=(Preview3DFileSource,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="preview-3d-source"\npublisher="dinkster"\n'
        'namespaces=["preview-3d-source"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_inference_types(registry)
        composer = ServingComposer(
            registry=registry,
            worker_env={
                "DINKSTER_ASSET_VAULT": str(tmp_path / "vault"),
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
            },
        )
        try:
            for path in (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml",
                manifest,
            ):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            nodes = {"source": GraphNode("preview-3d-source.make")}
            for node in (Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud):
                nodes[node.__name__] = GraphNode(
                    node.__name__,
                    {
                        "model_3d": Link("source", node.__name__),
                        "viewport_state": Link("source", "viewport"),
                        "width": 371,
                        "height": 209,
                    },
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(Graph(nodes), list(nodes)[1:])
                assert not result.artifacts
                for name in list(nodes)[1:]:
                    out = result.outputs[name]
                    value = cast("dict[str, object]", out["model_3d"].resolve())
                    assert value["format"] == "ply"
                    with cast("AssetRef", value["source"]).open() as stream:
                        assert stream.read() == b"original point bytes"
                    resources = cast("dict[str, AssetRef]", value["resources"])
                    with resources["sidecar.bin"].open() as stream:
                        assert stream.read() == b"sidecar bytes"
                    assert out["camera_info"].resolve() == {
                        "position": {"x": 1.0, "y": 2.0, "z": -3.0},
                        "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                        "zoom": 2.5,
                        "cameraType": "perspective",
                    }
                    assert out["model_3d_info"].resolve() == []
                    assert (out["width"].resolve(), out["height"].resolve()) == (371, 209)
        finally:
            await composer.close()

    asyncio.run(scenario())


def glb_bytes(document: bytes = b'{"asset":{"version":"2.0"}}') -> bytes:
    padded = document + b" " * (-len(document) % 4)
    chunk = struct.pack("<I4s", len(padded), b"JSON") + padded
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "out"
    root.mkdir()
    index = root / ".dinkster-asset-index.json"
    index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(root),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root, snapshot


def _bound(ref: AssetRef, snapshot: Path) -> AssetRef:
    return AssetRef(
        ref.digest,
        ref.name,
        ref.size,
        ref.media_type,
        ref.virtual_path,
        MountSnapshotResolver(snapshot),
    )


@pytest.mark.parametrize("node", [Save3DAdvanced, SaveGaussianSplat, SavePointCloud])
@pytest.mark.parametrize("override", [False, True])
def test_advanced_3d_saves_preserve_file_and_viewport_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node: type[Save3DAdvanced | SaveGaussianSplat | SavePointCloud],
    override: bool,
) -> None:
    from dinkster_schema.reporting import capture_saved_artifacts

    output, _ = _mount(tmp_path, monkeypatch)
    payload = b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n1 2 -7\n"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    source = AssetRef(digest, "original.ply", len(payload), "application/ply", resolver=vault)
    model = {"format": "ply", "source": source}
    camera = {
        "position": {"x": 1.0, "y": 2.0, "z": -3.0},
        "target": {"x": 0.0, "y": 1.0, "z": 0.0},
        "zoom": 2.5,
        "cameraType": "perspective",
    }
    placement = [
        {
            "position": {"x": 3.0, "y": -2.0, "z": 1.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "scale": {"x": 1.0, "y": 2.0, "z": 0.5},
        }
    ]
    viewport = {"camera_info": camera, "model_3d_info": placement}
    unchanged_viewport = json.loads(json.dumps(viewport))
    explicit_camera = {**camera, "zoom": 0.75}
    with capture_saved_artifacts() as claims:
        result = node.execute(
            model_3d=model,
            viewport_state=viewport,
            filename_prefix="scene/model",
            mount="comfy-output",
            width=789,
            height=321,
            camera_info=explicit_camera if override else None,
            model_3d_info=[] if override else None,
        )
    assert result == {
        "model_3d": model,
        "camera_info": explicit_camera if override else viewport["camera_info"],
        "model_3d_info": [] if override else viewport["model_3d_info"],
        "width": 789,
        "height": 321,
    }
    assert result["model_3d"] is model
    assert (output / "scene/model_00001.ply").read_bytes() == payload
    assert len(claims) == 1 and claims[0]["digest"] == digest
    assert viewport == unchanged_viewport


def test_save_then_load_round_trips_glb_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    data = glb_bytes()
    saved = SaveModel3D.execute(model=decode_model3d(data))
    ref = cast(AssetRef, saved["model"])
    assert ref.media_type == "model/gltf-binary"
    assert ref.name.endswith(".glb")
    assert ref.size == len(data)
    written = root / "3d" / ref.name
    assert written.read_bytes() == data
    assert ref.digest == digest_bytes(data)
    loaded = LoadModel3D.execute(model=_bound(ref, snapshot))
    assert loaded["model"] == decode_model3d(data)


def test_save_honors_explicit_target_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    saved = SaveModel3D.execute(
        model=decode_model3d(glb_bytes()),
        target={"mount": "comfy-output", "prefix": "meshes/scene"},
    )
    ref = cast(AssetRef, saved["model"])
    assert (root / "meshes" / ref.name).is_file()


def test_save_refuses_non_glb_values() -> None:
    with pytest.raises(ValueError, match="not a GLB"):
        SaveModel3D.execute(model={"format": "glb", "bytes": b"not a model"})


def test_save_requires_configured_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(AssetError, match="DINKSTER_MOUNTS_SNAPSHOT"):
        SaveModel3D.execute(model=decode_model3d(glb_bytes()))


def test_save_bounds_encoded_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mount(tmp_path, monkeypatch)
    monkeypatch.setattr(model3d_module, "MAX_ENCODED_MODEL3D_BYTES", 16)
    with pytest.raises(ValueError, match="output limit"):
        SaveModel3D.execute(model=decode_model3d(glb_bytes()))


def test_load_refuses_non_glb_assets(tmp_path: Path) -> None:
    payload = b"not a 3D model"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.glb", len(payload), "model/gltf-binary", resolver=vault)
    with pytest.raises(ValueError, match="cannot decode 'bad.glb' as a 3D model"):
        LoadModel3D.execute(model=ref)


def test_preview_validates_and_passes_the_value_through() -> None:
    value = decode_model3d(glb_bytes())
    result = PreviewModel3D.execute(model=value)
    assert result["model"] is value
    with pytest.raises(ValueError, match="not a GLB"):
        PreviewModel3D.execute(model={"format": "glb", "bytes": b"junk"})


def test_worker_hands_load_the_asset_and_decodes_it_for_preview(tmp_path: Path) -> None:
    """Both worker input routes: a declared asset<MODEL3D> input receives
    the AssetRef itself (no coercion), while a concrete MODEL3D input fed
    the same asset literal decodes through the registered asset decoder."""

    class DirResolver:
        def __init__(self, files: dict[str, Path]) -> None:
            self._files = files

        def resolve(self, digest: str) -> Path | None:
            return self._files.get(digest)

    async def scenario() -> None:
        data = glb_bytes()
        digest = digest_bytes(data)
        path = tmp_path / "model.glb"
        path.write_bytes(data)
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, DirResolver({digest: path}))
        register_media_types(registry)
        nodes = (LoadModel3D, PreviewModel3D)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        literal = TypedLiteral(
            "asset<MODEL3D>", {"digest": digest, "name": "model.glb", "size": len(data)}
        )
        graph = Graph(
            nodes={
                "load": GraphNode("dinkster.load_model3d", {"model": literal}),
                "preview": GraphNode("dinkster.preview_model3d", {"model": literal}),
            }
        )
        result = await engine.run(graph, ["load", "preview"])
        expected = decode_model3d(data)
        assert result.outputs["load"]["model"].resolve() == expected
        assert result.outputs["preview"]["model"].resolve() == expected

    asyncio.run(scenario())


class Advanced3DFileSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "advanced-3d-source.make",
            inputs=(InputSpec("case", TypeExpr.concrete(CORE_STRING), default="viewport"),),
            outputs=(
                OutputSpec("file", TypeExpr.concrete(FILE_3D_PLY_TYPE)),
                OutputSpec("viewport", TypeExpr.concrete(LOAD_3D_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, case: str = "viewport") -> Mapping[str, object]:
        vault = produced_output_scratch()
        assert vault is not None
        payload = b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n1 2 -7\n"
        digest = digest_bytes(payload)
        with vault.writer(digest) as writer:
            writer.write(payload)
            writer.commit()
        ref = AssetRef(digest, "source.ply", len(payload), "application/ply", resolver=vault)
        if case != "viewport":
            return cls.outputs(
                file={"format": "ply", "source": ref},
                viewport={"model_3d_info": None} if case == "null_placement" else {},
            )
        return cls.outputs(
            file={"format": "ply", "source": ref},
            viewport={
                "camera_info": {
                    "position": {"x": 1.0, "y": 2.0, "z": -3.0},
                    "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "zoom": 2.5,
                    "cameraType": "perspective",
                },
                "model_3d_info": [],
            },
        )


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("case", ["viewport", "empty", "null_placement"])
def test_advanced_3d_save_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, case: str
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    table.add(MountDef("output", output, mode="readwrite"))
    table.scan("output")
    table.select_output_mount("output")
    vault = tmp_path / "vault"
    environment = {
        "DINKSTER_MOUNTS_SNAPSHOT": str(mounts),
        "DINKSTER_ASSET_VAULT": str(vault),
        "DINKSTER_ACCELERATOR": "cpu",
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    module = f"advanced_source_{case}"
    (tmp_path / f"{module}.py").write_text(
        "from test_model3d_io import Advanced3DFileSource\n"
        "from dinkster_nodes_comfy import register_inference_types\n"
        "NODES=(Advanced3DFileSource,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="advanced-3d-source"\npublisher="dinkster"\n'
        'namespaces=["advanced-3d-source"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=environment)
        try:
            for path in (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml",
                manifest,
            ):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            graph = Graph(
                {
                    "source": GraphNode("advanced-3d-source.make", {"case": case}),
                    "save": GraphNode(
                        "Save3DAdvanced",
                        {
                            "model_3d": Link("source", "file"),
                            "viewport_state": Link("source", "viewport"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in (1, 2):
                result = await engine.run(graph, ["save"])
                assert "save" in result.executed and len(result.artifacts) == 1
                file = output / f"3d/ComfyUI_{attempt:05d}.ply"
                assert file.read_bytes() == (
                    b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n1 2 -7\n"
                )
                camera = result.outputs["save"]["camera_info"].resolve()
                if case == "viewport":
                    assert cast(dict[str, object], camera)["zoom"] == 2.5
                else:
                    assert camera is None
                assert result.outputs["save"]["model_3d_info"].resolve() == (
                    None if case == "null_placement" else []
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_file3d_null_roundtrip_preserves_type_and_declares_no_assets() -> None:
    registry = TypeRegistry()
    for spec in register_file3d_types(registry):
        value = registry.wrap(spec.type_id, None)
        assert value.type_id == spec.type_id
        assert value.resolve() is None
        assert value.meta.entries == {"asset_refs": []}
        assert spec.encode(None) == b"null"
        assert spec.decode(b"null") is None
        assert spec.validate_encoded is not None
        spec.validate_encoded(b"null", value.meta.entries)
        with pytest.raises(AssetError, match="metadata"):
            spec.validate_encoded(b"null", {"asset_refs": [], "format": "glb"})
        restored = registry.wrap(spec.type_id, spec.decode(spec.encode(None)))
        assert restored.fingerprint == value.fingerprint
        rendition = registry.render(value, "manifest")
        assert rendition.mime == "application/json"
        assert rendition.data == b"null"
        for malformed in (False, 0, "none", {"format": "glb", "source": None}):
            with pytest.raises(AssetError):
                registry.wrap(spec.type_id, malformed)


def test_file3d_sources_and_sidecars_keep_original_bytes_and_receiver_authority(
    tmp_path: Path,
) -> None:
    source = b'{"asset":{"version":"2.0"},"buffers":[{"uri":"mesh/data.bin"}]}'
    resource = b"\x00\x01\xfe\xff"
    vault = AssetVault(tmp_path / "file3d")

    def publish(data: bytes, name: str) -> AssetRef:
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        return AssetRef(digest, name, len(data), resolver=vault)

    model = publish(source, "source.gltf")
    buffer = publish(resource, "data.bin")
    sender = TypeRegistry()
    receiver = TypeRegistry()
    assert len(register_file3d_types(sender, vault)) == 13
    register_file3d_types(receiver, vault)
    obj = {"format": "gltf", "source": model, "resources": {"mesh/data.bin": buffer}}
    value = sender.wrap(FILE_3D_GLTF_TYPE, obj)
    payload = sender.spec(FILE_3D_GLTF_TYPE).encode(obj)
    assert len(payload) < 1000 and source not in payload and resource not in payload
    spec = receiver.spec(FILE_3D_GLTF_TYPE)
    assert spec.validate_encoded is not None
    spec.validate_encoded(payload, value.meta.entries)
    restored = cast("dict[str, object]", spec.decode(payload))
    assert cast(AssetRef, restored["source"]).read_bytes() == source
    assert (
        cast("dict[str, AssetRef]", restored["resources"])["mesh/data.bin"].read_bytes() == resource
    )
    assert receiver.wrap(FILE_3D_GLTF_TYPE, restored).fingerprint == value.fingerprint
    manifest = receiver.render(receiver.wrap(FILE_3D_GLTF_TYPE, restored), "manifest")
    assert manifest.mime == "application/json"
    assert json.loads(manifest.data) == {
        "format": "gltf",
        "source": model.to_wire(),
        "resources": {"mesh/data.bin": buffer.to_wire()},
    }
    altered = {**obj, "resources": {"mesh/other.bin": buffer}}
    assert sender.wrap(FILE_3D_GLTF_TYPE, altered).fingerprint != value.fingerprint
    wrong_format = sender.spec(FILE_3D_GLB_TYPE)
    with pytest.raises(AssetError, match="does not accept"):
        sender.wrap(FILE_3D_GLB_TYPE, obj)
    with pytest.raises(AssetError, match="does not accept"):
        wrong_format.encode(obj)
    assert wrong_format.fingerprint is not None
    with pytest.raises(AssetError, match="does not accept"):
        wrong_format.fingerprint(obj)
    with pytest.raises(AssetError, match="does not accept"):
        wrong_format.decode(payload)
    with pytest.raises(AssetError, match="metadata"):
        spec.validate_encoded(payload, {**value.meta.entries, "asset_refs": []})


@pytest.mark.parametrize(
    "name",
    (
        "../mesh.bin",
        "/mesh.bin",
        "a/../b",
        "a//b",
        ".",
        "C:/b",
        "a\\b",
        "%2e%2e/b",
        "https://host/b",
    ),
)
def test_file3d_resources_cannot_grant_paths(name: str) -> None:
    registry = TypeRegistry()
    register_file3d_types(registry)
    ref = AssetRef(digest_bytes(b"source"), "source.gltf", 6)
    with pytest.raises(AssetError, match="contained relative"):
        registry.wrap(FILE_3D_TYPE, {"format": "gltf", "source": ref, "resources": {name: ref}})


@pytest.mark.parametrize("change", ("revoke", "replace", "index"))
def test_file3d_local_binding_uses_live_grant_and_only_reconciles_indexed_content(
    tmp_path: Path, change: str
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    source = root / "model.obj"
    source.write_bytes(b"v 1 2 3\nf 1 1 1\n")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef(id="models", path=root), kind="media/model3d")
    ref = table.bind_name("model.obj", "media/model3d")
    resolver = MountSnapshotResolver(snapshot)
    assert isinstance(ref, LocalAssetRef)
    registry = TypeRegistry()
    register_file3d_types(registry, resolver)
    spec = registry.spec(FILE_3D_TYPE)
    obj = {"format": "obj", "source": ref}
    local = registry.wrap(FILE_3D_TYPE, obj)
    payload = spec.encode(obj)
    wire = json.loads(payload)
    assert wire["source"]["verification"] == "unverified"
    assert "digest" not in wire["source"]
    assert json.loads(registry.render(local, "manifest").data) == wire
    assert local.meta.entries["asset_refs"] == []
    assert (
        cast(LocalAssetRef, cast(dict, spec.decode(payload))["source"]).read_bytes()
        == source.read_bytes()
    )
    foreign = TypeRegistry()
    register_file3d_types(foreign, MountSnapshotResolver(tmp_path / "foreign-mounts.json"))
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        foreign.spec(FILE_3D_TYPE).decode(payload)
    if change == "index":
        expected = source.read_bytes()
        table.scan("models")
        reconciled = registry.wrap(FILE_3D_TYPE, obj)
        portable = cast(AssetRef, cast(dict, reconciled.resolve())["source"])
        assert portable.digest == digest_bytes(expected)
        assert portable.read_bytes() == expected
        assert reconciled.meta.entries["asset_refs"] == [portable.to_wire()]
        assert reconciled.fingerprint != local.fingerprint
    else:
        if change == "revoke":
            table.remove("models")
        else:
            source.write_bytes(b"changed geometry")
        with pytest.raises(AssetError, match="unavailable, revoked or changed"):
            spec.decode(payload)
        with pytest.raises(AssetError, match="unavailable, revoked or changed"):
            registry.render(local, "manifest")


@pytest.mark.parametrize("source", ("/host/model.obj", {"path": "/host/model.obj"}))
def test_file3d_source_cannot_introduce_filesystem_authority(source: object) -> None:
    registry = TypeRegistry()
    register_file3d_types(registry)
    with pytest.raises(AssetError):
        registry.wrap(FILE_3D_TYPE, {"format": "obj", "source": source})


@pytest.mark.parametrize(
    "type_id,format_id",
    (
        ("FILE_3D", "custom"),
        ("FILE_3D_GLB", "glb"),
        ("FILE_3D_GLTF", "gltf"),
        ("FILE_3D_FBX", "fbx"),
        ("FILE_3D_OBJ", "obj"),
        ("FILE_3D_STL", "stl"),
        ("FILE_3D_USDZ", "usdz"),
        ("FILE_3D_PLY", "ply"),
        ("FILE_3D_SPLAT", "splat"),
        ("FILE_3D_SPZ", "spz"),
        ("FILE_3D_KSPLAT", "ksplat"),
        ("FILE_3D_SPLAT_ANY", "ply"),
        ("FILE_3D_SPLAT_ANY", "splat"),
        ("FILE_3D_SPLAT_ANY", "spz"),
        ("FILE_3D_SPLAT_ANY", "ksplat"),
        ("FILE_3D_POINT_CLOUD_ANY", "ply"),
        ("FILE_3D", ""),
        ("FILE_3D_SPLAT_ANY", ""),
        ("FILE_3D_POINT_CLOUD_ANY", ""),
    ),
)
def test_file3d_format_families_retain_original_format(type_id: str, format_id: str) -> None:
    registry = TypeRegistry()
    register_file3d_types(registry)
    ref = AssetRef(digest_bytes(b"original"), "source", 8)
    resource = AssetRef(digest_bytes(b"sidecar"), "data.bin", 7)
    obj = {"format": format_id, "source": ref, "resources": {"mesh/data.bin": resource}}
    spec = registry.spec(type_id)
    encoded = spec.encode(obj)
    decoded = cast("dict[str, object]", spec.decode(encoded))
    assert decoded == obj
    value = registry.wrap(type_id, obj)
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, value.meta.entries)
    assert dict(value.meta.entries) == {
        "format": format_id,
        "asset_refs": [ref.to_wire(), resource.to_wire()],
    }
    rendition = registry.render(value, "manifest")
    assert json.loads(rendition.data) == {
        "format": format_id,
        "source": ref.to_wire(),
        "resources": {"mesh/data.bin": resource.to_wire()},
    }
    assert spec.fingerprint is not None
    assert spec.fingerprint(obj) == registry.wrap(type_id, decoded).fingerprint
    wrong = {**obj, "format": "gltf" if type_id == FILE_3D_GLB_TYPE else "glb"}
    if type_id != FILE_3D_TYPE:
        with pytest.raises(AssetError, match="does not accept"):
            spec.encode(wrong)


@pytest.mark.parametrize("format_id", (None, False, 0, ".ply", "PLY", "../ply"))
def test_file3d_unspecified_format_does_not_accept_malformed_formats(format_id: object) -> None:
    registry = TypeRegistry()
    register_file3d_types(registry)
    ref = AssetRef(digest_bytes(b"original"), "source", 8)
    obj = {"format": format_id, "source": ref}
    spec = registry.spec(FILE_3D_TYPE)
    with pytest.raises(AssetError, match="format"):
        spec.encode(obj)
    payload = json.dumps({**obj, "source": ref.to_wire()}).encode()
    with pytest.raises(AssetError, match="format"):
        spec.decode(payload)


@pytest.mark.parametrize(
    "type_id",
    (
        "FILE_3D_GLB",
        "FILE_3D_GLTF",
        "FILE_3D_FBX",
        "FILE_3D_OBJ",
        "FILE_3D_STL",
        "FILE_3D_USDZ",
        "FILE_3D_PLY",
        "FILE_3D_SPLAT",
        "FILE_3D_SPZ",
        "FILE_3D_KSPLAT",
    ),
)
def test_file3d_concrete_formats_require_declared_format(type_id: str) -> None:
    registry = TypeRegistry()
    register_file3d_types(registry)
    ref = AssetRef(digest_bytes(b"original"), "source", 8)
    obj = {"format": "", "source": ref}
    spec = registry.spec(type_id)
    with pytest.raises(AssetError, match="does not accept"):
        spec.encode(obj)
    payload = json.dumps({**obj, "source": ref.to_wire()}).encode()
    with pytest.raises(AssetError, match="does not accept"):
        spec.decode(payload)


FILE3D_SOURCE = b'{"asset":{"version":"2.0"},"buffers":[{"uri":"mesh/data.bin"}]}'
FILE3D_SIDECAR = b"\x00\x02\xfe\xff"


def register_file3d_fixture_types(registry: TypeRegistry) -> None:
    register_file3d_types(registry, resolver_from_env())


class File3DProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="file3d-source.make",
            inputs=(InputSpec("empty", TypeExpr.concrete("BOOLEAN"), default=False),),
            outputs=(OutputSpec("file", TypeExpr.concrete(FILE_3D_GLTF_TYPE)),),
        )

    @classmethod
    def execute(cls, *, empty: bool = False) -> Mapping[str, object]:
        if empty:
            return cls.outputs(file=None)
        vault = produced_output_scratch()
        assert vault is not None
        refs = []
        for data, name in ((FILE3D_SOURCE, "scene.gltf"), (FILE3D_SIDECAR, "data.bin")):
            digest = digest_bytes(data)
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
            refs.append(AssetRef(digest, name, len(data), resolver=vault))
        return cls.outputs(
            file={"format": "gltf", "source": refs[0], "resources": {"mesh/data.bin": refs[1]}}
        )


class File3DConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="file3d-consumer.read",
            inputs=(InputSpec("file", TypeExpr.concrete(FILE_3D_GLTF_TYPE)),),
            outputs=(OutputSpec("file", TypeExpr.concrete(FILE_3D_GLTF_TYPE)),),
        )

    @classmethod
    def execute(cls, *, file: dict[str, object] | None) -> Mapping[str, object]:
        if file is not None:
            assert cast(AssetRef, file["source"]).read_bytes() == FILE3D_SOURCE
            assert (
                cast(dict[str, AssetRef], file["resources"])["mesh/data.bin"].read_bytes()
                == FILE3D_SIDECAR
            )
        return cls.outputs(file=file)


@pytest.mark.parametrize("empty", (False, True))
def test_file3d_sources_and_sidecars_survive_workers_and_persistent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    vault = AssetVault(tmp_path / "engine-vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    manifests = []
    for name, node, namespace in (
        ("source", "File3DProducer", "file3d-source"),
        ("consumer", "File3DConsumer", "file3d-consumer"),
    ):
        module = f"file3d_{name}"
        (tmp_path / f"{module}.py").write_text(
            f"from test_model3d_io import {node}, register_file3d_fixture_types\n"
            f"NODES = ({node},)\n",
            encoding="ascii",
        )
        manifest = tmp_path / f"{name}.toml"
        manifest.write_text(
            f'[pack]\nname="file3d-{name}"\npublisher="dinkster"\n'
            f'namespaces=["{namespace}"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_file3d_fixture_types"\n',
            encoding="ascii",
        )
        manifests.append(manifest)

    async def scenario() -> None:
        graph = Graph(
            nodes={
                "source": GraphNode("file3d-source.make", {"empty": empty}),
                "consumer": GraphNode("file3d-consumer.read", {"file": Link("source", "file")}),
            }
        )
        for replay in (False, True):
            registry = TypeRegistry()
            register_core_types(registry)
            register_file3d_types(registry, vault)
            composer = ServingComposer(
                registry=registry,
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                pack_scratch_root=tmp_path / "scratch",
                worker_env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "DINKSTER_ACCELERATOR": "cpu",
                    "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                },
            )
            try:
                for manifest in manifests:
                    await composer.add_pack(
                        PackSpec(
                            manifest=manifest,
                            worker_group=manifest.stem,
                            group_manifests=(manifest,),
                        )
                    )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["consumer"]
                )
                value = result.outputs["consumer"]["file"]
                assert set(result.cached if replay else result.executed) == {"source", "consumer"}
            finally:
                await composer.close()
            shutil.rmtree(tmp_path / "scratch")
            if empty:
                assert value.type_id == FILE_3D_GLTF_TYPE
                assert value.resolve() is None
                assert value.meta.entries == {"asset_refs": []}
                continue
            restored = cast(dict[str, object], value.resolve())
            source = cast(AssetRef, restored["source"])
            sidecar = cast(dict[str, AssetRef], restored["resources"])["mesh/data.bin"]
            assert source.read_bytes() == FILE3D_SOURCE
            assert sidecar.read_bytes() == FILE3D_SIDECAR
            assert vault.has(source.digest) and vault.has(sidecar.digest)

    asyncio.run(scenario())


def test_schemas_declare_upload_save_and_preview_intent() -> None:
    load = LoadModel3D.define_schema()
    model_input = load.input("model")
    assert model_input is not None
    assert model_input.source_filename is not None
    assert model_input.source_filename.kind == "media/model3d"
    load_output = load.output("model")
    assert load_output is not None and load_output.preview
    save = SaveModel3D.define_schema()
    assert save.output_node and not save.idempotent
    preview = PreviewModel3D.define_schema()
    assert preview.output_node
    preview_output = preview.output("model")
    assert preview_output is not None and preview_output.preview


def test_authored_viewport_captures_rebind_without_granting_paths(tmp_path: Path) -> None:
    sender = AssetVault(tmp_path / "sender")
    receiver = AssetVault(tmp_path / "receiver")
    captures = {}
    for index, name in enumerate(("image", "mask", "normal", "recording")):
        data = bytes([index, 17, 93]) * (index + 2)
        digest = digest_bytes(data)
        for vault in (sender, receiver):
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
        captures[name] = AssetRef(digest, name, len(data), resolver=sender)
    source_registry = TypeRegistry()
    register_asset_type(source_registry, sender)
    source = register_load3d_type(source_registry)
    target_registry = TypeRegistry()
    register_asset_type(target_registry, receiver)
    target = register_load3d_type(target_registry)
    state = {
        **captures,
        "camera_info": {"position": {"x": 3, "y": -2, "z": 7}, "zoom": 1.75},
        "model_3d_info": [{"position": {"x": -4, "y": 8, "z": 1}}],
        "background": {"enabled": True, "color": "#182c67"},
    }
    encoded = source.encode(state)
    decoded = cast(dict[str, object], target.decode(encoded))
    assert decoded["camera_info"] == state["camera_info"]
    assert decoded["model_3d_info"] == state["model_3d_info"]
    assert decoded["background"] == state["background"]
    for name, original in captures.items():
        received = decoded[name]
        assert isinstance(received, AssetRef)
        assert received.resolver is receiver
        assert received.read_bytes() == original.read_bytes()
    assert source.fingerprint is not None and target.fingerprint is not None
    assert source.fingerprint(state) == target.fingerprint(decoded)
    assert target.meta is not None and target.validate_encoded is not None
    target.validate_encoded(encoded, target.meta(decoded))
    with pytest.raises(AssetError, match="metadata"):
        target.validate_encoded(encoded, {"asset_refs": []})
    with pytest.raises(AssetError, match="asset inputs"):
        source.encode({"image": "/tmp/ungranted.png"})
    assert target.decode(source.encode({"recording": "", "camera_info": None})) == {
        "recording": "",
        "camera_info": None,
    }
    assert target.decode(source.encode({})) == {}


@pytest.mark.parametrize("change", ("revoke", "replace", "index"))
def test_viewport_captures_retain_local_grant_authority(tmp_path: Path, change: str) -> None:
    root = tmp_path / "captures"
    root.mkdir()
    capture = root / "frame.png"
    capture.write_bytes(b"original capture bytes")
    snapshot = tmp_path / "mounts.json"
    mounts = MountTable(snapshot)
    mounts.add(MountDef(id="captures", path=root), kind="media/image")
    ref = mounts.bind_name("frame.png", "media/image")
    assert isinstance(ref, LocalAssetRef)
    registry = TypeRegistry()
    register_asset_type(registry, MountSnapshotResolver(snapshot))
    spec = register_load3d_type(registry)
    value = {"image": ref, "recording": ""}
    payload = spec.encode(value)
    foreign = TypeRegistry()
    register_asset_type(foreign, MountSnapshotResolver(tmp_path / "foreign.json"))
    foreign_spec = register_load3d_type(foreign)
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        foreign_spec.decode(payload)
    restored = cast(dict[str, object], spec.decode(payload))
    assert cast(LocalAssetRef, restored["image"]).read_bytes() == b"original capture bytes"
    if change == "index":
        mounts.scan("captures")
        indexed = cast(dict[str, object], spec.decode(payload))["image"]
        assert isinstance(indexed, AssetRef)
        assert indexed.digest == digest_bytes(b"original capture bytes")
        assert indexed.read_bytes() == b"original capture bytes"
    else:
        if change == "revoke":
            mounts.remove("captures")
        else:
            capture.write_bytes(b"replacement bytes")
        with pytest.raises(AssetError, match="unavailable, revoked or changed"):
            spec.decode(payload)


VIEWPORT_CAPTURE = b"\x01\xff\x03\x00asymmetric capture"
VIEWPORT_CAMERA = {"position": {"x": 3, "y": -7, "z": 2}, "zoom": 1.25}


def register_viewport_fixture_types(registry: TypeRegistry) -> None:
    register_asset_type(registry, resolver_from_env())
    register_load3d_type(registry)


class ViewportCaptureProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="viewport-source.make",
            outputs=(OutputSpec("viewport", TypeExpr.concrete(LOAD_3D_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        vault = produced_output_scratch()
        assert vault is not None
        digest = digest_bytes(VIEWPORT_CAPTURE)
        with vault.writer(digest) as writer:
            writer.write(VIEWPORT_CAPTURE)
            writer.commit()
        return cls.outputs(
            viewport={
                "image": AssetRef(digest, "frame.png", len(VIEWPORT_CAPTURE), resolver=vault),
                "camera_info": VIEWPORT_CAMERA,
                "recording": "",
            }
        )


class ViewportCaptureConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="viewport-consumer.read",
            inputs=(InputSpec("viewport", TypeExpr.concrete(LOAD_3D_TYPE)),),
            outputs=(OutputSpec("viewport", TypeExpr.concrete(LOAD_3D_TYPE)),),
        )

    @classmethod
    def execute(cls, *, viewport: dict[str, object]) -> Mapping[str, object]:
        assert cast(AssetRef, viewport["image"]).read_bytes() == VIEWPORT_CAPTURE
        assert viewport["camera_info"] == VIEWPORT_CAMERA
        assert viewport["recording"] == ""
        return cls.outputs(viewport=viewport)


def test_viewport_capture_survives_workers_and_persistent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = AssetVault(tmp_path / "engine-vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    manifests = []
    for name, node in (
        ("source", "ViewportCaptureProducer"),
        ("consumer", "ViewportCaptureConsumer"),
    ):
        module = f"viewport_{name}"
        (tmp_path / f"{module}.py").write_text(
            f"from test_model3d_io import {node}, register_viewport_fixture_types\n"
            f"NODES = ({node},)\n",
            encoding="ascii",
        )
        manifest = tmp_path / f"{name}.toml"
        manifest.write_text(
            f'[pack]\nname="viewport-{name}"\npublisher="dinkster"\n'
            f'namespaces=["viewport-{name}"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_viewport_fixture_types"\n',
            encoding="ascii",
        )
        manifests.append(manifest)

    async def scenario() -> None:
        graph = Graph(
            nodes={
                "source": GraphNode("viewport-source.make"),
                "consumer": GraphNode(
                    "viewport-consumer.read", {"viewport": Link("source", "viewport")}
                ),
            }
        )
        for replay in (False, True):
            registry = TypeRegistry()
            register_core_types(registry)
            register_asset_type(registry, vault)
            register_load3d_type(registry)
            composer = ServingComposer(
                registry=registry,
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                pack_scratch_root=tmp_path / "scratch",
                worker_env={
                    "DINKSTER_ACCELERATOR": "cpu",
                    "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                },
            )
            try:
                for manifest in manifests:
                    await composer.add_pack(
                        PackSpec(
                            manifest=manifest,
                            worker_group=manifest.stem,
                            group_manifests=(manifest,),
                        )
                    )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["consumer"]
                )
                value = result.outputs["consumer"]["viewport"]
                assert set(result.cached if replay else result.executed) == {"source", "consumer"}
            finally:
                await composer.close()
            shutil.rmtree(tmp_path / "scratch")
            restored = cast(dict[str, object], value.resolve())
            capture = cast(AssetRef, restored["image"])
            assert capture.read_bytes() == VIEWPORT_CAPTURE
            assert restored["camera_info"] == VIEWPORT_CAMERA
            assert vault.has(capture.digest)

    asyncio.run(scenario())
