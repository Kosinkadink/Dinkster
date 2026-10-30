"""Native asset-backed 3D model I/O contracts."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import cast

import dinkster_nodes_media_io.model3d as model3d_module
import pytest
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    MountSnapshotResolver,
    digest_bytes,
    register_asset_type,
    register_load3d_type,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_media_io import LoadModel3D, PreviewModel3D, SaveModel3D, register_media_types
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, decode_model3d, register_core_types
from dinkster_workers import InProcessWorker

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


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
