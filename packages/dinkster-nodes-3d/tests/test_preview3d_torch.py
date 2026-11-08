"""Deprecated 3D preview values through the ordinary public file boundary."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_3d import register_types as register_inference_types
from dinkster_3d.preview3d import Preview3D
from dinkster_api.v1 import (
    FILE_3D_TYPE,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_assets import (
    AssetVault,
    LibraryStore,
    LocalAssetRef,
    MountDef,
    MountTable,
    load_file3d,
)
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_server.library import ServerLibrary
from dinkster_server.preflight import bind_model_names
from dinkster_values import ABSENT_REASON_META_KEY, TypeRegistry, register_core_types
from geometry_pack_fixtures import geometry_pack_specs

from dinkster.compose import ServingComposer


class PreviewMetadata(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "preview-metadata.make",
            outputs=(
                OutputSpec("camera", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                OutputSpec("background", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            camera={
                "position": {"x": 1.0, "y": 2.0, "z": -3.0},
                "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                "zoom": 2.75,
                "cameraType": "orthographic",
            },
            background=np.arange(2 * 7 * 11 * 3, dtype=np.float32).reshape(2, 7, 11, 3) / 512,
        )


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_deprecated_preview_preserves_granted_file_and_optional_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    canonical: bool,
    metadata: bool,
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    primary = b'{"asset":{"version":"2.0"},"buffers":[{"uri":"mesh.bin"}]}'
    (root / "scene.gltf").write_bytes(primary)
    (root / "mesh.bin").write_bytes(b"original declared sidecar")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef("models", root, kind="media/model3d"))
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    source = table.bind_name("scene.gltf", "media/model3d")
    assert isinstance(source, LocalAssetRef)
    selection: object = "scene.gltf"
    if canonical:
        packet = load_file3d(source)
        assert packet is not None
        selection = TypedLiteral(
            FILE_3D_TYPE,
            {
                "format": "gltf",
                "source": source.to_wire(),
                "resources": {
                    name: ref.to_wire()
                    for name, ref in cast("dict[str, LocalAssetRef]", packet["resources"]).items()
                },
            },
        )
    nodes = {"preview": GraphNode("dinkster.3d.preview3_d", {"model_file": selection})}
    if metadata:
        nodes["metadata"] = GraphNode("preview-metadata.make")
        nodes["preview"] = GraphNode(
            "dinkster.3d.preview3_d",
            {
                "model_file": selection,
                "camera_info": Link("metadata", "camera"),
                "bg_image": Link("metadata", "background"),
            },
        )
    library = ServerLibrary(
        AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library.sqlite"), resolver=table
    )
    graph, missing = bind_model_names(
        Graph(nodes), {"dinkster.3d.preview3_d": Preview3D.define_schema()}, library
    )
    assert missing == []
    module = f"preview_metadata_{in_process}_{canonical}_{metadata}"
    (tmp_path / f"{module}.py").write_text(
        "from test_preview3d_torch import PreviewMetadata\n"
        "from dinkster_3d import register_types as register_inference_types\n"
        "NODES=(PreviewMetadata,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "metadata.toml"
    manifest.write_text(
        '[pack]\nname="preview-metadata"\npublisher="dinkster"\n'
        'namespaces=["preview-metadata"]\n[pack.entry]\n'
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
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "DINKSTER_ASSET_VAULT": str(tmp_path / "vault"),
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            },
        )
        try:
            for spec in geometry_pack_specs(manifest, in_process=in_process):
                await composer.add_pack(spec)
            result = await composer.composition.make_engine(lambda _event: None).run(
                graph, ["preview"]
            )
            assert not result.artifacts
            out = result.outputs["preview"]
            value = cast("dict[str, object]", out["model_3d"].resolve())
            assert value["format"] == "gltf"
            assert cast(LocalAssetRef, value["source"]).read_bytes() == primary
            resources = cast("dict[str, LocalAssetRef]", value["resources"])
            assert resources["mesh.bin"].read_bytes() == b"original declared sidecar"
            if metadata:
                assert out["camera_info"].resolve() == {
                    "position": {"x": 1.0, "y": 2.0, "z": -3.0},
                    "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "zoom": 2.75,
                    "cameraType": "orthographic",
                }
                np.testing.assert_array_equal(
                    materialize_image(out["bg_image"].resolve()),
                    np.arange(2 * 7 * 11 * 3, dtype=np.float32).reshape(2, 7, 11, 3) / 512,
                )
            else:
                for name, reason in (("camera_info", "no camera"), ("bg_image", "no background")):
                    assert out[name].type_id == "core.absent"
                    assert out[name].meta.get(ABSENT_REASON_META_KEY) == reason
            assert sorted(p.name for p in root.iterdir()) == ["mesh.bin", "scene.gltf"]
            assert (root / "scene.gltf").read_bytes() == primary
        finally:
            await composer.close()

    asyncio.run(scenario())
