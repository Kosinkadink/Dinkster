"""Baked texture extraction preserves channel meaning and optional-map defaults."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    MESH_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    materialize_image,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_texture import MeshTextureToImage

from dinkster.compose import PackSpec, ServingComposer


def source() -> Any:
    return TriangleMeshBatch(
        vertices=torch.zeros(1, 3, 3),
        faces=torch.tensor([[[0, 1, 2]]]),
        texture=torch.tensor([[[[-1, 0.25, 2], [0.75, 0.5, 0]]]], dtype=torch.float64),
        metallic_roughness=torch.tensor([[[[0.125, 0.375, 0.625], [0.875, 1.25, -0.5]]]]),
        normal_map=torch.tensor([[[[0.2, 0.4, 0.9], [0.7, 0.3, 0.8]]]], dtype=torch.float64),
        occlusion_in_mr=True,
    )


def assert_channels(result: Any) -> None:
    expected = {
        "base_color": [[[[0, 0.25, 1], [0.75, 0.5, 0]]]],
        "metallic": [[[[0.625] * 3, [0] * 3]]],
        "roughness": [[[[0.375] * 3, [1] * 3]]],
        "occlusion": [[[[0.125] * 3, [0.875] * 3]]],
        "normal_map": [[[[0.2, 0.4, 0.9], [0.7, 0.3, 0.8]]]],
    }
    for name, values in expected.items():
        actual = materialize_image(result[name])
        torch.testing.assert_close(actual, torch.tensor(values), rtol=0, atol=0)
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        assert actual.is_contiguous()


@pytest.mark.parametrize("portable", [False, True])
def test_baked_channels_and_source_preservation(portable: bool) -> None:
    mesh = source()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(mesh)) if portable else mesh
    assert_channels(MeshTextureToImage.execute(mesh=value))
    assert mesh.texture.dtype == torch.float64
    assert mesh.texture[0, 0, 0, 0] == -1 and mesh.texture[0, 0, 0, 2] == 2


def test_optional_maps_unbatched_texture_and_missing_base() -> None:
    mesh = source()
    mesh.texture = mesh.texture.squeeze(0)
    mesh.metallic_roughness = mesh.normal_map = None
    mesh.occlusion_in_mr = False
    result = MeshTextureToImage.execute(mesh=mesh)
    assert result["base_color"].shape == (1, 1, 2, 3)
    for name in ("metallic", "roughness"):
        torch.testing.assert_close(result[name], torch.zeros(1, 1, 2, 3), rtol=0, atol=0)
    torch.testing.assert_close(result["occlusion"], torch.ones(1, 1, 2, 3), rtol=0, atol=0)
    torch.testing.assert_close(
        result["normal_map"], torch.tensor([[[[0.5, 0.5, 1], [0.5, 0.5, 1]]]]), rtol=0, atol=0
    )
    mesh.metallic_roughness = source().metallic_roughness
    result = MeshTextureToImage.execute(mesh=mesh)
    torch.testing.assert_close(result["occlusion"], torch.ones(1, 1, 2, 3), rtol=0, atol=0)
    mesh.texture = None
    with pytest.raises(ValueError, match="mesh has no baseColor texture"):
        MeshTextureToImage.execute(mesh=mesh)


class TextureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "texture-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_texture_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "texture_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_texture_torch import TextureSource\nNODES=(TextureSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="texture-fixture"\npublisher="dinkster"\nnamespaces=["texture-fixture"]\n'
            '[pack.entry]\nnodes="texture_fixture:NODES"\ntypes="texture_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "source": GraphNode("texture-fixture.source"),
                    "texture": GraphNode("MeshTextureToImage", {"mesh": Link("source", "mesh")}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["texture"])
                assert_channels({k: v.resolve() for k, v in result.outputs["texture"].items()})
        finally:
            await composer.close()

    asyncio.run(scenario())
