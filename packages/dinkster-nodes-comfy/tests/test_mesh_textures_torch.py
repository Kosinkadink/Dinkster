"""Baked mesh channel extraction and ordinary-host transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    IMAGE_TYPE,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_textures import MeshTextureToImage

from dinkster.compose import PackSpec, ServingComposer


def textured_mesh(baked: bool = True) -> TriangleMeshBatch[Any]:
    return TriangleMeshBatch(
        vertices=torch.zeros(1, 3, 3),
        faces=torch.tensor([[[0, 1, 2]]]),
        texture=torch.tensor([[[-0.2, 0.3, 1.4], [0.9, 0.6, 0.1]]], dtype=torch.float64),
        metallic_roughness=torch.tensor([[[0.2, 0.4, 0.7], [0.8, 0.6, 0.1]]]) if baked else None,
        normal_map=torch.tensor([[[0.1, 0.6, 1.2], [-0.1, 0.8, 0.9]]]) if baked else None,
        occlusion_in_mr=baked,
    )


@pytest.mark.parametrize("baked", (True, False))
def test_texture_channels_defaults_and_source_preservation(baked: bool) -> None:
    mesh = textured_mesh(baked)
    before = mesh.texture.clone()
    result: Any = MeshTextureToImage.execute(mesh=mesh)
    assert result["base_color"].shape == (1, 1, 2, 3)
    assert result["base_color"].dtype == torch.float32
    torch.testing.assert_close(
        result["base_color"], torch.tensor([[[[0.0, 0.3, 1.0], [0.9, 0.6, 0.1]]]])
    )
    expected = ((0.7, 0.1), (0.4, 0.6), (0.2, 0.8)) if baked else ((0, 0), (0, 0), (1, 1))
    for key, values in zip(("metallic", "roughness", "occlusion"), expected, strict=True):
        torch.testing.assert_close(
            result[key], torch.tensor(values).float()[None, None, :, None].expand(1, 1, 2, 3)
        )
        assert result[key].is_contiguous()
    normal = (
        torch.tensor([[[[0.1, 0.6, 1.0], [0.0, 0.8, 0.9]]]])
        if baked
        else torch.tensor([0.5, 0.5, 1.0]).expand(1, 1, 2, 3)
    )
    torch.testing.assert_close(result["normal_map"], normal)
    torch.testing.assert_close(mesh.texture, before, rtol=0, atol=0)


def test_unbaked_occlusion_and_missing_base_texture() -> None:
    mesh = textured_mesh()
    mesh.occlusion_in_mr = False
    result: Any = MeshTextureToImage.execute(mesh=mesh)
    assert torch.equal(result["occlusion"], torch.ones(1, 1, 2, 3))
    mesh.texture = None
    with pytest.raises(ValueError, match="mesh has no baseColor texture"):
        MeshTextureToImage.execute(mesh=mesh)


class TextureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mesh-texture-fixture.source",
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=textured_mesh())


class TextureObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mesh-texture-fixture.observe",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                for name in ("base_color", "metallic", "roughness", "occlusion", "normal_map")
            ),
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls, **images: Any) -> dict[str, object]:
        actual = {name: materialize_image(image) for name, image in images.items()}
        for name, values in (
            ("metallic", (0.7, 0.1)),
            ("roughness", (0.4, 0.6)),
            ("occlusion", (0.2, 0.8)),
        ):
            torch.testing.assert_close(
                actual[name], torch.tensor(values)[None, None, :, None].expand(1, 1, 2, 3)
            )
        torch.testing.assert_close(
            actual["normal_map"], torch.tensor([[[[0.1, 0.6, 1.0], [0.0, 0.8, 0.9]]]])
        )
        return cls.outputs(image=actual["base_color"])


@pytest.mark.parametrize("in_process", (True, False))
def test_mesh_textures_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "texture_fixture.py").write_text(
            "from test_mesh_textures_torch import TextureSource, TextureObserver\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES=(TextureSource, TextureObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="mesh-texture-fixture"\npublisher="dinkster"\nnamespaces=["mesh-texture-fixture"]\n[pack.entry]\nnodes="texture_fixture:NODES"\ntypes="texture_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="mesh-texture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("mesh-texture-fixture.source"),
                    "extract": GraphNode("MeshTextureToImage", {"mesh": Link("source", "mesh")}),
                    "observe": GraphNode(
                        "mesh-texture-fixture.observe",
                        {
                            name: Link("extract", name)
                            for name in (
                                "base_color",
                                "metallic",
                                "roughness",
                                "occlusion",
                                "normal_map",
                            )
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observe"])
                torch.testing.assert_close(
                    materialize_image(result.outputs["observe"]["image"].resolve()),
                    torch.tensor([[[[0.0, 0.3, 1.0], [0.9, 0.6, 0.1]]]]),
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
