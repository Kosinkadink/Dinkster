"""Baked texture extraction preserves channel meaning and optional-map defaults."""

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
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    materialize_image,
    materialize_mesh,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_texture import ApplyTextureToMesh, MeshTextureToImage

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


def test_apply_normals_weld_coincident_uv_seams() -> None:
    mesh = TriangleMeshBatch(
        vertices=torch.tensor(
            [[[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 3], [2, 0, 0]]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[[0, 1, 2], [3, 4, 5]]]),
        uvs=torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 0]]], dtype=torch.float32),
    )
    image = torch.ones(1, 1, 1, 3)
    result = materialize_mesh(
        ApplyTextureToMesh.execute(mesh=mesh, base_color=image, normal_map=image)["mesh"]
    )
    shared = [0, 3 / (10**0.5), 1 / (10**0.5)]
    expected = torch.tensor([[shared, shared, [0, 0, 1], shared, [0, 1, 0], shared]])
    torch.testing.assert_close(result.normals, expected, rtol=1e-6, atol=0)


def apply_source() -> Any:
    vertices = [[0, 0, 0], [2, 0, 0], [0, 3, 0], [9, 9, 9]]
    return TriangleMeshBatch(
        vertices=torch.tensor([vertices, vertices], dtype=torch.float64),
        faces=torch.tensor([[[0, 1, 2], [3, 3, 3]], [[0, 1, 2], [3, 3, 3]]]),
        vertex_counts=torch.tensor([3, 3]),
        face_counts=torch.tensor([1, 1]),
        uvs=torch.tensor(
            [
                [[-2, 4], [2, 4], [-2, 6], [9, 9]],
                [[4, 2], [0, 2], [4, 4], [19, 19]],
            ],
            dtype=torch.float64,
        ),
        vertex_colors=torch.full((2, 4, 4), 0.7),
        emissive=torch.full((2, 1, 1, 3), 0.3),
        material={"doubleSided": True},
        unlit=True,
    )


def apply_inputs() -> dict[str, object]:
    return {
        "mesh": apply_source(),
        "base_color": torch.tensor([[[[-1, 0.25, 2]]], [[[0.75, 0.5, -2]]]], dtype=torch.float64),
        "metallic": torch.tensor([[[[0.2, 0, 0], [0.4, 0, 0]]], [[[0.8, 0, 0], [0.9, 0, 0]]]]),
        "roughness": torch.tensor(
            [[[[0.25, 0, 0]], [[0.75, 0, 0]]], [[[0.5, 0, 0]], [[0.125, 0, 0]]]]
        ),
        "occlusion": torch.tensor([[[[0.6, 0, 0]]], [[[1.2, 0, 0]]]]),
        "normal_map": torch.tensor([[[[0.1, 0.5, 0.9]]], [[[0.4, 0.3, 0.7]]]]),
    }


def assert_applied(value: object) -> None:
    result = materialize_mesh(value)
    source_mesh = apply_source()
    for name in ("vertices", "faces", "vertex_counts", "face_counts", "vertex_colors", "emissive"):
        torch.testing.assert_close(
            getattr(result, name), getattr(source_mesh, name), rtol=0, atol=0
        )
    expected_uvs = torch.tensor(
        [
            [[0, 0], [1, 0], [0, 0.5], [9, 9]],
            [[1, 0], [0, 0], [1, 0.5], [19, 19]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.uvs, expected_uvs, rtol=0, atol=0)
    torch.testing.assert_close(
        result.texture, torch.tensor([[[[0, 0.25, 1]]], [[[0.75, 0.5, 0]]]]), rtol=0, atol=0
    )
    expected_orm = torch.tensor(
        [
            [[[0.6, 0.25, 0.2], [0.6, 0.25, 0.4]], [[0.6, 0.75, 0.2], [0.6, 0.75, 0.4]]],
            [[[1, 0.5, 0.8], [1, 0.5, 0.9]], [[1, 0.125, 0.8], [1, 0.125, 0.9]]],
        ]
    )
    torch.testing.assert_close(result.metallic_roughness, expected_orm, rtol=0, atol=0)
    normals = torch.tensor([[[0, 0, 1]] * 3 + [[0, 0, 0]]] * 2, dtype=torch.float32)
    tangents = torch.tensor(
        [[[1, 0, 0, 1]] * 3 + [[0, 0, 0, 0]], [[-1, 0, 0, -1]] * 3 + [[0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.normals, normals, rtol=0, atol=0)
    torch.testing.assert_close(result.tangents, tangents, rtol=0, atol=0)
    torch.testing.assert_close(result.normal_map, apply_inputs()["normal_map"], rtol=0, atol=0)
    assert result.unlit and result.occlusion_in_mr and result.material == {"doubleSided": True}


@pytest.mark.parametrize("portable", [False, True])
def test_apply_maps_uv_fit_basis_and_source_preservation(portable: bool) -> None:
    inputs = apply_inputs()
    mesh = inputs["mesh"]
    registry = TypeRegistry()
    register_inference_types(registry)
    if portable:
        codec = registry.spec(MESH_TYPE)
        inputs["mesh"] = codec.decode(codec.encode(mesh))
    assert_applied(ApplyTextureToMesh.execute(**inputs)["mesh"])
    torch.testing.assert_close(materialize_mesh(mesh).uvs, apply_source().uvs, rtol=0, atol=0)
    assert materialize_mesh(mesh).normals is None and materialize_mesh(mesh).tangents is None


@pytest.mark.parametrize("channel", ["metallic", "roughness", "occlusion"])
def test_apply_orm_defaults_and_retained_optional_attributes(channel: str) -> None:
    mesh = apply_source()
    mesh.normals = torch.full((2, 4, 3), 0.4)
    mesh.tangents = torch.full((2, 4, 4), 0.3)
    mesh.normal_map = torch.full((2, 1, 1, 3), 0.2)
    base = torch.ones(2, 1, 1, 3)
    result = materialize_mesh(
        ApplyTextureToMesh.execute(
            mesh=mesh, base_color=base, **{channel: torch.full((2, 2, 3, 3), 0.25)}
        )["mesh"]
    )
    expected = {"metallic": [1, 1, 0.25], "roughness": [1, 0.25, 0], "occlusion": [0.25, 1, 0]}[
        channel
    ]
    torch.testing.assert_close(
        result.metallic_roughness, torch.tensor(expected).expand(2, 2, 3, 3), rtol=0, atol=0
    )
    for name in ("normals", "tangents", "normal_map"):
        torch.testing.assert_close(getattr(result, name), getattr(mesh, name), rtol=0, atol=0)
    assert result.occlusion_in_mr == (channel == "occlusion")


def test_apply_existing_normals_empty_faces_degenerate_uvs_and_missing_uvs() -> None:
    mesh = apply_source()
    mesh.normals = torch.tensor([[[0, 0, 2]] * 4] * 2, dtype=torch.float64)
    mesh.face_counts[1] = 0
    mesh.uvs[0] = 0
    image = torch.ones(2, 1, 1, 3)
    result = materialize_mesh(
        ApplyTextureToMesh.execute(mesh=mesh, base_color=image, normal_map=image)["mesh"]
    )
    torch.testing.assert_close(
        result.normals[0, :3], torch.tensor([[0, 0, 2]] * 3, dtype=torch.float32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.tangents[0, :3],
        torch.tensor([[0, 0, 0, 1]] * 3, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert result.normals[1].count_nonzero() == 0 and result.tangents[1].count_nonzero() == 0
    mesh.uvs = None
    with pytest.raises(ValueError, match="mesh has no UVs"):
        ApplyTextureToMesh.execute(mesh=mesh, base_color=image)


class ApplyTextureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "apply-fixture.source",
            outputs=(
                OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                *(
                    OutputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                    for name in ("base_color", "metallic", "roughness", "occlusion", "normal_map")
                ),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(**apply_inputs())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_apply_texture_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "apply_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_texture_torch import ApplyTextureSource\nNODES=(ApplyTextureSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="apply-fixture"\npublisher="dinkster"\nnamespaces=["apply-fixture"]\n'
            '[pack.entry]\nnodes="apply_fixture:NODES"\ntypes="apply_fixture:register_inference_types"\n',
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
                    "source": GraphNode("apply-fixture.source"),
                    "apply": GraphNode(
                        "ApplyTextureToMesh",
                        {name: Link("source", name) for name in apply_inputs()},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["apply"])
                assert_applied(result.outputs["apply"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
