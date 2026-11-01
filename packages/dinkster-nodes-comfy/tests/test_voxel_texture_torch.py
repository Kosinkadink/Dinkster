"""Sparse interpolation, PBR channel layout and ordinary-host voxel texture baking."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import MESH_TYPE, VOXEL_TYPE, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.voxel_texture import BakeTextureFromVoxel
from test_mesh_bake_torch import plane

from dinkster.compose import PackSpec, ServingComposer


def fixture() -> tuple[Any, dict]:
    mesh = plane(extent=0.25)
    mesh.uvs = mesh.uvs * 0.5 + 0.25
    voxel = {
        "data": torch.tensor([[0, 0, 0], [1, 0, 0]]),
        "voxel_colors": torch.tensor([[0.2, 0.4, 0.6, 0.1, 0.8, 0], [0.8, 0.6, 0.4, 0.9, 0.3, 1]]),
        "resolution": 2,
    }
    return mesh, voxel


def test_voxel_bake_normalized_corners_pbr_projection_and_empty_batches() -> None:
    mesh, voxel = fixture()
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    # Only the x-axis varies: missing y/z corners must not dilute occupied attributes.
    x_weight = torch.tensor([1.0 / 64, 63.0 / 64])
    expected = torch.lerp(voxel["voxel_colors"][0], voxel["voxel_colors"][1], x_weight[:, None])
    torch.testing.assert_close(result["base_color"][0, 32, [16, 47]], expected[:, :3])
    torch.testing.assert_close(result["metallic"][0, 32, [16, 47], 0], expected[:, 3])
    torch.testing.assert_close(result["roughness"][0, 32, [16, 47], 0], expected[:, 4])
    torch.testing.assert_close(result["base_color"][0, 32, 0], expected[0, :3])
    reference = plane(extent=0.25)
    reference.vertices[..., 0] = -0.25
    reference.vertices[0, 1:3, 2] = 0.5
    projected = BakeTextureFromVoxel.execute(
        mesh=mesh, voxel_colors=voxel, texture_size=64, reference_mesh=reference
    )
    torch.testing.assert_close(
        projected["base_color"], voxel["voxel_colors"][0, :3].expand(1, 64, 64, 3)
    )
    mesh.vertices, mesh.faces, mesh.uvs = (
        mesh.vertices.repeat(3, 1, 1),
        mesh.faces.repeat(3, 1, 1),
        mesh.uvs.repeat(3, 1, 1),
    )
    mesh.vertex_counts, mesh.face_counts = torch.tensor([4, 3, 4]), torch.tensor([2, 0, 2])
    voxel["data"] = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])
    batched = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    for key, image in batched.items():
        torch.testing.assert_close(image[:1], result[key])
        assert torch.count_nonzero(image[1:]) == 0
        assert image.dtype == torch.float32 and image.device.type == "cpu"


def test_voxel_bake_nearest_fallback_rgb_and_missing_uv() -> None:
    mesh, voxel = fixture()
    mesh.vertices += 10
    voxel["voxel_colors"] = voxel["voxel_colors"][:, :3]
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    torch.testing.assert_close(result["base_color"], voxel["voxel_colors"][1].expand(1, 64, 64, 3))
    assert not torch.count_nonzero(result["metallic"]) and not torch.count_nonzero(
        result["roughness"]
    )
    mesh.uvs = None
    with pytest.raises(ValueError, match="no UVs"):
        BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)


def test_voxel_bake_large_sparse_keys_and_many_distant_texels() -> None:
    mesh, voxel = fixture()
    # Cubic keys exceed int32; more than 8192 stragglers use the spatial-tree tail.
    voxel["resolution"] = 4096
    voxel["data"] = torch.tensor([[4095, 4095, 4095], [2048, 1024, 512]])
    mesh.vertices += 10
    mesh.uvs = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=128)
    torch.testing.assert_close(
        result["base_color"], voxel["voxel_colors"][0, :3].expand(1, 128, 128, 3)
    )
    torch.testing.assert_close(
        result["metallic"], voxel["voxel_colors"][0, 3].expand(1, 128, 128, 3)
    )


def test_voxel_bake_oom_fallback_does_not_hide_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_nodes_comfy import voxel_texture

    mesh, voxel = fixture()
    voxel["data"] = voxel["data"][:1]
    voxel["voxel_colors"] = voxel["voxel_colors"][:1]
    original = voxel_texture._trilinear

    def exhausted(positions: Any, *args: Any) -> Any:
        if positions.dtype == torch.float32:
            raise torch.OutOfMemoryError("injected allocation failure")
        return original(positions, *args)

    monkeypatch.setattr(voxel_texture, "_trilinear", exhausted)
    result = BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)
    torch.testing.assert_close(
        result["base_color"], voxel["voxel_colors"][0, :3].expand(1, 64, 64, 3)
    )

    def invalid(*args: Any) -> Any:
        raise ValueError("invalid voxel attributes")

    monkeypatch.setattr(voxel_texture, "_trilinear", invalid)
    with pytest.raises(ValueError, match="invalid voxel attributes"):
        BakeTextureFromVoxel.execute(mesh=mesh, voxel_colors=voxel, texture_size=64)


class VoxelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="voxel-bake-source.make",
            inputs=(),
            outputs=(
                OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                OutputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict:
        mesh, voxel = fixture()
        return cls.outputs(mesh=mesh, voxel=voxel)


@pytest.mark.parametrize("in_process", [True, False])
def test_voxel_bake_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    module = f"voxel_bake_source_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from test_voxel_texture_torch import VoxelSource\n"
        "from dinkster_nodes_comfy import register_inference_types\nNODES=(VoxelSource,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="voxel-bake-source"\npublisher="dinkster"\nnamespaces=["voxel-bake-source"]\n[pack.entry]\n'
        + f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
            }
        )
        try:
            for path in (Path(__file__).parents[1] / "dinkster-pack.toml", manifest):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            graph = Graph(
                {
                    "source": GraphNode("voxel-bake-source.make"),
                    "bake": GraphNode(
                        "BakeTextureFromVoxel",
                        {
                            "mesh": Link("source", "mesh"),
                            "voxel_colors": Link("source", "voxel"),
                            "texture_size": 64,
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = BakeTextureFromVoxel.execute(
                mesh=fixture()[0], voxel_colors=fixture()[1], texture_size=64
            )
            for _ in range(2):
                result = await engine.run(graph, ["bake"])
                for name, image in expected.items():
                    torch.testing.assert_close(
                        torch.as_tensor(result.outputs["bake"][name].resolve()),
                        image,
                        rtol=0,
                        atol=0,
                    )
        finally:
            await composer.close()

    asyncio.run(scenario())
