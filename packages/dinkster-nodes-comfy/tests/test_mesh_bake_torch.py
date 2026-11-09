"""Observable ambient occlusion, controls and counted mesh batches."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import MESH_TYPE, Node, NodeSchema, OutputSpec, TriangleMeshBatch, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_bake import BakeAmbientOcclusion, BakeNormalMapFromMesh

from dinkster.compose import PackSpec, ServingComposer


def plane(z: float = 0.0, extent: float = 1.0) -> TriangleMeshBatch:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [
                [
                    [-extent, -extent, z],
                    [extent, -extent, z],
                    [extent, extent, z],
                    [-extent, extent, z],
                ]
            ]
        ),
        faces=torch.tensor([[[0, 1, 2], [0, 2, 3]]]),
        uvs=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]),
    )


def bake(low: TriangleMeshBatch, high: TriangleMeshBatch, **overrides: object) -> torch.Tensor:
    parameters = dict(resolution=64, samples=16, max_distance=0.5, strength=1.0, bias=0.001)
    parameters.update(overrides)
    torch.manual_seed(749)
    return BakeAmbientOcclusion.execute(low_poly=low, high_poly=high, **parameters)["IMAGE"]


def test_ambient_occlusion_distance_strength_and_gutters() -> None:
    low = plane()
    high = plane(0.25, 4.0)
    # Keep the original surface in the high mesh so projection cannot jump to the roof.
    high.vertices = torch.cat([low.vertices, high.vertices], dim=1)
    high.faces = torch.cat([low.faces, high.faces + 4], dim=1)
    high.uvs = None
    open_surface = bake(low, low)
    torch.testing.assert_close(open_surface, torch.ones_like(open_surface), rtol=0, atol=0)
    occluded = bake(low, high)
    assert occluded.mean() < 0.15
    short = bake(low, high, max_distance=0.01)
    torch.testing.assert_close(short, torch.ones_like(short), rtol=0, atol=0)
    half = bake(low, high, strength=0.5)
    torch.testing.assert_close(half, (occluded + 1.0) / 2.0, rtol=0, atol=0)
    low.uvs = low.uvs * 0.4 + 0.3
    gutters = bake(low, high)
    assert gutters[0, 0, 0].mean() < 0.15
    assert gutters.shape == (1, 64, 64, 3) and gutters.device.type == "cpu"
    lifted = bake(low, high, bias=0.1)
    torch.testing.assert_close(lifted, torch.ones_like(lifted), rtol=0, atol=0)
    low.normals = torch.tensor([[[0.0, 0.0, -1.0]] * 4])
    downward = bake(low, high)
    torch.testing.assert_close(downward, torch.ones_like(downward), rtol=0, atol=0)


def test_ambient_occlusion_ragged_empty_and_shared_high_mesh() -> None:
    low = plane()
    low.vertices = low.vertices.repeat(2, 1, 1)
    low.faces = low.faces.repeat(2, 1, 1)
    low.uvs = low.uvs.repeat(2, 1, 1)
    low.vertex_counts = torch.tensor([4, 3])
    low.face_counts = torch.tensor([2, 0])
    result = bake(low, plane())
    assert result.shape == (2, 64, 64, 3)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    low.uvs = None
    with pytest.raises(ValueError, match="low_poly has no UVs"):
        bake(low, plane())


class MeshSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ambient-occlusion-source.make",
            display_name="Mesh source",
            inputs=(),
            outputs=(
                OutputSpec("low", TypeExpr.concrete(MESH_TYPE)),
                OutputSpec("high", TypeExpr.concrete(MESH_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict:
        low, high = plane(), plane(0.25, 4.0)
        high.vertices = torch.cat([low.vertices, high.vertices], dim=1)
        high.faces = torch.cat([low.faces, high.faces + 4], dim=1)
        high.uvs = None
        return cls.outputs(low=low, high=high)


def test_normal_bake_tangent_handedness_fallback_and_counted_batches() -> None:
    low, high = plane(), plane(2.0)
    high.normals = torch.tensor([[[0.0, 0.6, 0.8]] * 4])
    low.uvs = low.uvs * 0.4 + 0.2
    # The high plane is outside the cage: nearest-surface fallback still retains +Y.
    image = BakeNormalMapFromMesh.execute(low_poly=low, high_poly=high, resolution=64)["IMAGE"]
    expected = torch.tensor([0.5, 0.8, 0.9]).expand(1, 64, 64, 3)
    torch.testing.assert_close(image, expected)
    low.uvs[..., 1] = 1.0 - low.uvs[..., 1]
    mirrored = BakeNormalMapFromMesh.execute(low_poly=low, high_poly=high, resolution=64)["IMAGE"]
    torch.testing.assert_close(mirrored, torch.tensor([0.5, 0.2, 0.9]).expand_as(mirrored))
    low.vertices = low.vertices.repeat(3, 1, 1)
    low.faces = low.faces.repeat(3, 1, 1)
    low.uvs = low.uvs.repeat(3, 1, 1)
    low.vertex_counts, low.face_counts = torch.tensor([4, 4, 3]), torch.tensor([2, 2, 0])
    result = BakeNormalMapFromMesh.execute(low_poly=low, high_poly=high, resolution=64)["IMAGE"]
    torch.testing.assert_close(result[:2], mirrored.expand(2, -1, -1, -1))
    torch.testing.assert_close(result[2], torch.full((64, 64, 3), 0.5))
    low.uvs = None
    with pytest.raises(ValueError, match="low_poly has no UVs"):
        BakeNormalMapFromMesh.execute(low_poly=low, high_poly=high, resolution=64)


def test_normal_bake_cage_and_backfaces() -> None:
    low, high = plane(), plane()
    low.normals = torch.tensor([[[0.0, 0.0, 1.0]] * 4])
    roof = plane(0.15)
    high.vertices = torch.cat([high.vertices, roof.vertices], dim=1)
    high.faces = torch.cat([high.faces, roof.faces.flip(-1) + 4], dim=1)
    high.normals = torch.tensor([[[0.0, 0.0, 1.0]] * 4 + [[0.6, 0.0, 0.8]] * 4])

    def run(cage: float, cull: bool) -> torch.Tensor:
        return BakeNormalMapFromMesh.execute(
            low_poly=low,
            high_poly=high,
            resolution=64,
            cage_distance=cage,
            ignore_backfaces=cull,
        )["IMAGE"]

    front = run(0.1, True)
    torch.testing.assert_close(front, torch.tensor([0.5, 0.5, 1.0]).expand_as(front))
    both = run(0.1, False)
    torch.testing.assert_close(both, torch.tensor([0.8, 0.5, 0.9]).expand_as(both))
    narrow = run(0.001, False)
    torch.testing.assert_close(narrow, front)
    high.normals[..., 2] = -1.0
    safe = run(0.1, False)
    torch.testing.assert_close(safe, front)


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("node_type", ["BakeAmbientOcclusion", "BakeNormalMapFromMesh"])
def test_ambient_occlusion_ordinary_hosts(tmp_path: Path, in_process: bool, node_type: str) -> None:
    module = f"mesh_bake_source_{node_type}_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from test_mesh_bake_torch import MeshSource\n"
        "from dinkster_nodes_comfy import register_inference_types\nNODES=(MeshSource,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="ambient-occlusion-source"\npublisher="dinkster"\n'
        'namespaces=["ambient-occlusion-source"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
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
                    "source": GraphNode("ambient-occlusion-source.make"),
                    "bake": GraphNode(
                        node_type,
                        {
                            "low_poly": Link("source", "low"),
                            "high_poly": Link("source", "high"),
                            "resolution": 64,
                            **(
                                dict(samples=16, max_distance=0.5, strength=0.5, bias=0.001)
                                if node_type == "BakeAmbientOcclusion"
                                else dict(cage_distance=0.05, ignore_backfaces=True)
                            ),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["bake"])
                image = torch.as_tensor(result.outputs["bake"]["IMAGE"].resolve())
                assert image.shape == (1, 64, 64, 3) and image.dtype == torch.float32
                if node_type == "BakeAmbientOcclusion":
                    assert 0.5 <= image.mean() < 0.6
                    torch.testing.assert_close(image[..., 0], image[..., 2], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(
                        image, torch.tensor([0.5, 0.5, 1.0]).expand_as(image)
                    )
        finally:
            await composer.close()

    asyncio.run(scenario())
