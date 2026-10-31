"""CPU mesh painting preserves nearest colors and sparse batch association."""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    materialize_mesh,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_paint import PaintMesh

from dinkster.compose import PackSpec, ServingComposer


def sources() -> tuple[TriangleMeshBatch, dict[str, object]]:
    mesh = TriangleMeshBatch(
        torch.tensor(
            [
                [[-0.49, -0.5, -0.5], [0.49, -0.5, -0.5], [99, 99, 99]],
                [[-0.48, -0.5, -0.5], [88, 88, 88], [77, 77, 77]],
                [[0, 0, 0], [0.2, 0.1, 0.3], [66, 66, 66]],
            ],
            dtype=torch.float64,
        ),
        torch.tensor([[[0, 1, 0]], [[0, 0, 0]], [[1, 0, 1]]]),
        vertex_counts=torch.tensor([2, 1, 2]),
        face_counts=torch.tensor([1, 1, 1]),
        texture=torch.ones(3, 1, 2, 3),
        material={"roughnessFactor": 0.3},
        unlit=True,
    )
    voxel = {
        "data": torch.tensor([[1, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0]]),
        "voxel_colors": torch.tensor([[0.8, 0.2, 0.4, 9], [0.25, 0.75, 0.5, 8], [-1, 0.5, 2, 7]]),
        "resolution": 4,
    }
    return mesh, voxel


def assert_batch(value: object) -> None:
    mesh = materialize_mesh(value)
    assert mesh.vertex_counts.tolist() == [2, 1, 2]
    assert mesh.face_counts.tolist() == [1, 1, 1]
    assert mesh.vertices.shape == (3, 2, 3)
    assert mesh.faces.tolist() == [[[0, 1, 0]], [[0, 0, 0]], [[1, 0, 1]]]
    expected = torch.zeros(3, 2, 3)
    # Torch's vectorized pow differs by a float32 ULP across tensor sizes.
    expected[0] = torch.tensor([[0, 0.5, 1], [0.25, 0.75, 0.5]]).pow(2.2)
    expected[1, :1] = torch.tensor([[0.8, 0.2, 0.4]]).pow(2.2)
    torch.testing.assert_close(mesh.vertex_colors, expected, rtol=0, atol=0)
    assert mesh.texture is None and mesh.material is None and not mesh.unlit
    assert mesh.vertices[1, 1].tolist() == [0, 0, 0]


@pytest.mark.parametrize("portable", [False, True])
def test_paint_sparse_batches_use_real_counts_and_no_cross_batch_neighbors(portable: bool) -> None:
    mesh, voxel = sources()
    original = copy.deepcopy(mesh)
    registry = TypeRegistry()
    register_inference_types(registry)
    if portable:
        m = registry.spec(MESH_TYPE)
        v = registry.spec(VOXEL_TYPE)
        mesh_value, voxel_value = m.decode(m.encode(mesh)), v.decode(v.encode(voxel))
    else:
        mesh_value, voxel_value = mesh, voxel
    assert_batch(PaintMesh.execute(mesh=mesh_value, voxel_colors=voxel_value)["mesh"])
    torch.testing.assert_close(mesh.vertices, original.vertices, rtol=0, atol=0)
    assert mesh.vertex_colors is None and mesh.material == original.material
    torch.testing.assert_close(mesh.texture, original.texture, rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
def test_single_item_paint_preserves_materials_and_source(empty: bool) -> None:
    mesh, _ = sources()
    mesh.vertices = mesh.vertices[:1, :2]
    mesh.faces = mesh.faces[:1]
    mesh.vertex_counts = mesh.face_counts = None
    voxel = {
        "data": torch.empty(0, 3) if empty else torch.tensor([[0, 0, 0, 0], [0, 4, 0, 0]]),
        "voxel_colors": torch.empty(0, 3) if empty else torch.tensor([[0.5, 0, 1], [1, 0.25, 0]]),
        "resolution": 4,
    }
    result = materialize_mesh(PaintMesh.execute(mesh=mesh, voxel_colors=voxel)["mesh"])
    expected = (
        torch.zeros(1, 2, 3, dtype=torch.float64)
        if empty
        else torch.tensor([[[0.5, 0, 1], [1, 0.25, 0]]]).pow(2.2)
    )
    torch.testing.assert_close(result.vertex_colors, expected, rtol=0, atol=0)
    assert result is not mesh and mesh.vertex_colors is None
    assert result.unlit and result.material == mesh.material
    torch.testing.assert_close(result.texture, mesh.texture, rtol=0, atol=0)


class PaintSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "paint-fixture.source",
            outputs=(
                OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                OutputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        mesh, voxel = sources()
        return cls.outputs(mesh=mesh, voxel=voxel)


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_paint_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        module = f"paint_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_paint_torch import PaintSource\nNODES=(PaintSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="paint-fixture"\npublisher="dinkster"\n'
            'namespaces=["paint-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
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
            native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "source": GraphNode("paint-fixture.source"),
                    "paint": GraphNode(
                        "PaintMesh",
                        {"mesh": Link("source", "mesh"), "voxel_colors": Link("source", "voxel")},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["paint"])
                assert_batch(result.outputs["paint"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
