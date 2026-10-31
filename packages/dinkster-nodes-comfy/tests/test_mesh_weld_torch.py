"""Vertex welding preserves per-item topology, averaging and material contracts."""

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
    materialize_mesh,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_weld import WeldVertices

from dinkster.compose import PackSpec, ServingComposer


def source() -> Any:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [
                [[0, 0, 0], [0.1, 0, 0], [2, 0, 0], [0, 2, 0]],
                [[5, 0, 0], [5, 0, 0], [99, 99, 99], [99, 99, 99]],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[[0, 2, 3], [1, 2, 3]], [[0, 1, 0], [3, 3, 3]]]),
        vertex_colors=torch.tensor(
            [
                [[1, 0, 0, 0.2], [0, 0, 1, 0.8], [0, 1, 0, 1], [1, 1, 1, 1]],
                [[0, 0.2, 0, 1], [0, 0.6, 0, 1], [9, 9, 9, 9], [9, 9, 9, 9]],
            ]
        ),
        vertex_counts=torch.tensor([4, 2]),
        face_counts=torch.tensor([2, 1]),
        uvs=torch.ones(2, 4, 2),
        normals=torch.ones(2, 4, 3),
        tangents=torch.ones(2, 4, 4),
        texture=torch.ones(1, 2, 2, 3),
        unlit=True,
        material={"doubleSided": True},
    )


def assert_welded(value: Any) -> None:
    value = materialize_mesh(value)
    torch.testing.assert_close(
        value.vertices,
        torch.tensor(
            [
                [[0.05, 0, 0], [0, 2, 0], [2, 0, 0]],
                [[5, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        value.faces,
        torch.tensor(
            [
                [[0, 2, 1], [0, 2, 1]],
                [[0, 0, 0], [0, 0, 0]],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(value.vertex_counts, torch.tensor([3, 1]), rtol=0, atol=0)
    torch.testing.assert_close(value.face_counts, torch.tensor([2, 1]), rtol=0, atol=0)
    torch.testing.assert_close(
        value.vertex_colors[0],
        torch.tensor(
            [
                [0.5, 0, 0.5, 0.5],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
            ]
        ),
        rtol=0,
        atol=0,
    )
    assert value.uvs is None and value.normals is None and value.tangents is None
    assert value.unlit and value.material == {"doubleSided": True}
    torch.testing.assert_close(value.texture, torch.ones(1, 2, 2, 3), rtol=0, atol=0)


def test_weld_absolute_override_and_padded_batch() -> None:
    mesh = source()
    out = WeldVertices.execute(mesh=mesh, epsilon_rel=100, epsilon_abs=0.5)["mesh"]
    assert_welded(out)
    assert mesh.uvs is not None and mesh.vertices.shape == (2, 4, 3)
    torch.testing.assert_close(mesh.vertices[0, 1], torch.tensor([0.1, 0, 0]), rtol=0, atol=0)


@pytest.mark.parametrize("relative", [0.0, 0.1])
def test_weld_relative_and_empty_faces(relative: float) -> None:
    mesh = TriangleMeshBatch(
        torch.tensor([[[0, 0, 0], [0.1, 0, 0], [2, 0, 0]]], dtype=torch.float32),
        torch.empty((1, 0, 3), dtype=torch.int64),
    )
    out = WeldVertices.execute(mesh=mesh, epsilon_rel=relative, epsilon_abs=0.0)["mesh"]
    expected = torch.tensor([[[0.05, 0, 0], [2, 0, 0]]]) if relative else mesh.vertices
    torch.testing.assert_close(out.vertices, expected, rtol=0, atol=0)
    assert out.faces.shape == (1, 0, 3) and out.vertex_colors is None


class WeldSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "weld-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_weld_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "weld_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_weld_torch import WeldSource\nNODES=(WeldSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="weld-fixture"\npublisher="dinkster"\nnamespaces=["weld-fixture"]\n'
            '[pack.entry]\nnodes="weld_fixture:NODES"\ntypes="weld_fixture:register_inference_types"\n',
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
                    "source": GraphNode("weld-fixture.source"),
                    "weld": GraphNode(
                        "WeldVertices",
                        {"mesh": Link("source", "mesh"), "epsilon_rel": 100.0, "epsilon_abs": 0.5},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["weld"])
                assert_welded(result.outputs["weld"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
