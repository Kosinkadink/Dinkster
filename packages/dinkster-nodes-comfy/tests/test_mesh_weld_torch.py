"""Vertex welding preserves per-item topology, averaging and material contracts."""

from __future__ import annotations

import asyncio
import math
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
from dinkster_nodes_comfy.mesh_holes import FillHoles
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


def hole_source(count: int = 4) -> Any:
    ring = (
        [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]]
        if count == 4
        else [
            [2 * math.cos(i * 2 * math.pi / count), math.sin(i * 2 * math.pi / count), 0]
            for i in range(count)
        ]
    )
    vertices = torch.tensor([ring + [[0.5, 0.25, 2]]], dtype=torch.float32)
    return TriangleMeshBatch(
        vertices=vertices,
        faces=torch.tensor([[[count, i, (i + 1) % count] for i in range(count)]]),
        vertex_colors=torch.arange((count + 1) * 4, dtype=torch.float32).reshape(1, count + 1, 4)
        / 40,
        uvs=torch.ones(1, count + 1, 2),
        normals=torch.ones_like(vertices),
        tangents=torch.ones(1, count + 1, 4),
        texture=torch.full((1, 1, 2, 3), 0.3),
        material={"doubleSided": True},
        unlit=True,
    )


def fill_hole(mesh: object, **options: Any) -> Any:
    inputs = dict(max_perimeter=100.0, weld_epsilon_rel=0.0, max_vertices=16, fill_chains=False)
    inputs.update(options)
    return materialize_mesh(FillHoles.execute(mesh=mesh, **inputs)["mesh"])


def assert_filled(value: object) -> None:
    mesh = materialize_mesh(value)
    source_mesh = hole_source()
    torch.testing.assert_close(mesh.vertices, source_mesh.vertices, rtol=0, atol=0)
    torch.testing.assert_close(mesh.faces[0, :4], source_mesh.faces[0], rtol=0, atol=0)
    torch.testing.assert_close(
        mesh.faces[0, 4:], torch.tensor([[2, 1, 0], [3, 2, 0]]), rtol=0, atol=0
    )
    torch.testing.assert_close(mesh.vertex_colors, source_mesh.vertex_colors, rtol=0, atol=0)
    assert mesh.vertex_counts.tolist() == [5] and mesh.face_counts.tolist() == [6]
    assert mesh.uvs is None and mesh.normals is None and mesh.tangents is None
    assert mesh.material == {"doubleSided": True} and mesh.unlit
    torch.testing.assert_close(mesh.texture, source_mesh.texture, rtol=0, atol=0)


def test_fill_holes_perimeter_vertex_limits_and_source_preservation() -> None:
    source_mesh = hole_source()
    assert_filled(fill_hole(source_mesh, max_perimeter=14.001))
    for options in ({"max_perimeter": 14.0}, {"max_perimeter": 0.0}, {"max_vertices": 3}):
        out = fill_hole(source_mesh, **options)
        torch.testing.assert_close(out.faces, source_mesh.faces, rtol=0, atol=0)
    assert source_mesh.uvs is not None and source_mesh.faces.shape == (1, 4, 3)


@pytest.mark.parametrize("count", [8, 9])
def test_fill_holes_vertex_and_centroid_fan_threshold(count: int) -> None:
    source_mesh = hole_source(count)
    out = fill_hole(source_mesh)
    extra = int(count > 8)
    assert out.vertex_counts.tolist() == [count + 1 + extra]
    assert out.face_counts.tolist() == [count + (count if extra else count - 2)]
    torch.testing.assert_close(out.faces[0, :count], source_mesh.faces[0], rtol=0, atol=0)
    added = out.faces[0, count:]
    assert added[:, 2].tolist() == [count + 1 if extra else 0] * added.shape[0]
    triangles = out.vertices[0, added]
    cross = torch.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    )
    assert (cross[:, 2] < 0).all()
    if extra:
        torch.testing.assert_close(out.vertices[0, -1], source_mesh.vertices[0, :count].mean(0))
        torch.testing.assert_close(
            out.vertex_colors[0, -1], source_mesh.vertex_colors[0, :count].mean(0)
        )


def test_fill_holes_open_chain_requires_opt_in() -> None:
    mesh = hole_source(9)
    mesh.faces = torch.cat([mesh.faces, mesh.faces[:, :1]], dim=1)
    unchanged = fill_hole(mesh)
    torch.testing.assert_close(unchanged.faces, mesh.faces, rtol=0, atol=0)
    out = fill_hole(mesh, fill_chains=True)
    assert out.vertex_counts.tolist() == [11] and out.face_counts.tolist() == [19]
    torch.testing.assert_close(out.faces[0, -1], torch.tensor([0, 1, 10]), rtol=0, atol=0)
    torch.testing.assert_close(out.vertices[0, -1], mesh.vertices[0, :9].mean(0))


def test_fill_holes_adaptive_weld_and_padded_empty_item() -> None:
    source_mesh = hole_source()
    vertices = source_mesh.vertices[0, source_mesh.faces[0]].reshape(-1, 3)
    faces = torch.arange(12).reshape(4, 3)
    mesh = TriangleMeshBatch(vertices=vertices.unsqueeze(0), faces=faces.unsqueeze(0))
    out = fill_hole(mesh, weld_epsilon_rel=1e-5)
    assert out.vertex_counts.tolist() == [5] and out.face_counts.tolist() == [6]
    mesh = hole_source()
    mesh.vertices = mesh.vertices.expand(2, -1, -1).clone()
    mesh.faces = mesh.faces.expand(2, -1, -1).clone()
    mesh.vertex_colors = mesh.vertex_colors.expand(2, -1, -1).clone()
    for name in ("uvs", "normals", "tangents"):
        setattr(mesh, name, getattr(mesh, name).expand(2, -1, -1).clone())
    mesh.vertex_counts, mesh.face_counts = torch.tensor([5, 2]), torch.tensor([4, 0])
    out = fill_hole(mesh)
    assert out.vertex_counts.tolist() == [5, 2] and out.face_counts.tolist() == [6, 0]
    assert out.faces[1].count_nonzero() == 0
    torch.testing.assert_close(out.vertices[1, :2], mesh.vertices[1, :2], rtol=0, atol=0)


class HoleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "hole-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=hole_source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_hole_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "hole_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_weld_torch import HoleSource\nNODES=(HoleSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="hole-fixture"\npublisher="dinkster"\nnamespaces=["hole-fixture"]\n'
            '[pack.entry]\nnodes="hole_fixture:NODES"\ntypes="hole_fixture:register_inference_types"\n',
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
                    "source": GraphNode("hole-fixture.source"),
                    "fill": GraphNode(
                        "FillHoles",
                        {
                            "mesh": Link("source", "mesh"),
                            "max_perimeter": 15.0,
                            "weld_epsilon_rel": 0.0,
                            "max_vertices": 16,
                            "fill_chains": False,
                        },
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["fill"])
                assert_filled(result.outputs["fill"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
