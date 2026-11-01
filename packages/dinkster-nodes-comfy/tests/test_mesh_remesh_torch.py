"""Mesh re-extraction retains field choices, colors and ragged batch boundaries."""

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
from dinkster_nodes_comfy.mesh_remesh import RemeshMesh

from dinkster.compose import PackSpec, ServingComposer


def source(*, open_top: bool = False) -> Any:
    vertices = torch.tensor(
        [[x * 0.31, y * 0.22, z * 0.38] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
            [4, 5, 6],
            [5, 7, 6],
        ],
        dtype=torch.int32,
    )
    if open_top:
        faces = faces[:-2]
    return TriangleMeshBatch(
        vertices=vertices[None],
        faces=faces[None],
        vertex_colors=(vertices + 0.5)[None].float(),
        uvs=torch.zeros(1, 8, 2),
        normals=torch.ones(1, 8, 3),
        texture=torch.full((1, 2, 3, 3), 0.3),
        material={"doubleSided": True},
    )


def options() -> dict[str, Any]:
    return dict(
        resolution=32,
        band=1.0,
        project_back=0.0,
        fix_poles=False,
        smooth_iters=0,
        drop_small_components=0.0,
        precluster_max_verts=0,
    )


@pytest.mark.parametrize(
    "mode,controls",
    [
        ("udf", {}),
        ("sdf", {}),
        ("sdf", {"sign_mode.manifold": True}),
        ("solid", {"sign_mode.fill": 0.0}),
    ],
)
def test_fields_retain_colors_and_leave_input_unchanged(
    mode: str, controls: dict[str, Any]
) -> None:
    mesh = source()
    saved = mesh.vertices.clone()
    result = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode=mode, **options(), **controls)["mesh"]
    )
    assert result.vertices.dtype == torch.float32 and result.faces.dtype == torch.int32
    assert result.vertices.device.type == "cpu"
    assert result.face_counts[0] > 12 and result.vertex_counts[0] > 8
    assert torch.isfinite(result.vertices).all() and torch.isfinite(result.vertex_colors).all()
    assert int(result.faces.max()) < int(result.vertex_counts[0])
    assert result.vertex_colors.amin() >= 0 and result.vertex_colors.amax() <= 1
    assert result.material == mesh.material
    torch.testing.assert_close(result.texture, mesh.texture, rtol=0, atol=0)
    assert result.uvs is None and result.normals is None and result.tangents is None
    torch.testing.assert_close(mesh.vertices, saved, rtol=0, atol=0)
    assert mesh.uvs is not None and mesh.normals is not None


def test_projected_vertices_retain_interpolated_colors() -> None:
    args = options()
    args["project_back"] = 1.0
    result = materialize_mesh(RemeshMesh.execute(mesh=source(), sign_mode="udf", **args)["mesh"])
    torch.testing.assert_close(result.vertex_colors, result.vertices + 0.5, rtol=0, atol=1e-6)
    distance_to_planes = (result.vertices.abs() - torch.tensor([0.31, 0.22, 0.38])).abs()
    assert float(distance_to_planes.amin(-1).amax()) < 1e-6


def test_solid_fill_changes_openings_and_ragged_batches() -> None:
    mesh = source(open_top=True)
    filled = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode="solid", **options(), **{"sign_mode.fill": 0.0})[
            "mesh"
        ]
    )
    open_mesh = materialize_mesh(
        RemeshMesh.execute(mesh=mesh, sign_mode="solid", **options(), **{"sign_mode.fill": 1000.0})[
            "mesh"
        ]
    )
    assert filled.face_counts.tolist() != open_mesh.face_counts.tolist()
    full = source()
    full.vertices = full.vertices.repeat(2, 1, 1)
    full.faces = full.faces.repeat(2, 1, 1)
    full.vertex_colors = full.vertex_colors.repeat(2, 1, 1)
    full.vertex_counts = torch.tensor([8, 8])
    full.face_counts = torch.tensor([12, 10])
    full.texture = full.uvs = full.normals = None
    batched = materialize_mesh(
        RemeshMesh.execute(mesh=full, sign_mode="solid", **options(), **{"sign_mode.fill": 0.0})[
            "mesh"
        ]
    )
    nv, nf = int(batched.vertex_counts[1]), int(batched.face_counts[1])
    torch.testing.assert_close(batched.vertices[1, :nv], filled.vertices[0], rtol=0, atol=0)
    torch.testing.assert_close(batched.faces[1, :nf], filled.faces[0], rtol=0, atol=0)
    assert (batched.vertices[1, nv:] == 0).all() and (batched.faces[1, nf:] == 0).all()


class RemeshSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "remesh-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_remesh_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "remesh_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_remesh_torch import RemeshSource\nNODES=(RemeshSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="remesh-fixture"\npublisher="dinkster"\n'
            'namespaces=["remesh-fixture"]\n[pack.entry]\n'
            'nodes="remesh_fixture:NODES"\ntypes="remesh_fixture:register_inference_types"\n',
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
                    "source": GraphNode("remesh-fixture.source"),
                    "remesh": GraphNode(
                        "RemeshMesh",
                        {"mesh": Link("source", "mesh"), **options()},
                        slot_variants={"sign_mode": "sdf"},
                    ),
                }
            )
            expected = materialize_mesh(
                RemeshMesh.execute(mesh=source(), sign_mode="sdf", **options())["mesh"]
            )
            for _ in range(2):
                result = await engine.run(graph, ["remesh"])
                value = materialize_mesh(result.outputs["remesh"]["mesh"].resolve())
                torch.testing.assert_close(value.vertices, expected.vertices, rtol=0, atol=0)
                torch.testing.assert_close(value.faces, expected.faces, rtol=0, atol=0)
                torch.testing.assert_close(
                    value.vertex_colors, expected.vertex_colors, rtol=0, atol=0
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
