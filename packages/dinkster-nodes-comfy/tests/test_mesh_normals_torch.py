"""Normal smoothing respects face area, crease topology and per-vertex attributes."""

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
    TypeRegistry,
    materialize_mesh,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_normals import MeshSmoothNormals

from dinkster.compose import PackSpec, ServingComposer


def source() -> Any:
    return TriangleMeshBatch(
        vertices=torch.tensor(
            [[[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], [[4, 5, 6]] * 4],
            dtype=torch.float64,
        ),
        faces=torch.tensor([[[0, 1, 2], [1, 0, 3]], [[3, 3, 3], [3, 3, 3]]]),
        vertex_counts=torch.tensor([4, 2]),
        face_counts=torch.tensor([2, 0]),
        vertex_colors=torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32,
        uvs=torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16,
        tangents=torch.arange(32, dtype=torch.float32).reshape(2, 4, 4),
        texture=torch.full((2, 1, 1, 3), 0.2),
        metallic_roughness=torch.full((2, 1, 1, 3), 0.3),
        normal_map=torch.full((2, 1, 1, 3), 0.4),
        emissive=torch.full((2, 1, 1, 3), 0.5),
        material={"doubleSided": True},
        occlusion_in_mr=True,
        unlit=True,
    )


def assert_result(value: object, angle: float) -> None:
    result = materialize_mesh(value)
    original = source()
    hard = angle < 90
    remap = [0, 1, 2, 1, 0, 3] if hard else [0, 1, 2, 3]
    expected_vertices = original.vertices[:1, remap]
    expected_faces = torch.tensor([[[0, 1, 2], [3, 4, 5] if hard else [1, 0, 3]]])
    # Cross products have magnitudes 2 (+Z) and 6 (+Y), not equal face weights.
    shared = [0, 3 / math.sqrt(10), 1 / math.sqrt(10)]
    normals = [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3 if hard else [shared, shared, [0, 0, 1], [0, 1, 0]]
    expected_normals = torch.tensor([normals], dtype=torch.float32).to(torch.float64)
    if angle >= 180:
        expected_vertices = original.vertices
        expected_faces = original.faces
        expected_normals = torch.cat((expected_normals, torch.zeros_like(expected_normals)))
    torch.testing.assert_close(result.vertices, expected_vertices, rtol=0, atol=0)
    torch.testing.assert_close(result.faces, expected_faces, rtol=0, atol=0)
    torch.testing.assert_close(result.normals, expected_normals, rtol=0, atol=0)
    for name in ("vertex_colors", "uvs", "tangents"):
        expected = getattr(original, name) if angle >= 180 else getattr(original, name)[:1, remap]
        torch.testing.assert_close(getattr(result, name), expected, rtol=0, atol=0)
    for name in ("texture", "metallic_roughness", "normal_map", "emissive"):
        torch.testing.assert_close(getattr(result, name), getattr(original, name), rtol=0, atol=0)
    assert result.material == original.material and result.unlit and result.occlusion_in_mr
    assert result.vertex_counts.tolist() == ([4, 2] if angle >= 180 else [len(remap)])
    assert result.face_counts.tolist() == ([2, 0] if angle >= 180 else [2])


@pytest.mark.parametrize("angle,portable", [(180.0, False), (89.0, True), (91.0, False)])
def test_smooth_area_and_crease_split_preserve_attributes(angle: float, portable: bool) -> None:
    mesh = source()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(mesh)) if portable else mesh
    assert_result(MeshSmoothNormals.execute(mesh=value, crease_angle=angle)["mesh"], angle)
    assert mesh.normals is None and mesh.vertices.shape == (2, 4, 3)


def test_coincident_position_welding_and_empty_mesh() -> None:
    mesh = TriangleMeshBatch(
        source().vertices[:1, [0, 1, 2, 1, 0, 3]],
        torch.tensor([[[0, 1, 2], [3, 4, 5]]]),
    )
    result = MeshSmoothNormals.execute(mesh=mesh, crease_angle=180.0)["mesh"]
    shared = torch.tensor([0, 3 / math.sqrt(10), 1 / math.sqrt(10)], dtype=torch.float32).double()
    torch.testing.assert_close(result.normals[0, [0, 1, 3, 4]], shared.expand(4, 3), rtol=0, atol=0)
    empty = TriangleMeshBatch(torch.empty(1, 0, 3), torch.empty(1, 0, 3, dtype=torch.int64))
    assert MeshSmoothNormals.execute(mesh=empty, crease_angle=0.0)["mesh"] is empty
    assert MeshSmoothNormals.execute(mesh=empty, crease_angle=180.0)["mesh"].normals.shape == (
        1,
        0,
        3,
    )


class NormalsSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "normals-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_normals_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "normals_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_normals_torch import NormalsSource\nNODES=(NormalsSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="normals-fixture"\npublisher="dinkster"\nnamespaces=["normals-fixture"]\n'
            '[pack.entry]\nnodes="normals_fixture:NODES"\ntypes="normals_fixture:register_inference_types"\n',
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
                    "source": GraphNode("normals-fixture.source"),
                    "normals": GraphNode(
                        "MeshSmoothNormals", {"mesh": Link("source", "mesh"), "crease_angle": 89.0}
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["normals"])
                assert_result(result.outputs["normals"]["mesh"].resolve(), 89.0)
        finally:
            await composer.close()

    asyncio.run(scenario())
