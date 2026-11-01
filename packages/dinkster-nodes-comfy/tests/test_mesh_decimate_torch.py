"""Mesh simplification keeps per-item limits, colors and source values intact."""

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
from dinkster_nodes_comfy.mesh_decimate import DecimateMesh

from dinkster.compose import PackSpec, ServingComposer


def source() -> Any:
    size = 8
    vertices = torch.tensor(
        [[x * 0.7, y * 1.2, 0.1 * ((x * y) % 3)] for y in range(size) for x in range(size)],
        dtype=torch.float32,
    )
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend([[a, a + 1, a + size], [a + 1, a + size + 1, a + size]])
    return TriangleMeshBatch(
        vertices=vertices[None],
        faces=torch.tensor([faces]),
        vertex_colors=torch.tensor([0.2, 0.7, 0.4, 0.9]).expand(1, len(vertices), 4).clone(),
        normals=torch.ones(1, len(vertices), 3),
        uvs=torch.ones(1, len(vertices), 2),
        tangents=torch.ones(1, len(vertices), 4),
        texture=torch.full((1, 2, 3, 3), 0.3),
        material={"doubleSided": True},
        unlit=True,
    )


@pytest.mark.parametrize("target", [0, 98, 99])
def test_disabled_and_satisfied_target_preserve_geometry(target: int) -> None:
    original = source()
    result = materialize_mesh(
        DecimateMesh.execute(mesh=original, target_face_count=target, placement_mode="midpoint")[
            "mesh"
        ]
    )
    torch.testing.assert_close(result.vertices, original.vertices, rtol=0, atol=0)
    torch.testing.assert_close(result.faces, original.faces, rtol=0, atol=0)
    torch.testing.assert_close(result.vertex_colors, original.vertex_colors, rtol=0, atol=0)
    assert result.face_counts.tolist() == [98]
    assert result.uvs is None and result.normals is None and result.tangents is None
    assert original.normals is not None and original.uvs is not None


@pytest.mark.parametrize("mode", ["midpoint", "qem"])
def test_reduction_preserves_colors_materials_and_input(mode: str) -> None:
    original = source()
    saved_vertices = original.vertices.clone()
    saved_faces = original.faces.clone()
    outputs = [
        materialize_mesh(
            DecimateMesh.execute(mesh=original, target_face_count=60, placement_mode=mode)["mesh"]
        )
        for _ in range(2)
    ]
    first, second = outputs
    assert first.face_counts.tolist() == [60]
    assert first.vertex_counts[0] < original.vertices.shape[1]
    assert torch.isfinite(first.vertices).all()
    assert first.faces.min() >= 0 and first.faces.max() < first.vertex_counts[0]
    torch.testing.assert_close(first.vertices, second.vertices, rtol=0, atol=0)
    torch.testing.assert_close(first.faces, second.faces, rtol=0, atol=0)
    expected_colors = torch.tensor([0.2, 0.7, 0.4, 0.9]).expand_as(first.vertex_colors)
    torch.testing.assert_close(first.vertex_colors, expected_colors)
    torch.testing.assert_close(first.texture, original.texture, rtol=0, atol=0)
    assert first.material == {"doubleSided": True} and first.unlit
    assert first.vertices.dtype == original.vertices.dtype
    assert first.faces.dtype == original.faces.dtype
    assert first.vertices.device == original.vertices.device
    assert first.uvs is None and first.normals is None and first.tangents is None
    torch.testing.assert_close(original.vertices, saved_vertices, rtol=0, atol=0)
    torch.testing.assert_close(original.faces, saved_faces, rtol=0, atol=0)


def test_padded_counts_and_nonconstant_colors() -> None:
    original = source()
    original.vertices = original.vertices.repeat(2, 1, 1).double()
    original.faces = original.faces.repeat(2, 1, 1).to(torch.int32)
    original.faces[1, 0] = torch.tensor([0, 1, 2], dtype=torch.int32)
    original.vertex_colors = (original.vertices / 10).float()
    original.uvs = original.uvs.repeat(2, 1, 1)
    original.normals = original.normals.repeat(2, 1, 1)
    original.tangents = original.tangents.repeat(2, 1, 1)
    original.vertex_counts = torch.tensor([64, 3])
    original.face_counts = torch.tensor([98, 1])
    result = materialize_mesh(
        DecimateMesh.execute(mesh=original, target_face_count=60, placement_mode="midpoint")["mesh"]
    )
    assert result.face_counts.tolist() == [60, 1]
    assert result.vertex_counts[1] == 3
    nv = int(result.vertex_counts[0])
    torch.testing.assert_close(result.vertex_colors[0, :nv], (result.vertices[0, :nv] / 10).float())
    torch.testing.assert_close(result.vertices[1, :3], original.vertices[1, :3], rtol=0, atol=0)
    torch.testing.assert_close(result.faces[1, 0], original.faces[1, 0], rtol=0, atol=0)
    assert (result.vertices[1, 3:] == 0).all() and (result.faces[1, 1:] == 0).all()
    assert result.vertices.dtype == torch.float64 and result.faces.dtype == torch.int32


def feature_controls() -> dict[str, object]:
    return {
        "placement_mode.line_quadric_weight": 1.0,
        "placement_mode.feature_edge_quadric_weight": 10.0,
        "placement_mode.feature_edge_min_dihedral_deg": 5.0,
        "placement_mode.clamp_v_to_edge": False,
    }


def assert_feature_result(value: object) -> None:
    result = materialize_mesh(value)
    assert result.face_counts.tolist() == [60]
    assert result.vertex_counts.tolist() == [40]
    assert torch.isfinite(result.vertices).all()
    expected = torch.tensor([0.2, 0.7, 0.4, 0.9]).expand_as(result.vertex_colors)
    torch.testing.assert_close(result.vertex_colors, expected)
    assert result.material == {"doubleSided": True} and result.unlit
    assert result.normals is None and result.uvs is None and result.tangents is None


class DecimateSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "decimate-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_decimate_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "decimate_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_mesh_decimate_torch import DecimateSource\nNODES=(DecimateSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="decimate-fixture"\npublisher="dinkster"\n'
            'namespaces=["decimate-fixture"]\n[pack.entry]\n'
            'nodes="decimate_fixture:NODES"\ntypes="decimate_fixture:register_inference_types"\n',
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
                    "source": GraphNode("decimate-fixture.source"),
                    "decimate": GraphNode(
                        "DecimateMesh",
                        {
                            "mesh": Link("source", "mesh"),
                            "target_face_count": 60,
                            **feature_controls(),
                        },
                        slot_variants={"placement_mode": "qem"},
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["decimate"])
                assert_feature_result(result.outputs["decimate"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
