"""Point-map triangulation, frame conversion, batch selection and textures."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MESH_TYPE,
    MOGE_GEOMETRY_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_mesh,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moge_mesh import MoGePointMapToMesh
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


def geometry(perspective: bool, size: int = 3) -> dict[str, Any]:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size + 2), indexing="ij")
    # Spherical Z is negative here while the supplied radial depth is positive.
    points = torch.stack((x * 2 + 1, y * 3 + 2, torch.full_like(x, -7)), -1).float()
    packet: dict[str, Any] = {
        "points": torch.stack((points + 100, points)),
        "depth": torch.full((2, size, size + 2), 9.0),
        "image": torch.arange(2 * size * (size + 2) * 3).reshape(2, size, size + 2, 3).float(),
    }
    if perspective:
        packet["intrinsics"] = torch.eye(3).repeat(2, 1, 1)
    return packet


@pytest.mark.parametrize("perspective", (False, True))
@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("texture", (False, True))
def test_selected_geometry_mesh_frames_uvs_and_texture(
    perspective: bool, portable: bool, texture: bool
) -> None:
    packet = geometry(perspective)
    before = {k: v.clone() for k, v in packet.items()}
    value = decode_latent(encode_latent(packet), require_samples=False) if portable else packet
    mesh: Any = MoGePointMapToMesh.execute(
        moge_geometry=value,
        batch_index=1,
        decimation=2,
        discontinuity_threshold=0,
        texture=texture,
    )["MESH"]
    # Stride two selects a 2x3 grid; its two quads yield four triangles.
    vertices = torch.tensor(
        [[1, 2, -7], [5, 2, -7], [9, 2, -7], [1, 8, -7], [5, 8, -7], [9, 8, -7]],
        dtype=torch.float32,
    )
    expected = vertices * torch.tensor([1, -1, -1]) if perspective else vertices[:, [1, 2, 0]]
    faces = torch.tensor([[0, 1, 4], [1, 2, 5], [0, 4, 3], [1, 5, 4]])
    if perspective:
        faces = faces[:, [0, 2, 1]]
    torch.testing.assert_close(mesh.vertices, expected[None])
    torch.testing.assert_close(mesh.faces, faces[None])
    torch.testing.assert_close(
        mesh.uvs,
        torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]]),
    )
    if texture:
        torch.testing.assert_close(mesh.texture, before["image"][1:2])
    else:
        assert mesh.texture is None
    for key in before:
        torch.testing.assert_close(packet[key], before[key])


def test_spherical_radial_depth_prevents_false_discontinuity() -> None:
    packet = geometry(False)
    packet["points"][1, ..., 2] = -torch.arange(1.0, 6.0)
    # Negative spherical Z would fail the relative-depth edge check; radial
    # depth remains constant and permits every quad on the spherical mesh.
    mesh: Any = MoGePointMapToMesh.execute(
        moge_geometry=packet,
        batch_index=1,
        discontinuity_threshold=0.04,
    )["MESH"]
    assert mesh.vertices.shape == (1, 15, 3)
    assert mesh.faces.shape == (1, 16, 3)
    del packet["depth"]
    with pytest.raises(ValueError, match="MoGe produced an empty mesh"):
        MoGePointMapToMesh.execute(moge_geometry=packet, batch_index=1)


def test_missing_depth_uses_point_depth_without_texture_image() -> None:
    packet = geometry(True)
    packet["points"][..., 2] = 3
    del packet["depth"], packet["image"]
    mesh: Any = MoGePointMapToMesh.execute(moge_geometry=packet, texture=False)["MESH"]
    assert mesh.faces.shape == (1, 16, 3) and mesh.texture is None


def test_geometry_errors_and_empty_mesh() -> None:
    with pytest.raises(ValueError, match="moge_geometry has no points output"):
        MoGePointMapToMesh.execute(moge_geometry={})
    with pytest.raises(
        ValueError, match="batch_index 2 out of range; moge_geometry has batch size 2"
    ):
        MoGePointMapToMesh.execute(moge_geometry=geometry(True), batch_index=2)
    packet = geometry(True)
    packet["points"][:] = float("inf")
    with pytest.raises(ValueError, match="MoGe produced an empty mesh"):
        MoGePointMapToMesh.execute(moge_geometry=packet)


class MeshGeometrySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="moge-mesh-fixture.source",
            outputs=(OutputSpec("geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(geometry=geometry(True))


class MeshGeometryInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="moge-mesh-fixture.inspect",
            inputs=(
                InputSpec("textured", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("bare", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("source_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, textured: object, bare: object, source_geometry: Any) -> dict[str, object]:
        from dinkster_api.v1 import materialize_latent

        packet = materialize_latent(source_geometry)
        expected_vertices = torch.tensor(
            [[[1, -2, 7], [5, -2, 7], [9, -2, 7], [1, -8, 7], [5, -8, 7], [9, -8, 7]]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[[0, 4, 1], [1, 5, 2], [0, 3, 4], [1, 4, 5]]])
        expected_uvs = torch.tensor(
            [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]]
        )
        for value in (textured, bare):
            mesh = materialize_mesh(value)
            torch.testing.assert_close(mesh.vertices, expected_vertices, rtol=0, atol=0)
            torch.testing.assert_close(mesh.faces, expected_faces, rtol=0, atol=0)
            torch.testing.assert_close(mesh.uvs, expected_uvs, rtol=0, atol=0)
            assert mesh.vertices.dtype == torch.float32 and mesh.faces.dtype == torch.int64
            assert mesh.vertex_counts is None and mesh.face_counts is None
        torch.testing.assert_close(
            materialize_mesh(textured).texture,
            torch.arange(45.0, 90.0).reshape(1, 3, 5, 3),
            rtol=0,
            atol=0,
        )
        assert materialize_mesh(bare).texture is None
        for key, expected in geometry(True).items():
            torch.testing.assert_close(packet[key], expected, rtol=0, atol=0)
        return cls.outputs(summary="selected MoGe mesh retains vertices, winding, UVs and texture")


@pytest.mark.parametrize("in_process", (True, False))
def test_moge_mesh_ordinary_manifest_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"moge_mesh_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_moge_mesh_torch import MeshGeometrySource, MeshGeometryInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (MeshGeometrySource, MeshGeometryInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "moge-mesh-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["moge-mesh-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="moge-mesh-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            nodes = {"source": GraphNode("moge-mesh-fixture.source")}
            for name, texture in (("textured", True), ("bare", False)):
                nodes[name] = GraphNode(
                    "MoGePointMapToMesh",
                    {
                        "moge_geometry": Link("source", "geometry"),
                        "batch_index": 1,
                        "decimation": 2,
                        "discontinuity_threshold": 0.0,
                        "texture": texture,
                    },
                )
            nodes["inspect"] = GraphNode(
                "moge-mesh-fixture.inspect",
                {
                    "textured": Link("textured", "MESH"),
                    "bare": Link("bare", "MESH"),
                    "source_geometry": Link("source", "geometry"),
                },
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(Graph(nodes=nodes), ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == (
                    "selected MoGe mesh retains vertices, winding, UVs and texture"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
