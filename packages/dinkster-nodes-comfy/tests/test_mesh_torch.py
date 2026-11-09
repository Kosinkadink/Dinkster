"""CPU geometry consumers preserve complete portable mesh and voxel values."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    MESH_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    TypeRegistry,
    materialize_mesh,
    materialize_voxel,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hunyuan3d import VAEDecodeHunyuan3D
from dinkster_nodes_comfy.mesh import GetMeshInfo, MergeMeshes, RotateMesh
from dinkster_nodes_comfy.voxel_mesh import VoxelToMesh, VoxelToMeshBasic

from dinkster.compose import PackSpec, ServingComposer


def mesh() -> TriangleMeshBatch[Any]:
    return TriangleMeshBatch(
        torch.tensor([[[1.0, 2.0, 3.0], [-2.0, 5.0, 7.0]]], dtype=torch.float64),
        torch.tensor([[[0, 1, 0]]]),
        normals=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        tangents=torch.tensor([[[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0]]], dtype=torch.float64),
        texture=torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        material={"roughnessFactor": 0.3, "baseColorFactor": [0.2, 0.4, 0.6, 1.0]},
        unlit=True,
        occlusion_in_mr=True,
    )


@pytest.mark.parametrize("portable", [False, True])
def test_rotation_preserves_materials_normals_tangent_handedness_and_source(portable: bool) -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    value = spec.decode(spec.encode(source)) if portable else source
    result = RotateMesh.execute(
        mesh=value,
        mode="euler_xyz",
        **{"mode.angle_x": 0.0, "mode.angle_y": 0.0, "mode.angle_z": 90.0},
    )["mesh"]
    assert isinstance(result, TriangleMeshBatch)
    expected = torch.tensor([[[-2.0, 1.0, 3.0], [-5.0, -2.0, 7.0]]], dtype=torch.float64)
    torch.testing.assert_close(result.vertices, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        result.normals,
        torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(result.tangents[..., 3], source.tangents[..., 3], rtol=0, atol=0)
    torch.testing.assert_close(source.vertices, mesh().vertices, rtol=0, atol=0)
    assert result.material == source.material
    assert result.unlit and result.occlusion_in_mr
    torch.testing.assert_close(result.texture, source.texture, rtol=0, atol=0)
    quaternion = RotateMesh.execute(
        mesh=value,
        mode="quaternion",
        **{"mode.qw": 3.0, "mode.qx": 0.0, "mode.qy": 0.0, "mode.qz": 3.0},
    )["mesh"]
    assert isinstance(quaternion, TriangleMeshBatch)
    torch.testing.assert_close(quaternion.vertices, expected, rtol=0, atol=4e-15)
    assert materialize_mesh(source) is source
    assert (
        RotateMesh.execute(
            mesh=source,
            mode="euler_xyz",
            **dict.fromkeys(("mode.angle_x", "mode.angle_y", "mode.angle_z"), 0.0),
        )["mesh"]
        is source
    )
    with pytest.raises(ValueError, match="zero magnitude"):
        RotateMesh.execute(
            mesh=source,
            mode="quaternion",
            **dict.fromkeys(("mode.qw", "mode.qx", "mode.qy", "mode.qz"), 0.0),
        )


def test_materialization_retains_every_mesh_field_and_voxel_layout() -> None:
    source = mesh()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    restored = materialize_mesh(spec.decode(spec.encode(source)))
    for field in fields(source):
        left, right = getattr(restored, field.name), getattr(source, field.name)
        if isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    voxel = {
        "data": torch.tensor([[1, 3, 7], [9, 5, 2]], dtype=torch.int32),
        "voxel_colors": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "resolution": 64,
    }
    live = materialize_voxel(voxel)
    assert live.data is voxel["data"] and live.voxel_colors is voxel["voxel_colors"]
    spec = registry.spec(VOXEL_TYPE)
    restored_voxel = materialize_voxel(spec.decode(spec.encode(voxel)))
    torch.testing.assert_close(restored_voxel.data, voxel["data"], rtol=0, atol=0)
    torch.testing.assert_close(restored_voxel.voxel_colors, voxel["voxel_colors"], rtol=0, atol=0)
    assert restored_voxel.resolution == 64


def test_mesh_info_uses_real_counts_not_padding() -> None:
    value = TriangleMeshBatch(
        torch.zeros(2, 10000, 3),
        torch.zeros(2, 9, 3, dtype=torch.int64),
        vertex_counts=torch.tensor([10000, 2]),
        face_counts=torch.tensor([9, 1]),
        texture=torch.zeros(2, 3, 5, 3),
    )
    result = GetMeshInfo.execute(mesh=value)
    assert result["MESH"] is value
    assert result["STRING"] == (
        "Batch:      2 meshes\nVertices:   10,002 (10.0K) total\nFaces:      10 total\n"
        "  [0]      10,000 verts  \u00b7           9 faces\n"
        "  [1]           2 verts  \u00b7           1 faces\nAttributes: texture 3\u00d75"
    )
    assert math.isfinite(value.vertices.sum().item())


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("attributes", [False, True])
def test_merge_first_items_excludes_padding_offsets_faces_and_fills_attributes(
    portable: bool, attributes: bool
) -> None:
    first = TriangleMeshBatch(
        torch.arange(24, dtype=torch.float64).reshape(2, 4, 3),
        torch.tensor([[[2, 0, 1], [99, 99, 99]], [[3, 2, 1], [1, 0, 2]]]),
        vertex_counts=torch.tensor([3, 4]),
        face_counts=torch.tensor([1, 2]),
        vertex_colors=torch.full((2, 4, 3), 0.25) if attributes else None,
        normals=torch.ones(2, 4, 3),
        material={"roughnessFactor": 0.4},
    )
    second = TriangleMeshBatch(
        torch.tensor([[[21.0, 22.0, 23.0], [31.0, 32.0, 33.0]]]),
        torch.tensor([[[1, 0, 1]]]),
        uvs=torch.tensor([[[0.2, 0.3], [0.7, 0.8]]]) if attributes else None,
        vertex_colors=torch.full((1, 2, 4), 0.75) if attributes else None,
        texture=torch.full((1, 2, 3, 3), 0.5) if attributes else None,
    )
    third = TriangleMeshBatch(
        torch.tensor([[[41.0, 42.0, 43.0]]]),
        torch.tensor([[[0, 0, 0]]]),
        texture=torch.ones(1, 1, 1, 3) if attributes else None,
    )
    sources = (first, second, third)
    snapshots = [(value.vertices.clone(), value.faces.clone()) for value in sources]
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(MESH_TYPE)
    values = [spec.decode(spec.encode(value)) if portable else value for value in sources]
    result = materialize_mesh(
        MergeMeshes.execute(meshes=dict(zip(("mesh10", "mesh2", "mesh7"), values, strict=True)))[
            "mesh"
        ]
    )
    torch.testing.assert_close(
        result.vertices,
        torch.tensor(
            [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [21, 22, 23], [31, 32, 33], [41, 42, 43]]],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=0,
    )
    assert result.faces.tolist() == [[[2, 0, 1], [4, 3, 4], [5, 5, 5]]]
    assert result.vertices.device.type == "cpu" and result.faces.device.type == "cpu"
    assert result.vertex_counts is None and result.face_counts is None
    assert result.normals is None and result.material is None and not result.unlit
    if attributes:
        torch.testing.assert_close(
            result.uvs,
            torch.tensor([[[0, 0], [0, 0], [0, 0], [0.2, 0.3], [0.7, 0.8], [0, 0]]]).double(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.vertex_colors,
            torch.tensor([[[0.25, 0.25, 0.25, 1]] * 3 + [[0.75] * 4] * 2 + [[1] * 4]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(result.texture, torch.full((1, 2, 3, 3), 0.5), rtol=0, atol=0)
    else:
        assert result.uvs is None and result.vertex_colors is None and result.texture is None
    for value, (vertices, faces) in zip(sources, snapshots, strict=True):
        torch.testing.assert_close(value.vertices, vertices, rtol=0, atol=0)
        torch.testing.assert_close(value.faces, faces, rtol=0, atol=0)
    with pytest.raises(ValueError, match="MergeMeshes: need at least one mesh"):
        MergeMeshes.execute(meshes={})


@pytest.mark.parametrize("portable", [False, True])
def test_voxel_basic_threshold_orientation_and_padded_batch(portable: bool) -> None:
    data = torch.zeros(2, 2, 3, 4)
    data[0, 0, 1, 2] = 0.6
    data[1, 0, 1, 2] = 0.7
    source = data.clone()
    voxel = {"data": data}
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(VOXEL_TYPE)
    value = spec.decode(spec.encode(voxel)) if portable else voxel
    result = VoxelToMeshBasic.execute(voxel=value, threshold=0.6)["MESH"]
    assert isinstance(result, TriangleMeshBatch)
    assert result.vertices.shape == (2, 24, 3)
    assert result.faces.shape == (2, 12, 3)
    assert result.vertex_counts.tolist() == [1, 24]
    assert result.face_counts.tolist() == [1, 12]
    torch.testing.assert_close(
        result.vertices[0, 0], torch.tensor([-1.0, -1.0, -1.0]), rtol=0, atol=0
    )
    # First exposed face is +x in source z/y/x coordinates, then converted to x/y/z.
    torch.testing.assert_close(
        result.vertices[1, :4],
        torch.tensor(
            [
                [0.5, -0.5, -1.0],
                [0.5, 0.0, -1.0],
                [0.5, 0.0, -0.5],
                [0.5, -0.5, -0.5],
            ]
        ),
        rtol=0,
        atol=0,
    )
    assert result.faces[1, :2].tolist() == [[0, 1, 2], [0, 2, 3]]
    assert result.faces.dtype == torch.float32  # Empty first row determines packed dtype in F.
    torch.testing.assert_close(data, source, rtol=0, atol=0)
    regular = VoxelToMesh.execute(
        voxel={"data": data[1:].repeat(2, 1, 1, 1)}, algorithm="basic", threshold=0.6
    )["MESH"]
    assert isinstance(regular, TriangleMeshBatch)
    assert regular.vertex_counts is None and regular.face_counts is None
    assert regular.faces.dtype == torch.int64


def test_surface_net_small_and_empty_grids_do_not_depend_on_progress_divisors() -> None:
    data = torch.zeros(1, 2, 2, 2)
    data[0, 0, 0, 0] = 1.0
    result = VoxelToMesh.execute(voxel={"data": data}, algorithm="surface net", threshold=0.5)[
        "MESH"
    ]
    assert isinstance(result, TriangleMeshBatch)
    assert result.vertices.shape == (1, 8, 3)
    assert result.faces.shape == (1, 12, 3)
    assert result.faces.dtype == torch.int64
    assert torch.isfinite(result.vertices).all()
    empty = VoxelToMesh.execute(
        voxel={"data": torch.zeros_like(data)}, algorithm="surface net", threshold=0.5
    )["MESH"]
    assert isinstance(empty, TriangleMeshBatch)
    assert empty.vertices.shape == (1, 0, 3) and empty.faces.shape == (1, 0, 3)


def test_hunyuan_decode_preserves_options_and_wraps_voxel_data() -> None:
    class VAE:
        def decode(self, samples: torch.Tensor, *, vae_options: dict[str, int]) -> torch.Tensor:
            assert samples is latent
            assert vae_options == {"num_chunks": 1234, "octree_resolution": 48}
            return data

    latent = torch.tensor([2.0, 5.0, 11.0])
    data = torch.zeros(1, 2, 3, 4)
    value = VAEDecodeHunyuan3D.execute(
        samples={"samples": latent}, vae=VAE(), num_chunks=1234, octree_resolution=48
    )["VOXEL"]
    materialized = materialize_voxel(value)
    assert materialized.data is data
    assert materialized.voxel_colors is None and materialized.resolution is None


class GeometryVAE:
    def decode(self, samples: torch.Tensor, *, vae_options: dict[str, int]) -> torch.Tensor:
        assert samples.tolist() == [2.0, 5.0, 11.0]
        assert vae_options == {"num_chunks": 1234, "octree_resolution": 48}
        data = torch.zeros(2, 2, 3, 4)
        data[0, 0, 1, 2] = 1.0
        data[1, 0, 1, 1] = 1.0
        return data


class GeometrySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="geometry-fixture.source",
            outputs=(
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(vae=GeometryVAE(), samples={"samples": torch.tensor([2.0, 5.0, 11.0])})


class GeometryInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="geometry-fixture.inspect",
            inputs=(
                InputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),
                InputSpec("basic", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("surface", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("rotated", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("basic_info", TypeExpr.concrete(CORE_STRING)),
                InputSpec("surface_info", TypeExpr.concrete(CORE_STRING)),
                InputSpec("merged", TypeExpr.concrete(MESH_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        voxel: object,
        basic: object,
        surface: object,
        rotated: object,
        basic_info: str,
        surface_info: str,
        merged: object,
    ) -> dict[str, object]:
        voxel = materialize_voxel(voxel)
        basic, surface, rotated = (materialize_mesh(value) for value in (basic, surface, rotated))
        assert voxel.data.shape == (2, 2, 3, 4) and voxel.data.sum().item() == 2.0
        assert voxel.voxel_colors is None and voxel.resolution is None
        assert basic.vertices.shape == (2, 24, 3) and basic.faces.shape == (2, 12, 3)
        assert surface.vertices.shape == (2, 8, 3) and surface.faces.shape == (2, 12, 3)
        for value in (basic, surface, rotated):
            assert value.vertex_counts is None and value.face_counts is None
            assert torch.isfinite(value.vertices).all()
        expected = basic.vertices.clone()
        expected[..., 0] = -basic.vertices[..., 1]
        expected[..., 1] = basic.vertices[..., 0]
        torch.testing.assert_close(rotated.vertices, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(rotated.faces, basic.faces, rtol=0, atol=0)
        assert "Vertices:   48 total" in basic_info and "Faces:      24 total" in basic_info
        assert "Vertices:   16 total" in surface_info and "Faces:      24 total" in surface_info
        merged = materialize_mesh(merged)
        torch.testing.assert_close(
            merged.vertices,
            torch.cat((rotated.vertices[0], surface.vertices[0])).unsqueeze(0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            merged.faces,
            torch.cat((rotated.faces[0], surface.faces[0] + 24)).unsqueeze(0),
            rtol=0,
            atol=0,
        )
        assert merged.vertices.shape == (1, 32, 3) and merged.faces.shape == (1, 24, 3)
        return cls.outputs(summary="voxel to mesh to rotation and inspection")


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_geometry_sequence_in_both_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"geometry_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_mesh_torch import GeometrySource, GeometryInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (GeometrySource, GeometryInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "geometry-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["geometry-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(__file__).parent),
                        os.environ.get("PYTHONPATH", ""),
                    )
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
                        worker_group="geometry-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("geometry-fixture.source"),
                    "decode": GraphNode(
                        "VAEDecodeHunyuan3D",
                        {
                            "samples": Link("source", "samples"),
                            "vae": Link("source", "vae"),
                            "num_chunks": 1234,
                            "octree_resolution": 48,
                        },
                    ),
                    "basic": GraphNode(
                        "VoxelToMeshBasic", {"voxel": Link("decode", "VOXEL"), "threshold": 0.6}
                    ),
                    "surface": GraphNode(
                        "VoxelToMesh",
                        {
                            "voxel": Link("decode", "VOXEL"),
                            "algorithm": "surface net",
                            "threshold": 0.6,
                        },
                    ),
                    "rotate": GraphNode(
                        "RotateMesh",
                        {"mesh": Link("basic", "MESH"), "mode.angle_z": 90.0},
                        slot_variants={"mode": "euler_xyz"},
                    ),
                    "basic-info": GraphNode("GetMeshInfo", {"mesh": Link("rotate", "mesh")}),
                    "surface-info": GraphNode("GetMeshInfo", {"mesh": Link("surface", "MESH")}),
                    "merge": GraphNode(
                        "MergeMeshes",
                        {
                            "meshes.mesh0": Link("rotate", "mesh"),
                            "meshes.mesh1": Link("surface", "MESH"),
                        },
                    ),
                    "inspect": GraphNode(
                        "geometry-fixture.inspect",
                        {
                            "voxel": Link("decode", "VOXEL"),
                            "basic": Link("basic", "MESH"),
                            "surface": Link("surface-info", "MESH"),
                            "rotated": Link("basic-info", "MESH"),
                            "basic_info": Link("basic-info", "STRING"),
                            "surface_info": Link("surface-info", "STRING"),
                            "merged": Link("merge", "mesh"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "voxel to mesh to rotation and inspection"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
