"""Atlas pixels distinguish chart connectivity, boundary edges and tile seams."""

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
    TypeRegistry,
    materialize_image,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.uv_atlas import RenderUVAtlas

from dinkster.compose import PackSpec, ServingComposer


def source() -> Any:
    # Two triangles share an edge; the third is a separate chart across tile 64.
    uvs = [
        [0.05, 0.05],
        [0.45, 0.05],
        [0.05, 0.45],
        [0.45, 0.45],
        [0.55, 0.55],
        [0.95, 0.55],
        [0.55, 0.95],
    ]
    return TriangleMeshBatch(
        vertices=torch.zeros(2, 7, 3),
        faces=torch.tensor([[[0, 1, 2], [1, 3, 2], [4, 5, 6]], [[0, 0, 0]] * 3]),
        uvs=torch.tensor([uvs, [[0, 0]] * 7], dtype=torch.float64),
        vertex_counts=torch.tensor([7, 0]),
        face_counts=torch.tensor([3, 0]),
    )


def assert_atlas(value: object) -> None:
    image = materialize_image(value)
    assert image.shape == (1, 128, 128, 3)
    assert image.dtype == torch.float32 and image.device.type == "cpu"
    blue = torch.tensor([0.121568627, 0.466666667, 0.705882353])
    pale_blue = torch.tensor([0.682352941, 0.780392157, 0.909803922])
    # Seed 42 permutes two charts to [1, 0]; shared-edge triangles keep one color.
    for y, x in ((15, 15), (45, 45), (31, 31)):
        torch.testing.assert_close(image[0, y, x], pale_blue, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 80, 80], blue, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 64, 64], torch.full((3,), 0.13), rtol=0, atol=0)
    torch.testing.assert_close(image[0, 6, 20], torch.zeros(3), rtol=0, atol=0)


@pytest.mark.parametrize("portable", [False, True])
def test_charts_boundaries_and_first_batch(portable: bool) -> None:
    mesh = source()
    before = mesh.uvs.clone()
    registry = TypeRegistry()
    register_inference_types(registry)
    codec = registry.spec(MESH_TYPE)
    value = codec.decode(codec.encode(mesh)) if portable else mesh
    assert_atlas(RenderUVAtlas.execute(mesh=value, resolution=128)["image"])
    torch.testing.assert_close(mesh.uvs, before, rtol=0, atol=0)


def test_unbatched_uvs_empty_faces_degenerate_and_missing_uvs() -> None:
    mesh = source()
    mesh.uvs = mesh.uvs[0]
    assert_atlas(RenderUVAtlas.execute(mesh=mesh, resolution=128)["image"])
    mesh.faces = torch.zeros(1, 0, 3, dtype=torch.int64)
    result = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(result, torch.full((1, 64, 64, 3), 0.13), rtol=0, atol=0)
    mesh.faces = torch.tensor([[[0, 0, 0]]])
    result = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(result, torch.full((1, 64, 64, 3), 0.13), rtol=0, atol=0)
    mesh.uvs = None
    with pytest.raises(RuntimeError, match="mesh has no UVs"):
        RenderUVAtlas.execute(mesh=mesh, resolution=64)


def test_clipped_uvs_and_first_overlapping_chart() -> None:
    mesh = TriangleMeshBatch(
        vertices=torch.zeros(1, 6, 3),
        faces=torch.tensor([[[0, 1, 2], [3, 4, 5]]]),
        uvs=torch.tensor([[[-1, -1], [2, -1], [-1, 2], [0, 0], [1, 0], [0, 1]]]),
    )
    image = RenderUVAtlas.execute(mesh=mesh, resolution=64)["image"]
    torch.testing.assert_close(
        image[0, 15, 15], torch.tensor([0.682352941, 0.780392157, 0.909803922]), rtol=0, atol=0
    )
    torch.testing.assert_close(image[0, 0, 20], torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(image[0, 50, 50], torch.full((3,), 0.13), rtol=0, atol=0)


class AtlasSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "atlas-fixture.source", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_atlas_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        (tmp_path / "atlas_fixture.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_uv_atlas_torch import AtlasSource\nNODES=(AtlasSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="atlas-fixture"\npublisher="dinkster"\nnamespaces=["atlas-fixture"]\n'
            '[pack.entry]\nnodes="atlas_fixture:NODES"\ntypes="atlas_fixture:register_inference_types"\n',
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
                    "source": GraphNode("atlas-fixture.source"),
                    "atlas": GraphNode(
                        "RenderUVAtlas", {"mesh": Link("source", "mesh"), "resolution": 128}
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["atlas"])
                assert_atlas(result.outputs["atlas"]["image"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
