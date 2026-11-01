"""Public mesh shading, counted geometry and ordinary worker transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import MESH_TYPE, Node, NodeSchema, OutputSpec, TriangleMeshBatch, TypeExpr
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_render import RenderMesh

from dinkster.compose import PackSpec, ServingComposer


def source() -> TriangleMeshBatch:
    vertices = torch.tensor(
        [
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [100.0, 100.0, 100.0],
            ]
        ]
    )
    return TriangleMeshBatch(
        vertices,
        torch.tensor([[[0, 1, 2], [0, 2, 3], [4, 4, 4]]]),
        vertex_counts=torch.tensor([4]),
        face_counts=torch.tensor([2]),
        vertex_colors=torch.tensor([[[0.25, 0.0625, 1.0]] * 5]),
        uvs=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]),
        texture=torch.tensor([[[[0.1, 0.7, 0.3]]]]),
    )


def parameters() -> dict:
    return dict(
        mode="auto",
        width=80,
        height=64,
        background="#204080",
        camera_info={
            "position": {"x": 0.0, "y": 0.0, "z": 3.0},
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "zoom": 1.0,
            "fov": 40.0,
            "cameraType": "perspective",
        },
    )


@pytest.mark.parametrize("mode", ["auto", "texture", "vertex colors", "solid", "normal", "depth"])
def test_render_mesh_shading_and_counted_padding(mode: str) -> None:
    mesh = source()
    args = parameters()
    args["mode"] = mode
    result = RenderMesh.execute(mesh=mesh, **args)
    image, mask = result["IMAGE"], result["MASK"]
    assert image.shape == (1, 64, 80, 3) and mask.shape == (1, 64, 80)
    assert image.device.type == mask.device.type == "cpu"
    assert image.dtype == mask.dtype == torch.float32
    assert mask[0, 32, 40] == 1 and mask[0, 0, 0] == 0
    torch.testing.assert_close(image[0, 0, 0], torch.tensor([32, 64, 128]) / 255)
    if mode in ("auto", "texture"):
        expected = torch.tensor([0.1, 0.7, 0.3])
    elif mode == "vertex colors":
        expected = torch.tensor([0.25, 0.0625, 1.0]).pow(1.0 / 2.2)
    elif mode == "normal":
        expected = torch.tensor([0.5, 0.5, 1.0])
    elif mode == "depth":
        assert image[0, 32, 40, 0] > image[0, 32, 60, 0]
        return
    else:
        assert 0.71 < image[0, 32, 40, 0] <= 0.72
        return
    torch.testing.assert_close(image[0, 32, 40], expected)


def test_render_mesh_camera_zoom_transform_and_auto_fallback() -> None:
    mesh = source()
    args = parameters()
    original = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    args["camera_info"]["zoom"] = 0.5
    wide = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    assert 0 < wide.sum() < original.sum()
    args["model_3d_info"] = [
        {
            "scale": {"x": 0.5, "y": 1.0, "z": 1.0},
            "position": {"x": 0.2, "y": 0.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    ]
    transformed = RenderMesh.execute(mesh=mesh, **args)["MASK"]
    assert transformed.sum() < wide.sum() and not torch.equal(wide, transformed)
    mesh.texture = None
    fallback = RenderMesh.execute(mesh=mesh, **parameters())["IMAGE"]
    torch.testing.assert_close(
        fallback[0, 32, 40], torch.tensor([0.25, 0.0625, 1.0]).pow(1.0 / 2.2)
    )
    mesh.vertex_colors = None
    args = parameters()
    args["camera_info"] = None
    default = RenderMesh.execute(mesh=mesh, **args)
    assert default["MASK"].sum() > 0
    assert torch.isfinite(default["IMAGE"]).all()


class MeshSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "render-mesh-source.make", outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict:
        return cls.outputs(mesh=source())


@pytest.mark.parametrize("in_process", [True, False])
def test_render_mesh_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    module = f"render_mesh_source_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from test_render_mesh_torch import MeshSource\n"
        "from dinkster_nodes_comfy import register_inference_types\nNODES=(MeshSource,)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "source.toml"
    manifest.write_text(
        '[pack]\nname="render-mesh-source"\npublisher="dinkster"\n'
        'namespaces=["render-mesh-source"]\n[pack.entry]\n'
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
            args = parameters()
            graph = Graph(
                {
                    "source": GraphNode("render-mesh-source.make"),
                    "render": GraphNode("RenderMesh", {"mesh": Link("source", "mesh"), **args}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["render"])
                image = result.outputs["render"]["IMAGE"].resolve()
                mask = result.outputs["render"]["MASK"].resolve()
                torch.testing.assert_close(
                    torch.as_tensor(image[0, 32, 40]), torch.tensor([0.1, 0.7, 0.3])
                )
                assert mask[0, 32, 40] == 1 and mask[0, 0, 0] == 0
        finally:
            await composer.close()

    asyncio.run(scenario())
