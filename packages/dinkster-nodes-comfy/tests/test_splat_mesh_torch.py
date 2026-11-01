"""CPU colored surface extraction and ordinary native host transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import SPLAT_TYPE, Node, NodeSchema, OutputSpec, TypeExpr, materialize_mesh
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.splat_mesh import SplatToMesh

from dinkster.compose import PackSpec, ServingComposer


def _source() -> SimpleNamespace:
    return SimpleNamespace(
        positions=torch.tensor(
            [[[0.3, 0.2, -0.1], [100, 200, 300]], [[0.3, 0.2, -0.1], [1, 2, 3]]]
        ),
        scales=torch.tensor([0.25, 0.15, 0.1]).expand(2, 2, 3).clone(),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(2, 2, 4).clone(),
        opacities=torch.tensor([[[0.8], [1.0]], [[0.01], [1.0]]]),
        sh=((torch.tensor([0.8, 0.3, 0.15]) - 0.5) / 0.28209479177387814)
        .expand(2, 2, 1, 3)
        .clone(),
        counts=torch.tensor([1, 1]),
    )


PARAMS = dict(
    resolution=64,
    kernel=5,
    smooth=0,
    level=0.4,
    min_component=0,
    min_opacity=0.02,
    color_sharpen=2.0,
)


def _assert_mesh(value: object) -> None:
    mesh = materialize_mesh(value)
    assert mesh.unlit
    nv, nf = int(mesh.vertex_counts[0]), int(mesh.face_counts[0])
    assert nv > 10 and nf > 10
    assert mesh.vertex_counts[1] == mesh.face_counts[1] == 0
    assert mesh.vertices.dtype == mesh.vertex_colors.dtype == torch.float32
    assert mesh.faces.dtype == torch.int64
    assert mesh.vertices.device.type == "cpu"
    assert torch.isfinite(mesh.vertices).all()
    assert mesh.faces[0, :nf].min() >= 0 and mesh.faces[0, :nf].max() < nv
    center = (mesh.vertices[0, :nv].amin(0) + mesh.vertices[0, :nv].amax(0)) / 2
    torch.testing.assert_close(center, torch.tensor([0.3, -0.2, 0.1]), atol=0.025, rtol=0)
    rgb = torch.tensor([0.8, 0.3, 0.15])
    linear = ((rgb + 0.055) / 1.055) ** 2.4
    torch.testing.assert_close(mesh.vertex_colors[0, :nv], linear.expand(nv, 3), atol=0.001, rtol=0)
    assert mesh.vertices[1].count_nonzero() == mesh.faces[1].count_nonzero() == 0


def test_splat_mesh_honors_counts_cutoff_linear_colors_and_frame() -> None:
    _assert_mesh(SplatToMesh.execute(splat=_source(), **PARAMS)["mesh"])


@pytest.mark.parametrize("smooth,minimum,sharpen", [(2, 0, 1.0), (0, 100000, 3.0)])
def test_splat_mesh_smoothing_and_component_cutoff(
    smooth: int, minimum: int, sharpen: float
) -> None:
    params = {**PARAMS, "smooth": smooth, "min_component": minimum, "color_sharpen": sharpen}
    mesh = materialize_mesh(SplatToMesh.execute(splat=_source(), **params)["mesh"])
    if minimum:
        assert mesh.vertices.shape == mesh.faces.shape == (2, 0, 3)
        assert mesh.vertex_counts.tolist() == mesh.face_counts.tolist() == [0, 0]
    else:
        _assert_mesh(mesh)


class SplatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-mesh-source.source",
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(splat=_source())


@pytest.mark.parametrize("in_process", [True, False])
def test_splat_mesh_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        module = f"splat_mesh_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_splat_mesh_torch import SplatSource\nNODES=(SplatSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="splat-mesh-source"\npublisher="dinkster"\n'
            'namespaces=["splat-mesh-source"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
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
            for path in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "source": GraphNode("splat-mesh-source.source"),
                    "mesh": GraphNode("SplatToMesh", {"splat": Link("source", "splat"), **PARAMS}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["mesh"])
                _assert_mesh(result.outputs["mesh"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
