"""Public Gaussian transform geometry, covariance, and ordinary host contracts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SPLAT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_splat,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.gaussian_splat import TransformSplat

from dinkster.compose import PackSpec, ServingComposer


def _source(case: str, dtype: torch.dtype) -> SimpleNamespace:
    size = 0 if case == "empty" else 2
    return SimpleNamespace(
        positions=torch.tensor([[[1, -2, 3], [-4, 5, 6]]], dtype=dtype)[:, :size],
        scales=torch.tensor([[[1, 2, 3], [2, 3, 4]]], dtype=dtype)[:, :size],
        rotations=torch.tensor([[[0.8, 0.6, 0, 0], [1, 0, 0, 0]]], dtype=dtype)[:, :size],
        opacities=torch.tensor([[[0.2], [0.7]]], dtype=dtype)[:, :size],
        sh=torch.arange(24, dtype=dtype).reshape(1, 2, 4, 3)[:, :size],
        counts=torch.tensor([size - 1 if size else 0], dtype=torch.int32),
    )


def _parameters(case: str) -> dict[str, float]:
    return dict(
        translate_x=5.0,
        translate_y=-7.0,
        translate_z=11.0,
        rotate_x=90.0,
        rotate_y=0.0,
        rotate_z=90.0,
        scale_x=2.0,
        scale_y=2.0 if case == "uniform" else 3.0,
        scale_z=2.0 if case == "uniform" else 4.0,
    )


def _assert_geometry(output: object, case: str, dtype: torch.dtype) -> None:
    out = materialize_splat(output)
    source = _source(case, dtype)
    assert out.positions.dtype == dtype and out.scales.dtype == dtype
    assert out.rotations.dtype == dtype and out.positions.device.type == "cpu"
    # X then Z quarter turns map (x,y,z) to (z,x,y), before per-axis scaling.
    rotation = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=dtype)
    diagonal = torch.diag(torch.tensor([2, 2, 2] if case == "uniform" else [2, 3, 4], dtype=dtype))
    transform = diagonal @ rotation
    expected = source.positions @ transform.T + torch.tensor([5, -7, 11], dtype=dtype)
    torch.testing.assert_close(out.positions, expected, rtol=2e-6, atol=2e-6)
    for name in ("opacities", "sh", "counts"):
        assert torch.equal(getattr(out, name), getattr(source, name))
    if case == "empty":
        assert out.scales.shape == (1, 0, 3) and out.rotations.shape == (1, 0, 4)
        return
    if case == "uniform":
        torch.testing.assert_close(out.scales, source.scales * 2, rtol=0, atol=0)
    # Independently reconstruct output covariance from axis-angle rotation matrices.
    for index, initial_rotation in enumerate(
        (
            torch.tensor([[1, 0, 0], [0, 0.28, -0.96], [0, 0.96, 0.28]], dtype=dtype),
            torch.eye(3, dtype=dtype),
        )
    ):
        initial_covariance = (
            initial_rotation @ torch.diag(source.scales[0, index].square()) @ initial_rotation.T
        )
        expected_covariance = transform @ initial_covariance @ transform.T
        q = out.rotations[0, index]
        torch.testing.assert_close(q.norm(), torch.tensor(1.0, dtype=dtype))
        w, vector = q[0], q[1:]
        x, y, z = vector
        skew = torch.stack(
            (torch.stack((x * 0, -z, y)), torch.stack((z, x * 0, -x)), torch.stack((-y, x, x * 0)))
        )
        axes = (
            (w.square() - vector.dot(vector)) * torch.eye(3, dtype=dtype)
            + 2 * torch.outer(vector, vector)
            + 2 * w * skew
        )
        actual_covariance = axes @ torch.diag(out.scales[0, index].square()) @ axes.T
        torch.testing.assert_close(actual_covariance, expected_covariance, rtol=3e-6, atol=3e-5)


@pytest.mark.parametrize("case", ["uniform", "nonuniform", "empty"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transform_splat_preserves_source_and_transforms_geometry(
    case: str, dtype: torch.dtype
) -> None:
    source = _source(case, dtype)
    before = {key: value.clone() for key, value in vars(source).items()}
    out = materialize_splat(TransformSplat.execute(splat=source, **_parameters(case))["splat"])
    _assert_geometry(out, case, dtype)
    for name, original in before.items():
        assert torch.equal(getattr(source, name), original)
    for name in ("opacities", "sh", "counts"):
        assert getattr(out, name) is getattr(source, name)


class TransformSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-transform-fixture.source",
            inputs=(InputSpec("case", TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(splat=_source(case, torch.float64))


class TransformCheck(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-transform-fixture.check",
            inputs=(
                InputSpec("case", TypeExpr.concrete(CORE_STRING)),
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, case: str, splat: object) -> dict[str, object]:
        _assert_geometry(splat, case, torch.float64)
        return cls.outputs(result="geometry verified")


@pytest.mark.parametrize("in_process", [True, False])
def test_transform_splat_through_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        module = f"splat_transform_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_transform_splat_torch import TransformSource, TransformCheck\n"
            "NODES = (TransformSource, TransformCheck)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="splat-transform-fixture"\npublisher="dinkster"\nnamespaces=["splat-transform-fixture"]\n[pack.entry]\n'
            + f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
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
            for path in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for case in ("uniform", "nonuniform", "empty"):
                graph = Graph(
                    {
                        "source": GraphNode("splat-transform-fixture.source", {"case": case}),
                        "transform": GraphNode(
                            "TransformSplat",
                            {"splat": Link("source", "splat"), **_parameters(case)},
                        ),
                        "check": GraphNode(
                            "splat-transform-fixture.check",
                            {"splat": Link("transform", "splat"), "case": case},
                        ),
                    }
                )
                for _attempt in range(2):
                    result = await engine.run(graph, ["check"])
                    assert result.outputs["check"]["result"].resolve() == "geometry verified"
        finally:
            await composer.close()

    asyncio.run(scenario())
