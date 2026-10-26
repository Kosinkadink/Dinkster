"""Marigold prediction conversion and ordinary manifest execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.marigold import MarigoldV2PostProcess

from dinkster.compose import PackSpec, ServingComposer


def test_depth_normalizes_each_image_and_preserves_constant_images() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 5.0]], [[11.0, 13.0], [15.0, 19.0]]])
    source = values.unsqueeze(-1).repeat(1, 1, 1, 3)
    before = source.clone()
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="depth")["IMAGE"]
    expected = torch.tensor([[1.0, 0.75], [0.5, 0.0]]).expand(2, 2, 2)
    assert torch.equal(result, expected.unsqueeze(-1).repeat(1, 1, 1, 3))
    assert torch.equal(source, before)
    constant: Any = MarigoldV2PostProcess.execute(
        image=torch.full((2, 2, 3, 3), 0.7), prediction="depth"
    )["IMAGE"]
    assert torch.count_nonzero(constant).item() == 0


def test_normals_use_channel_axis_and_zero_vectors_remain_midpoint() -> None:
    source = torch.tensor([[[[0.65, 0.7, 0.5], [0.5, 0.5, 0.5]]]], dtype=torch.float64)
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="normals")["IMAGE"]
    assert torch.allclose(
        result, torch.tensor([[[[0.8, 0.9, 0.5], [0.5, 0.5, 0.5]]]], dtype=torch.float64)
    )
    assert result.dtype == source.dtype


def test_albedo_transfer_includes_negative_and_above_one_values() -> None:
    source = torch.tensor([[[[-0.1, 0.0, 0.0031308], [0.003131, 0.25, 1.2]]]])
    expected = torch.tensor(
        [
            [
                [
                    [-1.292, 0.0, 0.0031308 * 12.92],
                    [
                        1.055 * 0.003131 ** (1 / 2.4) - 0.055,
                        1.055 * 0.25 ** (1 / 2.4) - 0.055,
                        1.055 * 1.2 ** (1 / 2.4) - 0.055,
                    ],
                ]
            ]
        ]
    )
    result: Any = MarigoldV2PostProcess.execute(image=source, prediction="albedo")["IMAGE"]
    assert torch.allclose(result, expected)
    assert torch.isfinite(result).all()


class PredictionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="marigold-fixture.source",
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=torch.tensor([[[[0.0] * 3, [0.5] * 3, [1.0] * 3]]]))


class InspectDepth(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="marigold-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, image: object) -> dict[str, object]:
        expected = torch.tensor([[[[1.0] * 3, [0.5] * 3, [0.0] * 3]]])
        assert torch.equal(materialize_image(image), expected)
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_marigold_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(root / "tests"), str(root / "src"), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            (tmp_path / "marigold_fixture.py").write_text(
                "from test_marigold_torch import PredictionSource, InspectDepth\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, PredictionSource, InspectDepth)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["marigold-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "marigold_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "marigold_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("marigold-fixture.source"),
                    "convert": GraphNode(
                        "MarigoldV2PostProcess",
                        {"image": Link("source", "image"), "prediction": "depth"},
                    ),
                    "inspect": GraphNode(
                        "marigold-fixture.inspect", {"image": Link("convert", "IMAGE")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
