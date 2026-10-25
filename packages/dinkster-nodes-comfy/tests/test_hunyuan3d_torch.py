"""Hunyuan vision conditioning keeps view indices and crosses ordinary hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hunyuan3d import (
    Hunyuan3Dv2Conditioning,
    Hunyuan3Dv2ConditioningMultiView,
)

from dinkster.compose import PackSpec, ServingComposer


def _features(index: int) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) / 8 + index * 2


def _expected(indices: tuple[int, ...]) -> torch.Tensor:
    return torch.cat(
        [
            _features(index)
            + torch.cat(
                (
                    torch.sin(torch.tensor([index, index / 100], dtype=torch.float32)),
                    torch.cos(torch.tensor([index, index / 100], dtype=torch.float32)),
                )
            ).reshape(1, 1, 4)
            for index in indices
        ],
        dim=1,
    )


@pytest.mark.parametrize("indices", ((0,), (1,), (3,), (0, 2), (1, 3), (0, 1, 2, 3)))
def test_ordered_views_preserve_slot_positions_and_sources(indices: tuple[int, ...]) -> None:
    inputs = {
        name: {"last_hidden_state": _features(index)}
        for index, name in enumerate(("front", "left", "back", "right"))
        if index in indices
    }
    before = {name: value["last_hidden_state"].clone() for name, value in inputs.items()}
    result: Any = Hunyuan3Dv2ConditioningMultiView.execute(**inputs)
    assert torch.equal(result["positive"][0][0], _expected(indices))
    assert result["positive"][0][1] == result["negative"][0][1] == {}
    assert torch.equal(result["negative"][0][0], torch.zeros_like(_expected(indices)))
    for name, value in inputs.items():
        assert torch.equal(value["last_hidden_state"], before[name])


def test_single_view_and_empty_multiview() -> None:
    source = _features(2)
    result: Any = Hunyuan3Dv2Conditioning.execute(clip_vision_output={"last_hidden_state": source})
    assert torch.equal(result["positive"][0][0], source)
    assert torch.equal(result["negative"][0][0], torch.zeros_like(source))
    with pytest.raises(ValueError, match="non-empty"):
        Hunyuan3Dv2ConditioningMultiView.execute()


class VisionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE))
                for name in ("left", "right")
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            left={"last_hidden_state": _features(1)}, right={"last_hidden_state": _features(3)}
        )


class InspectConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hunyuan-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("single", "multi", "negative")
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, single: Any, multi: Any, negative: Any) -> dict[str, object]:
        assert torch.equal(torch.as_tensor(single[0][0]), _features(1))
        assert torch.equal(torch.as_tensor(multi[0][0]), _expected((1, 3)))
        assert torch.count_nonzero(torch.as_tensor(negative[0][0])).item() == 0
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_hunyuan_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "hunyuan_fixture.py").write_text(
                "from test_hunyuan3d_torch import VisionSource, InspectConditioning\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, VisionSource, InspectConditioning)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["hunyuan-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "hunyuan_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "hunyuan_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hunyuan-fixture.source"),
                    "single": GraphNode(
                        "Hunyuan3Dv2Conditioning", {"clip_vision_output": Link("source", "left")}
                    ),
                    "multi": GraphNode(
                        "Hunyuan3Dv2ConditioningMultiView",
                        {"left": Link("source", "left"), "right": Link("source", "right")},
                    ),
                    "inspect": GraphNode(
                        "hunyuan-fixture.inspect",
                        {
                            "single": Link("single", "positive"),
                            "multi": Link("multi", "positive"),
                            "negative": Link("multi", "negative"),
                        },
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
