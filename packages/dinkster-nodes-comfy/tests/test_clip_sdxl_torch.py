"""SDXL scheduled tensor conditioning through ordinary CPU pack hosts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class TensorEncoder:
    def tokenize(self, text: str) -> dict[str, list[list[int]]]:
        values = [[ord(character)] for character in text] or [[0]]
        return {"g": values.copy(), "l": values.copy()}

    def encode_from_tokens_scheduled(
        self, tokens: dict[str, list[list[int]]], *, add_dict: dict[str, object]
    ) -> list[list[object]]:
        tensor = torch.tensor([tokens["g"], tokens["l"]], dtype=torch.float32)
        return [
            [tensor, {**add_dict, "start_percent": 0.125}],
            [tensor + 1, {**add_dict, "start_percent": 0.75}],
        ]


class TensorEncoderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-tensor.source",
            outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(CLIP=TensorEncoder())


class TensorConditioningInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sdxl-tensor.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("json", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> dict[str, object]:
        return cls.outputs(
            json=json.dumps(
                [[torch.as_tensor(tensor).tolist(), meta] for tensor, meta in conditioning]
            )
        )


TENSOR_FIXTURES = (TensorEncoderSource, TensorConditioningInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_transports_scheduled_sdxl_tensors(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src"))
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "sdxl_tensor_fixture.py").write_text(
                "from test_clip_sdxl_torch import TENSOR_FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *TENSOR_FIXTURES)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["sdxl-tensor",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "sdxl_tensor_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "sdxl_tensor_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            size = {
                "width": 768,
                "height": 1280,
                "crop_w": 13,
                "crop_h": 27,
                "target_width": 1536,
                "target_height": 896,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("sdxl-tensor.source"),
                    "base": GraphNode(
                        "CLIPTextEncodeSDXL",
                        {
                            "clip": Link("source", "CLIP"),
                            "text_g": "A",
                            "text_l": "az",
                            **size,
                        },
                    ),
                    "refiner": GraphNode(
                        "CLIPTextEncodeSDXLRefiner",
                        {
                            "clip": Link("source", "CLIP"),
                            "text": "B",
                            "ascore": 3.25,
                            "width": 640,
                            "height": 1152,
                        },
                    ),
                    "base-json": GraphNode(
                        "sdxl-tensor.inspect",
                        {
                            "conditioning": Link("base", "CONDITIONING"),
                        },
                    ),
                    "refiner-json": GraphNode(
                        "sdxl-tensor.inspect",
                        {
                            "conditioning": Link("refiner", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["base-json", "refiner-json"])
                assert json.loads(result.outputs["base-json"]["json"].resolve()) == [
                    [[[[65.0], [0.0]], [[97.0], [122.0]]], {**size, "start_percent": 0.125}],
                    [[[[66.0], [1.0]], [[98.0], [123.0]]], {**size, "start_percent": 0.75}],
                ]
                metadata = {"aesthetic_score": 3.25, "width": 640, "height": 1152}
                assert json.loads(result.outputs["refiner-json"]["json"].resolve()) == [
                    [[[[66.0]], [[66.0]]], {**metadata, "start_percent": 0.125}],
                    [[[[67.0]], [[67.0]]], {**metadata, "start_percent": 0.75}],
                ]
        finally:
            await composer.close()

    asyncio.run(scenario())
