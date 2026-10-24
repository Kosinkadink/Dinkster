"""Actual reset CLIP vision outputs round-trip through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_clip_vision_output,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


def _source(variant: int) -> Any:
    attributes = {
        "last_hidden_state": torch.arange(6, dtype=torch.float16).reshape(2, 3).T,
        "image_embeds": torch.tensor([0x3F80, 0x8000, 0x7FC1], dtype=torch.uint16).view(
            torch.bfloat16
        ),
        "penultimate_hidden_states": torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        "image_sizes": [(224, 336, 3), (448, 224, 3)],
        "mm_projected": None,
    }
    if variant:
        attributes["all_hidden_states"] = (
            torch.empty((2, 0, 3), dtype=torch.float32),
            {"layer": 4, "label": "last", "flag": True, "weight": 0.25, "empty": None},
        )
    return materialize_clip_vision_output(attributes)


def _inspect(value: Any) -> dict[str, Any]:
    output = materialize_clip_vision_output(value)
    assert type(output) is importlib.import_module("dinkster_inference.clip_vision").Output

    def plain(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "bytes": bytes(item.contiguous().reshape(-1).view(torch.uint8).tolist()).hex(),
            }
        if isinstance(item, dict):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return {"tuple": [plain(child) for child in item]}
        if isinstance(item, list):
            return [plain(child) for child in item]
        return item

    return plain(vars(output))


@pytest.mark.parametrize("variant", (0, 1))
def test_actual_output_materialization_is_lossless_and_does_not_mutate(variant: int) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(CLIP_VISION_OUTPUT_TYPE)
    source = _source(variant)
    original_attributes = vars(source).copy()
    before = _inspect(source)
    decoded = spec.decode(spec.encode(source))
    assert _inspect(decoded) == before
    assert spec.fingerprint is not None
    assert spec.fingerprint(source) == spec.fingerprint(decoded)
    for name, item in original_attributes.items():
        assert getattr(source, name) is item
    assert _inspect(source) == before
    if variant:
        assert "all_hidden_states" in before
    else:
        assert "all_hidden_states" not in before


class VisionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision-fixture.source",
            inputs=(InputSpec("variant", TypeExpr.concrete(CORE_INT)),),
            outputs=(OutputSpec("vision", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, variant: int) -> dict[str, object]:
        return cls.outputs(vision=_source(variant))


class VisionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision-fixture.inspect",
            inputs=(InputSpec("vision", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, vision: Any) -> dict[str, object]:
        return cls.outputs(text=json.dumps(_inspect(vision), sort_keys=True))


FIXTURES = (VisionSource, VisionInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_vision_output_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        test_dir = str(Path(__file__).resolve().parent)
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((test_dir, os.environ.get("PYTHONPATH", "")))}
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "vision_fixture.py").write_text(
                "from test_vision_output_torch import FIXTURES as NODES\n"
                "from dinkster_api.v1 import register_inference_types\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["vision-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "vision_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "vision_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for variant in (0, 1):
                graph = Graph(
                    nodes={
                        "source": GraphNode("vision-fixture.source", {"variant": variant}),
                        "inspect": GraphNode(
                            "vision-fixture.inspect", {"vision": Link("source", "vision")}
                        ),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["inspect"])
                    value = json.loads(result.outputs["inspect"]["text"].resolve())
                    assert value["last_hidden_state"] == {
                        "dtype": "torch.float16",
                        "shape": [3, 2],
                        "bytes": "00000042003c004400400045",
                    }
                    assert value["image_embeds"] == {
                        "dtype": "torch.bfloat16",
                        "shape": [3],
                        "bytes": "803f0080c17f",
                    }
                    assert value["penultimate_hidden_states"] == {
                        "dtype": "torch.float64",
                        "shape": [1, 2],
                        "bytes": "000000000000f03f00000000000000c0",
                    }
                    assert value["image_sizes"] == [
                        {"tuple": [224, 336, 3]},
                        {"tuple": [448, 224, 3]},
                    ]
                    assert value["mm_projected"] is None
                    if variant:
                        assert value["all_hidden_states"] == {
                            "tuple": [
                                {"dtype": "torch.float32", "shape": [2, 0, 3], "bytes": ""},
                                {
                                    "layer": 4,
                                    "label": "last",
                                    "flag": True,
                                    "weight": 0.25,
                                    "empty": None,
                                },
                            ]
                        }
                    else:
                        assert "all_hidden_states" not in value
        finally:
            await composer.close()

    asyncio.run(scenario())
