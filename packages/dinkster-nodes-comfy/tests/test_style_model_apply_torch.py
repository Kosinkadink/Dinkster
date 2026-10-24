"""Style-model application preserves ComfyUI conditioning semantics."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    STYLE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.style_model import StyleModelApply

from dinkster.compose import PackSpec, ServingComposer


def _model() -> Any:
    initialize_inference_process()
    projection = torch.nn.Linear(3, 2)
    projection.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]]))
    projection.bias = torch.nn.Parameter(torch.tensor([0.75, -0.5]))
    return importlib.import_module("dinkster_inference.sd").StyleModel(projection)


def _vision() -> dict[str, object]:
    return {
        "last_hidden_state": torch.tensor(
            [[[2.0, 3.0, -1.0], [-4.0, 0.5, 5.0]]], dtype=torch.float32
        ),
        "image_embeds": {"reference": torch.tensor([[5.0, -3.0]])},
        "image_sizes": [(448, 224, 3)],
    }


@pytest.mark.parametrize("strength", (0.0, 1.0, 2.5))
def test_multiply_strength_and_masks_preserve_inputs(strength: float) -> None:
    style = _model()
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    boolean_mask = torch.tensor(
        [
            [
                [True, False, True, True, False, True, True, False],
                [False, True, False, True, True, False, True, True],
                [True, True, False, False, True, True, False, True],
                [True, False, True, True, False, False, True, True],
                [False, True, True, False, True, True, False, False],
                [True, True, False, True, True, False, True, False],
                [False, True, True, False, True, False, True, True],
                [True, False, True, True, False, True, False, True],
            ]
        ],
        dtype=torch.bool,
    )
    additive_mask = torch.arange(8 * 8, dtype=torch.float16).reshape(1, 8, 8)
    source = [
        [text, {"attention_mask": boolean_mask, "attention_mask_img_shape": (2, 3), "tag": "one"}],
        [
            text + 10,
            {"attention_mask": additive_mask, "attention_mask_img_shape": (2, 3), "tag": "two"},
        ],
    ]
    original_text = [record[0].clone() for record in source]
    original_masks = [record[1]["attention_mask"].clone() for record in source]

    output = cast(
        list[list[Any]],
        StyleModelApply.execute(
            conditioning=source,
            style_model=style,
            clip_vision_output=_vision(),
            strength=strength,
            strength_type="multiply",
        )["CONDITIONING"],
    )

    expected = style.get_cond(materialize_clip_vision_output(_vision())).flatten(0, 1).unsqueeze(0)
    assert [record[1]["tag"] for record in output] == ["one", "two"]
    for index, (result, original, mask) in enumerate(
        zip(output, original_text, original_masks, strict=True)
    ):
        assert torch.equal(result[0][:, :2], original)
        assert torch.equal(result[0][:, 2:], expected * strength)
        new_mask = result[1]["attention_mask"]
        assert new_mask.shape == (1, 10, 10)
        assert result[1]["attention_mask_img_shape"] == (2, 3)
        expected_mask = torch.log(mask.to(torch.float16)) if index == 0 else mask
        assert torch.equal(new_mask[:, :2, :2], expected_mask[:, :2, :2])
        assert torch.equal(new_mask[:, :2, 4:], expected_mask[:, :2, 2:])
        assert torch.equal(new_mask[:, 4:, :2], expected_mask[:, 2:, :2])
        assert torch.equal(new_mask[:, 4:, 4:], expected_mask[:, 2:, 2:])
        assert torch.equal(new_mask[:, :2, 2:4], torch.zeros((1, 2, 2), dtype=torch.float16))
    for record, original, mask in zip(source, original_text, original_masks, strict=True):
        assert torch.equal(record[0], original)
        assert torch.equal(record[1]["attention_mask"], mask)


@pytest.mark.parametrize("strength", (0.0, 1.0, 0.25))
def test_attention_bias_strength_adds_mask_without_changing_tokens(strength: float) -> None:
    style = _model()
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    source = [[text, {"tag": "only"}]]

    output = cast(
        list[list[Any]],
        StyleModelApply.execute(
            conditioning=source,
            style_model=style,
            clip_vision_output=_vision(),
            strength=strength,
            strength_type="attn_bias",
        )["CONDITIONING"],
    )[0]
    expected = style.get_cond(materialize_clip_vision_output(_vision())).flatten(0, 1).unsqueeze(0)
    assert torch.equal(output[0][:, :2], text)
    assert torch.equal(output[0][:, 2:], expected)
    if strength == 1.0:
        assert "attention_mask" not in output[1]
    else:
        mask = output[1]["attention_mask"]
        assert mask.shape == (1, 5, 5)
        bias = torch.log(torch.tensor(strength, dtype=mask.dtype))
        assert torch.equal(mask[:, :2, 2:4], torch.full((1, 2, 2), bias, dtype=mask.dtype))
        assert torch.equal(mask[:, 4:, 2:4], torch.full((1, 1, 2), bias, dtype=mask.dtype))
    assert torch.equal(source[0][0], text)
    assert source[0][1] == {"tag": "only"}


class ConditioningSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-apply-fixture.conditioning-source",
            outputs=(OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(conditioning=[[torch.tensor([[[1.0, 2.0]]]), {"source": "raw"}]])


class StyleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-apply-fixture.style-source",
            outputs=(OutputSpec("style", TypeExpr.concrete(STYLE_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(style=_model())


class VisionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-apply-fixture.vision-source",
            outputs=(OutputSpec("vision", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(vision=_vision())


class ConditioningInspector(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="style-apply-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: object) -> dict[str, object]:
        records = cast(list[list[Any]], conditioning)
        return cls.outputs(
            summary=json.dumps(
                [[list(record[0].shape), record[1].get("source")] for record in records],
                separators=(",", ":"),
            )
        )


FIXTURES = (ConditioningSource, StyleSource, VisionSource, ConditioningInspector)


@pytest.mark.parametrize("in_process", (True, False))
def test_style_apply_runs_through_native_conditioning_consumer_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        test_dir = str(Path(__file__).resolve().parent)
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((test_dir, os.environ.get("PYTHONPATH", ""))),
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "style_apply_fixture.py").write_text(
                "from test_style_model_apply_torch import FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["style-apply-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "style_apply_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "style_apply_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "conditioning": GraphNode("style-apply-fixture.conditioning-source"),
                    "style": GraphNode("style-apply-fixture.style-source"),
                    "vision": GraphNode("style-apply-fixture.vision-source"),
                    "apply": GraphNode(
                        "StyleModelApply",
                        {
                            "conditioning": Link("conditioning", "conditioning"),
                            "style_model": Link("style", "style"),
                            "clip_vision_output": Link("vision", "vision"),
                            "strength": 1.0,
                            "strength_type": "multiply",
                        },
                    ),
                    "consume": GraphNode(
                        "ConditioningConcat",
                        {
                            "conditioning_to": Link("apply", "CONDITIONING"),
                            "conditioning_from": Link("conditioning", "conditioning"),
                        },
                    ),
                    "inspect": GraphNode(
                        "style-apply-fixture.inspect",
                        {"conditioning": Link("consume", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert json.loads(result.outputs["inspect"]["summary"].resolve()) == [
                    [[1, 4, 2], "raw"]
                ]
                assert "inspect" in result.cached if attempt else not result.cached
        finally:
            await composer.close()

    asyncio.run(scenario())
