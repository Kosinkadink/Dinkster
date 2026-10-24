"""Native unCLIP conditioning preserves ordered metadata through pack hosts."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_clip_vision_output,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.clip import unCLIPConditioning

from dinkster.compose import PackSpec, ServingComposer


def _vision_output(value: float) -> Any:
    return materialize_clip_vision_output(
        {
            "image_embeds": torch.tensor([[value, value + 0.5]], dtype=torch.float32),
            "image_sizes": [(224, 336, 3)],
        }
    )


def _vision_values(conditioning: list[list[Any]]) -> list[list[dict[str, object]]]:
    return [entry[1]["unclip_conditioning"] for entry in conditioning]


def _assert_record(
    record: dict[str, object],
    value: float,
    image_sizes: list[tuple[int, int, int]],
    strength: float | None,
    noise: float | None,
) -> None:
    output = record["clip_vision_output"]
    assert type(output) is importlib.import_module("dinkster_inference.clip_vision").Output
    assert torch.equal(
        output.image_embeds, torch.tensor([[value, value + 0.5]], dtype=torch.float32)
    )
    assert output.image_sizes == image_sizes
    if strength is None:
        assert "strength" not in record
    else:
        assert record["strength"] == strength
    if noise is None:
        assert "noise_augmentation" not in record
    else:
        assert record["noise_augmentation"] == noise


def test_unclip_conditioning_appends_without_mutating_ordered_metadata() -> None:
    earlier = _vision_output(1.0)
    earlier_second = _vision_output(2.0)
    conditioning = [
        [
            torch.tensor([[3.0]]),
            {"tag": "first", "unclip_conditioning": [{"clip_vision_output": earlier}]},
        ],
        [
            torch.tensor([[4.0]]),
            {
                "tag": "second",
                "unclip_conditioning": [
                    {"clip_vision_output": earlier},
                    {"clip_vision_output": earlier_second},
                ],
            },
        ],
        [torch.tensor([[6.0]]), {"tag": "third", "unclip_conditioning": None}],
    ]
    source_metadata = [entry[1].copy() for entry in conditioning]

    result: Any = unCLIPConditioning.execute(
        conditioning=conditioning,
        clip_vision_output={
            "image_embeds": torch.tensor([[5.0, 5.5]], dtype=torch.float32),
            "image_sizes": [(448, 224, 3)],
        },
        strength=-2.5,
        noise_augmentation=0.75,
    )["CONDITIONING"]

    assert [entry[0] is source[0] for entry, source in zip(result, conditioning, strict=True)] == [
        True,
        True,
        True,
    ]
    assert [entry[1]["tag"] for entry in result] == ["first", "second", "third"]
    assert [len(records) for records in _vision_values(result)] == [2, 3, 1]
    _assert_record(_vision_values(result)[0][0], 1.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[0][1], 5.0, [(448, 224, 3)], -2.5, 0.75)
    _assert_record(_vision_values(result)[1][0], 1.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[1][1], 2.0, [(224, 336, 3)], None, None)
    _assert_record(_vision_values(result)[1][2], 5.0, [(448, 224, 3)], -2.5, 0.75)
    _assert_record(_vision_values(result)[2][0], 5.0, [(448, 224, 3)], -2.5, 0.75)
    assert [entry[1] for entry in conditioning] == source_metadata


def test_unclip_conditioning_strength_zero_returns_source() -> None:
    conditioning = [[torch.tensor([[3.0]]), {"tag": "source"}]]

    result: Any = unCLIPConditioning.execute(
        conditioning=conditioning,
        clip_vision_output={"image_embeds": torch.tensor([[5.0]])},
        strength=0.0,
        noise_augmentation=0.25,
    )["CONDITIONING"]

    assert result is conditioning


class VisionEncoder:
    def encode_image(self, image: torch.Tensor, *, crop: bool) -> Any:
        assert image.shape == (1, 2, 2, 3)
        assert not crop
        return _vision_output(5.0)


class unCLIPSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="unclip-fixture.source",
            outputs=(
                OutputSpec("clip_vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        earlier = _vision_output(1.0)
        return cls.outputs(
            clip_vision=VisionEncoder(),
            image=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
            conditioning=[
                [
                    torch.tensor([[3.0]]),
                    {"tag": "first", "unclip_conditioning": [{"clip_vision_output": earlier}]},
                ],
                [
                    torch.tensor([[4.0]]),
                    {
                        "tag": "second",
                        "unclip_conditioning": [{"clip_vision_output": earlier}],
                    },
                ],
            ],
        )


class unCLIPInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="unclip-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("json", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: list[list[Any]]) -> dict[str, object]:
        records = _vision_values(conditioning)
        assert [entry[1]["tag"] for entry in conditioning] == ["first", "second"]
        assert [entry[1]["strength"] for entry in conditioning] == [1.75, 1.75]
        assert [len(value) for value in records] == [2, 2]
        for value in records:
            _assert_record(value[0], 1.0, [(224, 336, 3)], None, None)
            _assert_record(value[1], 5.0, [(224, 336, 3)], -2.5, 0.75)
        return cls.outputs(json="ok")


FIXTURES = (unCLIPSource, unCLIPInspect)


@pytest.mark.parametrize("in_process", (True, False))
def test_unclip_conditioning_ordinary_manifest_uses_vision_encoder_and_native_consumer(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        test_dir = str(Path(__file__).resolve().parent)
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((test_dir, os.environ.get("PYTHONPATH", "")))}
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "unclip_fixture.py").write_text(
                "from test_unclip_conditioning_torch import FIXTURES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["unclip-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "unclip_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "unclip_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("unclip-fixture.source"),
                    "encode": GraphNode(
                        "CLIPVisionEncode",
                        {
                            "clip_vision": Link("source", "clip_vision"),
                            "image": Link("source", "image"),
                            "crop": "none",
                        },
                    ),
                    "unclip": GraphNode(
                        "unCLIPConditioning",
                        {
                            "conditioning": Link("source", "conditioning"),
                            "clip_vision_output": Link("encode", "CLIP_VISION_OUTPUT"),
                            "strength": -2.5,
                            "noise_augmentation": 0.75,
                        },
                    ),
                    "consumer": GraphNode(
                        "ConditioningSetAreaStrength",
                        {"conditioning": Link("unclip", "CONDITIONING"), "strength": 1.75},
                    ),
                    "inspect": GraphNode(
                        "unclip-fixture.inspect",
                        {"conditioning": Link("consumer", "CONDITIONING")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["json"].resolve() == "ok"
        finally:
            await composer.close()

    asyncio.run(scenario())
