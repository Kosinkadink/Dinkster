"""Trellis vision preprocessing and conditioning through ordinary native hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

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
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.trellis_conditioning import Trellis2Conditioning

from dinkster.compose import PackSpec, ServingComposer


class TinyVision(torch.nn.Module):
    def forward(self, pixels: Any, *, skip_norm_elementwise: bool) -> Any:
        channels = pixels.mean(dim=(-2, -1))
        tokens = torch.stack((channels, channels + pixels.shape[-1] / 512), dim=1)
        return (tokens if skip_norm_elementwise else torch.zeros_like(tokens),)


class Vision:
    def __init__(self, custom: bool = True) -> None:
        self.model = TinyVision()
        self.patcher = object()
        self.image_mean = [0.25, 0.5, 0.75] if custom else None
        self.image_std = [0.5, 0.25, 0.125] if custom else None


def source_image() -> Any:
    # Hand-derived: alpha is ignored, RGB clamps to red and green. The two
    # differently ordered batch items must stay separate at both resolutions.
    return (
        torch.tensor([[2.0, -1.0, 0.0, 0.3], [0.0, 1.0, 0.0, 0.9]])[:, None, None, :]
        .expand(2, 7, 11, 4)
        .clone()
    )


def check_conditioning(positive: Any, negative: Any, custom: bool = True) -> None:
    # Hand-derived normalization of red/green; resolution contributes only to
    # token two in the controlled encoder. No implementation-derived goldens.
    if custom:
        channels = torch.tensor([[1.5, -2.0, -6.0], [-0.5, 2.0, -6.0]])
    else:
        channels = (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) - torch.tensor([0.485, 0.456, 0.406])
        ) / torch.tensor([0.229, 0.224, 0.225])
    for value, offset in ((positive[0][0], 1), (positive[0][1]["embeds"], 2)):
        assert value.shape == (2, 2, 3) and value.dtype == torch.float32
        torch.testing.assert_close(value, torch.stack((channels, channels + offset), dim=1))
    assert torch.count_nonzero(negative[0][0]) == 0
    assert torch.count_nonzero(negative[0][1]["embeds"]) == 0


@pytest.mark.parametrize("custom", (False, True))
def test_batch_normalization_resolution_tokens_and_independent_negatives(
    monkeypatch: pytest.MonkeyPatch, custom: bool
) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    monkeypatch.setattr(management, "load_model_gpu", lambda _patcher: None)
    image = source_image().double()
    before = image.clone()
    result = Trellis2Conditioning.execute(clip_vision_model=Vision(custom), image=image.numpy())
    check_conditioning(result["positive"], result["negative"], custom)
    result["negative"][0][0].fill_(99)
    assert not torch.any(result["positive"][0][0] == 99)
    assert torch.count_nonzero(result["negative"][0][1]["embeds"]) == 0
    assert torch.equal(image, before)


class VisionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-vision-fixture.source",
            outputs=(
                OutputSpec("vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        import dinkster_inference.model_management as management

        management.load_model_gpu = lambda _patcher: None
        return cls.outputs(vision=Vision(), image=source_image().numpy())


class VisionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-vision-fixture.inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any) -> Any:
        check_conditioning(materialize_latent(positive), materialize_latent(negative))
        return cls.outputs(summary="dual resolution vision conditioning transported")


@pytest.mark.parametrize("in_process", (False, True))
def test_native_manifest_conditions_twice(tmp_path: Path, in_process: bool) -> None:
    initialize_inference_process()
    import dinkster_inference.model_management as management

    original_load = management.load_model_gpu

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "trellis_vision_fixture.py").write_text(
            "from test_trellis_conditioning_torch import VisionSource, VisionInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (VisionSource, VisionInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "vision.toml"
        fixture.write_text(
            '[pack]\nname="trellis-vision-fixture"\npublisher="dinkster"\n'
            'namespaces=["trellis-vision-fixture"]\n[pack.entry]\n'
            'nodes="trellis_vision_fixture:NODES"\n'
            'types="trellis_vision_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "trellis-vision-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("trellis-vision-fixture.source"),
                    "conditioning": GraphNode(
                        "Trellis2Conditioning",
                        {
                            "clip_vision_model": Link("source", "vision"),
                            "image": Link("source", "image"),
                        },
                    ),
                    "inspect": GraphNode(
                        "trellis-vision-fixture.inspect",
                        {name: Link("conditioning", name) for name in ("positive", "negative")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == (
                    "dual resolution vision conditioning transported"
                )
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        management.load_model_gpu = original_load
