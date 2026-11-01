"""Public CPU image-upscale contracts with a controlled resident model."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    UPSCALE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_model_upscale import ImageUpscaleWithModel

from dinkster.compose import PackSpec, ServingComposer


class TinyUpscaler(torch.nn.Conv2d):
    def __init__(self, scale: int) -> None:
        super().__init__(3, 3, 1)
        self.scale = scale
        with torch.no_grad():
            self.weight.zero_()
            for channel in range(3):
                self.weight[channel, channel, 0, 0] = 0.5
            self.bias.copy_(torch.tensor([-0.25, 0.25, 0.5]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(value)
            .repeat_interleave(self.scale, -2)
            .repeat_interleave(self.scale, -1)
        )


class TinyDescriptor:
    def __init__(self, scale: int = 2, max_width: int | None = None, broken: bool = False):
        initialize_inference_process()
        from dinkster_inference.model_patcher import CoreModelPatcher

        self.model = TinyUpscaler(scale)
        self.patcher = CoreModelPatcher(self.model, torch.device("cpu"), torch.device("cpu"))
        self.scale = scale
        self.max_width = max_width
        self.broken = broken

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if self.broken:
            raise ValueError("controlled model failure")
        if self.max_width is not None and value.shape[-1] > self.max_width:
            raise torch.OutOfMemoryError("controlled tile capacity")
        return self.model(value)


def image_batch(channels: int = 4, width: int = 7) -> torch.Tensor:
    return (torch.arange(2 * 3 * width * channels).reshape(2, 3, width, channels) % 7) / 4 - 0.5


def expected_image(image: torch.Tensor, scale: int) -> torch.Tensor:
    # Hand-derived: diagonal half-gain convolution, channel biases, nearest replication;
    # alpha is bilinear and is not passed through the model or clamped with RGB.
    rgb = (image[..., :3].float() * 0.5 + torch.tensor([-0.25, 0.25, 0.5])).clamp(0, 1)
    rgb = rgb.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
    if image.shape[-1] == 4:
        alpha = torch.nn.functional.interpolate(
            image[..., 3:].movedim(-1, 1).float(),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)
        rgb = torch.cat((rgb, alpha), -1)
    return rgb


@pytest.mark.parametrize(
    "channels,scale,dtype", ((3, 1, torch.float32), (4, 2, torch.float16), (3, 3, torch.float32))
)
def test_upscale_preserves_model_values_alpha_and_input(channels, scale, dtype) -> None:
    source = image_batch(channels).to(dtype)
    before = source.clone()
    result = ImageUpscaleWithModel.execute(upscale_model=TinyDescriptor(scale), image=source)[
        "IMAGE"
    ]
    torch.testing.assert_close(result, expected_image(source, scale), rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert result.dtype is torch.float32


def test_upscale_retries_smaller_tiles_without_changing_values() -> None:
    source = image_batch(4, 517)
    result = ImageUpscaleWithModel.execute(
        upscale_model=TinyDescriptor(max_width=256), image=source
    )["IMAGE"]
    torch.testing.assert_close(result, expected_image(source, 2), rtol=0, atol=0)


def test_upscale_propagates_model_errors_and_exhausted_capacity() -> None:
    with pytest.raises(ValueError, match="controlled model failure"):
        ImageUpscaleWithModel.execute(
            upscale_model=TinyDescriptor(broken=True), image=image_batch()
        )
    with pytest.raises(torch.OutOfMemoryError, match="controlled tile capacity"):
        ImageUpscaleWithModel.execute(
            upscale_model=TinyDescriptor(max_width=127), image=image_batch(3, 517)
        )


class UpscaleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="upscale-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(UPSCALE_MODEL_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=TinyDescriptor(), image=image_batch())


class UpscaleConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="upscale-fixture.consumer",
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("original", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: object, original: object) -> dict[str, object]:
        torch.testing.assert_close(
            materialize_image(image), expected_image(image_batch(), 2), rtol=0, atol=0
        )
        torch.testing.assert_close(materialize_image(original), image_batch(), rtol=0, atol=0)
        return cls.outputs(summary="upscaled")


@pytest.mark.parametrize("in_process", (True, False))
def test_upscale_executes_and_caches_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "upscale_fixture.py").write_text(
            "from test_image_model_upscale_torch import UpscaleSource, UpscaleConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (UpscaleSource, UpscaleConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "upscale-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["upscale-fixture"]\n[pack.entry]\n'
            'nodes = "upscale_fixture:NODES"\ntypes = "upscale_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_ACCELERATOR": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(native.parent / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest, worker_group="upscale", group_manifests=(native, fixture)
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("upscale-fixture.source"),
                    "upscale": GraphNode(
                        "ImageUpscaleWithModel",
                        {
                            "upscale_model": Link("source", "model"),
                            "image": Link("source", "image"),
                        },
                    ),
                    "consumer": GraphNode(
                        "upscale-fixture.consumer",
                        {"image": Link("upscale", "IMAGE"), "original": Link("source", "image")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "upscaled"
                if _ == 1:
                    assert set(result.cached) == {"source", "upscale", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
