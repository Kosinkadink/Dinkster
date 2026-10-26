"""Autoregressive start images and model options through ordinary CPU hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ar_video import ARVideoI2V

from dinkster.compose import PackSpec, ServingComposer


class Encoder:
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        assert pixels.shape == (1, 16, 32, 3)
        return pixels[:, ::8, ::8, :1].permute(0, 3, 1, 2).unsqueeze(2)


def image() -> torch.Tensor:
    return torch.arange(2 * 4 * 8 * 4, dtype=torch.float32).reshape(2, 4, 8, 4) / 128


def model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    patcher = ModelPatcher(torch.nn.Linear(2, 3), torch.device("cpu"), torch.device("cpu"))
    patcher.model_options["transformer_options"] = {
        "preserved": 19,
        "ar_config": {"frames": 3, "initial_latent": "replace-me"},
    }
    return patcher


def check(patched: Any, latent: object, frames: int) -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    assert type(patched) is ModelPatcher
    assert patched.model_options["transformer_options"]["preserved"] == 19
    ar_config = patched.model_options["transformer_options"]["ar_config"]
    assert ar_config["frames"] == 3
    # Independent bilinear samples at output (y,x)=(0,0),(0,8),...,(8,24).
    expected = torch.tensor([[0, 6.5, 14.5, 22.5], [52, 58.5, 66.5, 74.5]]) / 128
    encoded = ar_config["initial_latent"]
    torch.testing.assert_close(encoded, expected.reshape(1, 1, 1, 2, 4), rtol=0, atol=0)
    assert encoded.dtype == torch.float32 and encoded.device.type == "cpu"
    samples = materialize_latent(latent)["samples"]
    assert samples.shape == (2, 16, frames, 2, 4)
    assert samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()


@pytest.mark.parametrize(("length", "frames"), ((1, 1), (8, 2), (9, 3)))
def test_start_image_conditions_clone_without_modifying_sources(length: int, frames: int) -> None:
    source, pixels = model(), image()
    before = pixels.clone()
    result = ARVideoI2V.execute(
        model=source,
        vae=Encoder(),
        start_image=pixels,
        width=32,
        height=16,
        length=length,
        batch_size=2,
    )
    assert result["MODEL"] is not source
    check(result["MODEL"], result["LATENT"], frames)
    assert source.model_options["transformer_options"] == {
        "preserved": 19,
        "ar_config": {"frames": 3, "initial_latent": "replace-me"},
    }
    assert torch.equal(pixels, before)


class ARImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ar-image-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=model(), vae=Encoder(), image=image())


class ARImageConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ar-image-fixture.consumer",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, latent: object) -> dict[str, object]:
        check(model, latent, 3)
        return cls.outputs(summary="ar-image-verified")


@pytest.mark.parametrize("in_process", (True, False))
def test_ar_image_conditioning_runs_twice_through_ordinary_hosts(
    tmp_path: Path,
    in_process: bool,
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        module = "ar_image_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_ar_video_conditioning_torch import ARImageSource, ARImageConsumer\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (ARImageSource, ARImageConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="ar-image-fixture"\npublisher="dinkster"\n'
            'namespaces=["ar-image-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "ar-image-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("ar-image-fixture.source", {}),
                    "condition": GraphNode(
                        "ARVideoI2V",
                        {
                            "model": Link("source", "model"),
                            "vae": Link("source", "vae"),
                            "start_image": Link("source", "image"),
                            "width": 32,
                            "height": 16,
                            "length": 9,
                            "batch_size": 2,
                        },
                    ),
                    "consumer": GraphNode(
                        "ar-image-fixture.consumer",
                        {
                            "model": Link("condition", "MODEL"),
                            "latent": Link("condition", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "ar-image-verified"
        finally:
            await composer.close()

    asyncio.run(scenario())
