"""CPU execution and ordinary manifest hosting for Stable Cascade."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.stable_cascade import (
    StableCascade_EmptyLatentImage,
    StableCascade_StageB_Conditioning,
    StableCascade_StageC_VAEEncode,
    StableCascade_SuperResolutionControlnet,
)

from dinkster.compose import PackSpec, ServingComposer


def test_latent_shapes_and_compression_rounding() -> None:
    stages = StableCascade_EmptyLatentImage.execute(
        width=264, height=280, compression=42, batch_size=2
    )
    assert stages["stage_c"]["samples"].shape == (2, 16, 6, 6)
    assert stages["stage_b"]["samples"].shape == (2, 4, 70, 66)
    assert not stages["stage_c"]["samples"].count_nonzero()
    assert not stages["stage_b"]["samples"].count_nonzero()
    assert stages["stage_c"]["samples"].dtype == torch.float32


class CascadeVAE:
    downscale_ratio = 4

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.mean(-1, keepdim=True).movedim(-1, 1)[:, :, ::4, ::4].repeat(1, 16, 1, 1)


def test_vae_encoding_preserves_crop_rgb_and_stage_dimensions() -> None:
    # Deliberately non-divisible dimensions distinguish stage B's two rounding rules.
    image = np.zeros((2, 35, 51, 4), dtype=np.float32)
    image[..., :3] = 0.25
    image[..., 3] = 0.9
    before = image.copy()
    encoded = StableCascade_StageC_VAEEncode.execute(image=image, vae=CascadeVAE(), compression=8)
    assert encoded["stage_c"]["samples"].shape == (2, 16, 4, 6)
    torch.testing.assert_close(encoded["stage_c"]["samples"], torch.full((2, 16, 4, 6), 0.25))
    assert encoded["stage_b"]["samples"].shape == (2, 4, 8, 12)
    control = StableCascade_SuperResolutionControlnet.execute(image=image, vae=CascadeVAE())
    assert control["controlnet_input"].shape == (2, 9, 13, 16)
    torch.testing.assert_close(control["controlnet_input"], torch.full((2, 9, 13, 16), 0.25))
    assert control["stage_c"]["samples"].shape == (2, 16, 2, 3)
    assert control["stage_b"]["samples"].shape == (2, 4, 17, 25)
    assert not control["stage_c"]["samples"].count_nonzero()
    assert not control["stage_b"]["samples"].count_nonzero()
    np.testing.assert_array_equal(image, before)


def test_prior_conditioning_materializes_without_source_mutation() -> None:
    samples = np.arange(96, dtype=np.float32).reshape(1, 16, 2, 3)
    text = torch.ones(1, 2, 4)
    conditioning = [[text, {"strength": 0.7}], [text + 1, {"other": True}]]
    result = StableCascade_StageB_Conditioning.execute(
        conditioning=conditioning, stage_c={"samples": samples}
    )["CONDITIONING"]
    assert result[0][0] is text
    assert result[0][1]["strength"] == 0.7
    assert result[1][1]["other"] is True
    torch.testing.assert_close(result[1][1]["stable_cascade_prior"], torch.from_numpy(samples))
    assert conditioning[0][1] == {"strength": 0.7}
    assert conditioning[1][1] == {"other": True}


class CascadeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cascade-fixture.source",
            outputs=(
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            conditioning=[[np.ones((1, 2, 4), dtype=np.float32), {"strength": 0.7}]],
            image=np.full((2, 35, 51, 3), 0.25, dtype=np.float32),
            vae=CascadeVAE(),
        )


class CascadeInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cascade-fixture.inspect",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("stage_b", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("encoded", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("control", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Any, stage_b: Any, encoded: Any, control: Any
    ) -> dict[str, object]:
        prior = materialize_latent({"samples": conditioning[0][1]["stable_cascade_prior"]})[
            "samples"
        ]
        b = materialize_latent(stage_b)["samples"]
        prior, b = map(materialize_image, (prior, b))
        encoded = materialize_image(materialize_latent(encoded)["samples"])
        control = materialize_image(control)
        assert not prior.count_nonzero() and not b.count_nonzero()
        torch.testing.assert_close(encoded, torch.full((2, 16, 4, 6), 0.25))
        torch.testing.assert_close(control, torch.full((2, 9, 13, 16), 0.25))
        return cls.outputs(
            summary=f"{tuple(prior.shape)};{tuple(b.shape)};{conditioning[0][1]['strength']}"
        )


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_latents_and_prior_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "cascade_fixture.py").write_text(
                "from test_stable_cascade_torch import CascadeSource, CascadeInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, CascadeSource, CascadeInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["cascade-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "cascade_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "cascade_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "cascade_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("cascade-fixture.source"),
                    "empty": GraphNode(
                        "StableCascade_EmptyLatentImage",
                        {"width": 264, "height": 280, "compression": 42, "batch_size": 2},
                    ),
                    "prior": GraphNode(
                        "StableCascade_StageB_Conditioning",
                        {
                            "conditioning": Link("source", "conditioning"),
                            "stage_c": Link("empty", "stage_c"),
                        },
                    ),
                    "encode": GraphNode(
                        "StableCascade_StageC_VAEEncode",
                        {
                            "image": Link("source", "image"),
                            "vae": Link("source", "vae"),
                            "compression": 8,
                        },
                    ),
                    "control": GraphNode(
                        "StableCascade_SuperResolutionControlnet",
                        {"image": Link("source", "image"), "vae": Link("source", "vae")},
                    ),
                    "inspect": GraphNode(
                        "cascade-fixture.inspect",
                        {
                            "conditioning": Link("prior", "CONDITIONING"),
                            "stage_b": Link("empty", "stage_b"),
                            "encoded": Link("encode", "stage_c"),
                            "control": Link("control", "controlnet_input"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "(2, 16, 6, 6);(2, 4, 70, 66);0.7"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
