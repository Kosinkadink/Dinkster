"""CPU tensor and ordinary-pack execution behavior for inpainting nodes."""

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
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.inpaint import InpaintModelConditioning, VAEEncodeForInpaint

from dinkster.compose import PackSpec, ServingComposer

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class FakeVAE:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []

    def spacial_compression_encode(self) -> int:
        return 4

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encoded.append(pixels.clone())
        return pixels.movedim(-1, 1) + len(self.encoded)


def test_inpaint_encoders_crop_grow_and_preserve_sources() -> None:
    pixels = torch.arange(2 * 7 * 11 * 3, dtype=torch.float32).reshape(2, 7, 11, 3) / 255
    mask = torch.zeros((2, 3, 5), dtype=torch.float32)
    mask[:, 1, 2] = 1
    pixels_before, mask_before = pixels.clone(), mask.clone()
    vae = FakeVAE()
    inpaint = VAEEncodeForInpaint.execute(
        vae=vae, pixels=pixels.numpy(), mask=mask.numpy(), grow_mask_by=0
    )["LATENT"]
    assert inpaint["samples"].shape == (2, 3, 4, 8)
    assert inpaint["noise_mask"].shape == (2, 1, 4, 8)
    assert torch.equal(inpaint["noise_mask"], inpaint["noise_mask"].round())
    grown = VAEEncodeForInpaint.execute(
        vae=FakeVAE(), pixels=pixels.numpy(), mask=mask.numpy(), grow_mask_by=3
    )["LATENT"]
    assert grown["noise_mask"].shape == (2, 1, 4, 8)
    assert torch.count_nonzero(grown["noise_mask"] - inpaint["noise_mask"]) > 0
    assert torch.equal(pixels, pixels_before)
    assert torch.equal(mask, mask_before)


def test_inpaint_values_use_center_crop_and_only_masked_pixels() -> None:
    pixels = torch.arange(7 * 11 * 3, dtype=torch.float32).reshape(1, 7, 11, 3) / 256
    mask = torch.zeros((1, 7, 11), dtype=torch.float32)
    mask[0, 2, 3] = 1
    result = VAEEncodeForInpaint.execute(
        vae=FakeVAE(), pixels=pixels.numpy(), mask=mask.numpy(), grow_mask_by=3
    )["LATENT"]
    # A 4x8 crop starts at (1,1); the mask lands at (1,2).
    expected_mask = torch.zeros((1, 1, 4, 8))
    expected_mask[0, 0, :3, 1:4] = 1
    assert torch.equal(result["noise_mask"], expected_mask)
    assert torch.equal(result["samples"][0, :, 1, 2], torch.full((3,), 1.5))
    assert torch.equal(
        result["samples"][0, :, 0, 0],
        torch.tensor([1 + 36 / 256, 1 + 37 / 256, 1 + 38 / 256]),
    )


@pytest.mark.parametrize("noise_mask", (False, True))
def test_inpaint_conditioning_keeps_sources_and_sets_metadata(noise_mask: bool) -> None:
    pixels = torch.arange(11 * 13 * 3, dtype=torch.float32).reshape(1, 11, 13, 3) / 511
    mask = torch.zeros((1, 3, 5), dtype=torch.float32)
    mask[:, 1, 2] = 1
    text = torch.tensor([[[3.0]]])
    positive = [[text, {"tag": "positive"}]]
    negative = [[text + 1, {"tag": "negative"}]]
    vae = FakeVAE()
    result = InpaintModelConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=pixels.numpy(),
        mask=mask.numpy(),
        noise_mask=noise_mask,
    )
    assert len(vae.encoded) == 2
    assert vae.encoded[0].shape == (1, 8, 8, 3)
    assert vae.encoded[1].shape == pixels.shape
    assert positive[0][1] == {"tag": "positive"}
    assert negative[0][1] == {"tag": "negative"}
    for name in ("positive", "negative"):
        metadata = result[name][0][1]
        assert metadata["concat_latent_image"] is not vae.encoded[0]
        assert metadata["concat_mask"].shape == (1, 1, 8, 8)
    latent = result["latent"]
    assert ("noise_mask" in latent) is noise_mask
    assert latent["samples"].shape == (1, 3, 11, 13)


class InpaintSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="inpaint-fixture.source",
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("vae", _VAE),
                OutputSpec("pixels", _IMAGE),
                OutputSpec("mask", _MASK),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[[torch.ones((1, 1, 1)), {"tag": "positive"}]],
            negative=[[torch.zeros((1, 1, 1)), {"tag": "negative"}]],
            vae=FakeVAE(),
            pixels=np.arange(11 * 13 * 3, dtype=np.float32).reshape(1, 11, 13, 3) / 511,
            mask=np.array([[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=np.float32),
        )


class InpaintInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="inpaint-fixture.inspect",
            inputs=(InputSpec("image", _IMAGE), InputSpec("mask", _MASK)),
            outputs=(OutputSpec("pixel", _FLOAT), OutputSpec("mask", _FLOAT)),
        )

    @classmethod
    def execute(cls, *, image: Any, mask: Any) -> dict[str, object]:
        return cls.outputs(
            pixel=float(materialize_image(image)[0, 0, 0, 0]),
            mask=float(materialize_image(mask)[0, 0, 0]),
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_executes_inpainting_nodes_with_decoded_inputs(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(__file__).parent),
                        str(source_root),
                    )
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "inpaint_fixture.py").write_text(
                "from test_inpaint_torch import InpaintInspect, InpaintSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, InpaintSource, InpaintInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["inpaint-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "inpaint_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "inpaint_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("inpaint-fixture.source"),
                    "encode": GraphNode(
                        "VAEEncodeForInpaint",
                        {
                            "vae": Link("source", "vae"),
                            "pixels": Link("source", "pixels"),
                            "mask": Link("source", "mask"),
                            "grow_mask_by": 2,
                        },
                    ),
                    "inspect": GraphNode(
                        "inpaint-fixture.inspect",
                        {"image": Link("source", "pixels"), "mask": Link("source", "mask")},
                    ),
                    "conditioning": GraphNode(
                        "InpaintModelConditioning",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "pixels": Link("source", "pixels"),
                            "mask": Link("source", "mask"),
                            "noise_mask": True,
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["encode", "inspect", "conditioning"])
                assert result.outputs["inspect"]["pixel"].resolve() == 0.0
                assert result.outputs["inspect"]["mask"].resolve() == 0.0
                assert result.outputs["encode"]["LATENT"].resolve()["noise_mask"].shape == (
                    1,
                    1,
                    8,
                    12,
                )
                assert result.outputs["conditioning"]["latent"].resolve()["noise_mask"].shape == (
                    1,
                    1,
                    8,
                    8,
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
