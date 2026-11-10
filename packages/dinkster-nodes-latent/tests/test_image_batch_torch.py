"""Public CPU image dimension and rebatch behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_latent.rebatch import RebatchImages, RebatchLatents


def test_empty_image_initializes_cpu_policy_in_a_cold_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_image.ported.image_core import EmptyImage\n"
            "assert 'dinkster_inference.model_management' not in sys.modules\n"
            "image = EmptyImage.execute(\n"
            "    width=73, height=41, batch_size=2, color=0x32B76D)['IMAGE']\n"
            "assert image.device.type == 'cpu'\n"
            "assert tuple(image.shape) == (2, 41, 73, 3)\n"
            "import torch\n"
            "expected = torch.tensor([50 / 255, 183 / 255, 109 / 255], dtype=image.dtype)\n"
            "torch.testing.assert_close(image, expected.expand_as(image), rtol=0, atol=0)\n",
        ],
        env={**os.environ, "DINKSTER_ACCELERATOR": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rebatch_preserves_frame_order_and_latent_indices() -> None:
    first = torch.arange(2 * 3 * 5 * 3, dtype=torch.float32).reshape(2, 3, 5, 3)
    second = torch.full((1, 3, 5, 3), 123.0)
    images = RebatchImages.execute(images=[first, second], batch_size=2)["IMAGE"]
    assert len(images) == 2
    assert torch.equal(images[0], first)
    assert torch.equal(images[1], second)

    samples = torch.arange(3 * 4 * 2 * 5, dtype=torch.float32).reshape(3, 4, 2, 5)
    latents = RebatchLatents.execute(
        latents=[{"samples": samples, "batch_index": [8, 3, 11]}], batch_size=2
    )["LATENT"]
    assert [item["batch_index"] for item in latents] == [[8, 3], [11]]
    assert torch.equal(latents[0]["samples"], samples[:2])
    assert torch.equal(latents[1]["samples"], samples[2:])
    assert all("noise_mask" not in item for item in latents)

    mask = torch.full((1, 1, 2, 3), 0.25)
    masked = RebatchLatents.execute(
        latents=[{"samples": torch.zeros((2, 4, 2, 3)), "noise_mask": mask}], batch_size=2
    )["LATENT"]
    assert torch.equal(masked[0]["noise_mask"], mask.repeat(2, 1, 1, 1))
    assert mask.shape == (1, 1, 2, 3)


class DimensionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-dimension-fixture.source",
            inputs=(InputSpec("video", TypeExpr.concrete(CORE_BOOLEAN)),),
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, video: bool) -> Any:
        return cls.outputs(
            image=torch.zeros((2, 3, 7, 4)),
            latent={"samples": torch.zeros((3, 4, 5, 6, 9) if video else (3, 4, 6, 9))},
        )


class RebatchSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "rebatch-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))),
                OutputSpec("latents", TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))),
                OutputSpec("sizes", TypeExpr.list_of(TypeExpr.concrete(CORE_INT))),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(
            images=[torch.zeros((1, 3, 7, 3)), torch.ones((2, 3, 7, 3))],
            latents=[
                {
                    "samples": torch.zeros((1, 4, 2, 5)),
                    "batch_index": [8],
                },
                {
                    "samples": torch.ones((2, 4, 2, 5)),
                    "batch_index": [3, 11],
                },
            ],
            sizes=[2, 1],
        )
