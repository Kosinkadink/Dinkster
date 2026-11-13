"""Empty model latents retain shape and metadata through ordinary pack hosts."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from dinkster_3d_models.latent_model_specific import EmptyTrellis2LatentStructure
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_flux.latent_model_specific import EmptyFlux2LatentImage
from dinkster_hidream.latent_model_specific import EmptyHiDreamO1LatentImage
from dinkster_hunyuan.latent_model_specific import EmptyLatentHunyuan3Dv2
from dinkster_misc_models.latent_model_specific import EmptyChromaRadianceLatentImage
from dinkster_qwen.latent_model_specific import EmptyQwenImageLayeredLatentImage
from dinkster_sd.latent_model_specific import EmptySD3LatentImage

CASES = (
    (
        EmptySD3LatentImage,
        {"width": 48, "height": 80, "batch_size": 2},
        (2, 16, 10, 6),
        {"downscale_ratio_spacial": 8},
    ),
    (EmptyFlux2LatentImage, {"width": 48, "height": 80, "batch_size": 2}, (2, 128, 5, 3), {}),
    (
        EmptyChromaRadianceLatentImage,
        {"width": 48, "height": 80, "batch_size": 2},
        (2, 3, 80, 48),
        {},
    ),
    (EmptyHiDreamO1LatentImage, {"width": 64, "height": 96, "batch_size": 2}, (2, 3, 96, 64), {}),
    (
        EmptyQwenImageLayeredLatentImage,
        {"width": 48, "height": 80, "layers": 2, "batch_size": 2},
        (2, 16, 3, 10, 6),
        {},
    ),
    (
        EmptyQwenImageLayeredLatentImage,
        {"width": 16, "height": 32, "layers": 0, "batch_size": 1},
        (1, 16, 1, 4, 2),
        {},
    ),
    (
        EmptyLatentHunyuan3Dv2,
        {"resolution": 17, "batch_size": 2},
        (2, 64, 17),
        {"type": "hunyuan3dv2"},
    ),
    (EmptyTrellis2LatentStructure, {"batch_size": 2}, (2, 32, 16, 16, 16), {"type": "trellis2"}),
)


@pytest.mark.parametrize(("node", "inputs", "shape", "metadata"), CASES)
def test_model_latent_layout(
    node: type[Node], inputs: dict[str, int], shape: tuple[int, ...], metadata: dict[str, Any]
) -> None:
    result = node.execute(**inputs)["LATENT"]
    samples = result["samples"]
    assert tuple(samples.shape) == shape
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert torch.count_nonzero(samples).item() == 0
    assert {key: value for key, value in result.items() if key != "samples"} == metadata
    with torch.inference_mode():
        samples.fill_(3)
    assert torch.count_nonzero(node.execute(**inputs)["LATENT"]["samples"]).item() == 0


class InspectLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-fixture.inspect",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: dict[str, Any]) -> dict[str, object]:
        samples = materialize_latent(latent)["samples"]
        return cls.outputs(
            text=json.dumps(
                {
                    "shape": list(samples.shape),
                    "zeros": torch.count_nonzero(samples).item() == 0,
                    "metadata": {k: v for k, v in latent.items() if k != "samples"},
                },
                sort_keys=True,
            )
        )


FIXTURES = (InspectLatent,)
