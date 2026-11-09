"""Native SD 4x upscaler conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


class SD_4XUpscale_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SD_4XUpscale_Conditioning",
            display_name="SD_4XUpscale_Conditioning",
            category="model/conditioning/stable diffusion upscaler",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                _float("scale_ratio", 4.0, 0.0, 10.0, 0.01),
                _float("noise_augmentation", 0.0, 0.0, 1.0, 0.001, advanced=True),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        scale_ratio: float = 4.0,
        noise_augmentation: float = 0.0,
    ) -> Mapping[str, object]:
        image = materialize_image(images)
        width = max(1, round(image.shape[-2] * scale_ratio))
        height = max(1, round(image.shape[-3] * scale_ratio))
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        pixels = utils.common_upscale(
            image.movedim(-1, 1) * 2.0 - 1.0, width // 4, height // 4, "bilinear", "center"
        )
        torch = importlib.import_module("torch")
        values = {"concat_image": pixels, "noise_augmentation": noise_augmentation}
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
            latent={"samples": torch.zeros((image.shape[0], 4, height // 4, width // 4))},
        )


SD_UPSCALE_CONDITIONING_NODES = (SD_4XUpscale_Conditioning,)
