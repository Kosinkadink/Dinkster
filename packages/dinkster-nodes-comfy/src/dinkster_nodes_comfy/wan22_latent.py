"""Wan 2.2 image-to-video latent conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class Wan22ImageToVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Wan22ImageToVideoLatent",
            category="model/conditioning/wan",
            inputs=(
                InputSpec("vae", _VAE),
                InputSpec(
                    "width", _INT, default=1280, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec(
                    "height", _INT, default=704, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec(
                    "length", _INT, default=49, widget=NumberWidget(min=1, max=16384, step=4)
                ),
                InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
                InputSpec("start_image", _IMAGE, required=False),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            (1, 48, ((length - 1) // 4) + 1, height // 16, width // 16),
            device=management.intermediate_device(),
        )
        if start_image is None:
            return cls.outputs(LATENT={"samples": latent})

        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(start_image)[:length]
        image = utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        latent_temp = vae.encode(image)
        mask = torch.ones(
            (1, 1, latent.shape[-3], latent.shape[-2], latent.shape[-1]),
            device=management.intermediate_device(),
        )
        latent[:, :, : latent_temp.shape[-3]] = latent_temp
        mask[:, :, : latent_temp.shape[-3]] = 0.0
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        latent = latent_formats.Wan22().process_out(latent) * mask + latent * (1.0 - mask)
        return cls.outputs(
            LATENT={
                "samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1)),
                "noise_mask": mask.repeat((batch_size,) + (1,) * (mask.ndim - 1)),
            }
        )
