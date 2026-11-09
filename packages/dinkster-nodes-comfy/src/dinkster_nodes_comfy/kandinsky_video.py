"""Native Kandinsky 5 image-to-video conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
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

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


class Kandinsky5ImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Kandinsky5ImageToVideo",
            category="model/conditioning/kandinsky",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "width", _INT, default=768, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", _INT, default=512, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "length", _INT, default=121, widget=NumberWidget(min=1, max=16384, step=4)
                ),
                InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
                InputSpec("start_image", TypeExpr.concrete(IMAGE_TYPE), required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT, doc="Empty video latent"),
                OutputSpec(
                    "cond_latent",
                    _LATENT,
                    doc=(
                        "Clean encoded start images, used to replace the noisy start "
                        "of the model output latents"
                    ),
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
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
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        clean: dict[str, object] = {}
        if start_image is not None:
            image = materialize_image(start_image)[:length]
            utils = importlib.import_module("dinkster_inference.utils")
            image = utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            encoded = vae.encode(image[:, :, :, :3])
            clean["samples"] = encoded
            mask = torch.ones(
                (1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
                device=image.device,
                dtype=image.dtype,
            )
            mask[:, :, : ((image.shape[0] - 1) // 4) + 1] = 0.0
            values = {"time_dim_replace": encoded, "concat_mask": mask}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(
            positive=positive, negative=negative, latent={"samples": latent}, cond_latent=clean
        )
