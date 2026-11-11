"""Image-edit and reference-latent conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    with_identity_replacement,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


class InstructPixToPixConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.misc.models.instruct_pix_to_pix_conditioning",
            category="model/conditioning/instructpix2pix",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("pixels", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
        )
        return with_identity_replacement(schema, "InstructPixToPixConditioning")

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        pixels: object,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image: Any = materialize_image(pixels)
        height = (image.shape[1] // 8) * 8
        width = (image.shape[2] // 8) * 8
        if image.shape[1] != height or image.shape[2] != width:
            top = (image.shape[1] % 8) // 2
            left = (image.shape[2] % 8) // 2
            image = image[:, top : height + top, left : width + left, :]
        encoded = vae.encode(image)
        return cls.outputs(
            positive=[
                [text, {**metadata, "concat_latent_image": encoded}] for text, metadata in positive
            ],
            negative=[
                [text, {**metadata, "concat_latent_image": encoded}] for text, metadata in negative
            ],
            latent={"samples": torch.zeros_like(encoded)},
        )
