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
    materialize_latent,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


class InstructPixToPixConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="InstructPixToPixConditioning",
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


class ReferenceLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ReferenceLatent",
            display_name="Set Reference Latent",
            category="model/conditioning",
            description=(
                "This node sets the guiding latent for an edit model. If the model supports it "
                "you can chain multiple to set multiple reference images."
            ),
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("latent", _LATENT, required=False),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], latent: object = None
    ) -> Mapping[str, object]:
        if latent is None:
            return cls.outputs(CONDITIONING=conditioning)
        samples = materialize_image(materialize_latent(latent)["samples"])
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            references = metadata.get("reference_latents")
            values = [samples] if references is None else references + [samples]
            result.append([text, {**metadata, "reference_latents": values}])
        return cls.outputs(CONDITIONING=result)


REFERENCE_CONDITIONING_NODES = (InstructPixToPixConditioning, ReferenceLatent)
