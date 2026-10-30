"""Bernini in-context video and image conditioning."""

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
    InputFamilySpec,
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
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _integer(name: str, default: int, minimum: int, maximum: int, step: int | None) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _resize_long_edge(image: object, max_size: int) -> Any:
    pixels = materialize_image(image)
    height, width = pixels.shape[1:3]
    scale = min(max_size / max(height, width), 1.0)
    resized_height = max(16, round(height * scale / 16) * 16)
    resized_width = max(16, round(width * scale / 16) * 16)
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(
        pixels[..., :3].movedim(-1, 1), resized_width, resized_height, "area", "disabled"
    ).movedim(1, -1)


class BerniniConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BerniniConditioning",
            display_name="Bernini Conditioning",
            category="model/conditioning/bernini",
            description=(
                "Conditioning node for Bernini in-context video/image conditioning. It can be used "
                "for t2v, v2v, rv2v, r2v, and ads2v. Reference images are encoded at their own "
                "native aspect ratio."
            ),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 8192, 16),
                _integer("height", 480, 16, 8192, 16),
                _integer("length", 81, 1, 8192, 4),
                _integer("batch_size", 1, 1, 4096, None),
                InputSpec("source_video", _IMAGE, required=False),
                InputSpec("reference_video", _IMAGE, required=False),
                InputSpec(
                    "ref_max_size",
                    _INT,
                    default=848,
                    required=False,
                    widget=NumberWidget(min=16, max=8192, step=16),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "reference_images",
                    (InputSpec("reference_image", _IMAGE, required=False),),
                    min_members=0,
                    max_members=8,
                    required=False,
                    member_prefix="reference_image_",
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
            input_order=(
                "positive",
                "negative",
                "vae",
                "width",
                "height",
                "length",
                "batch_size",
                "source_video",
                "reference_video",
                "reference_images",
                "ref_max_size",
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
        source_video: object | None = None,
        reference_video: object | None = None,
        reference_images: Mapping[str, object] | None = None,
        ref_max_size: int = 848,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        context: list[Any] = []
        utils = importlib.import_module("dinkster_inference.utils")
        if source_video is not None:
            source = materialize_image(source_video)[:length, :, :, :3]
            source = utils.common_upscale(
                source.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            context.append(vae.encode(source))
        if reference_video is not None:
            context.append(
                vae.encode(
                    _resize_long_edge(materialize_image(reference_video)[:length], ref_max_size)
                )
            )
        for name in sorted(reference_images or {}):
            images = (reference_images or {})[name]
            if images is None:
                continue
            pixels = materialize_image(images)
            for index in range(pixels.shape[0]):
                context.append(
                    vae.encode(_resize_long_edge(pixels[index : index + 1], ref_max_size))
                )
        if not context:
            return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})
        values = {"context_latents": context}
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
            latent={"samples": latent},
        )


BERNINI_CONDITIONING_NODES = (BerniniConditioning,)
