"""Boogu Image edit conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

from .conditioning import conditioning_set_values

_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


class TextEncodeBooguEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeBooguEdit",
            category="model/conditioning/boogu",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "negative_prompt",
                    _STRING,
                    advanced=True,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("vae", _VAE),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE, required=False),),
                    min_members=0,
                    member_names=tuple(f"image_{index}" for index in range(1, 17)),
                ),
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("negative", _CONDITIONING)),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        negative_prompt: str,
        vae: Any = None,
        images: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        references = [
            materialize_image(image)
            for _, image in sorted(
                (images or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])
            )
            if image is not None
        ]
        vision: list[Any] = []
        latents: list[Any] = []
        if references:
            initialize_inference_process()
            common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
            for image in references:
                samples = image.movedim(-1, 1)
                height, width = samples.shape[2:]
                vision_scale = math.sqrt(384 * 384 / (width * height))
                vision.append(
                    common_upscale(
                        samples,
                        round(width * vision_scale),
                        round(height * vision_scale),
                        "area",
                        "disabled",
                    ).movedim(1, -1)[..., :3]
                )
                if vae is not None:
                    latent_scale = math.sqrt(1_048_576 / (width * height))
                    latent_width = round(width * latent_scale / 16.0) * 16
                    latent_height = round(height * latent_scale / 16.0) * 16
                    latents.append(
                        vae.encode(
                            common_upscale(
                                samples, latent_width, latent_height, "area", "disabled"
                            ).movedim(1, -1)[..., :3]
                        )
                    )

        positive: Any = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=vision))
        negative: Any = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        if latents:
            values = {"reference_latents": latents}
            positive = cast(
                "Any",
                conditioning_set_values(
                    positive,
                    values,
                    append=True,
                ),
            )
            negative = cast(
                "Any",
                conditioning_set_values(
                    negative,
                    values,
                    append=True,
                ),
            )
        return cls.outputs(positive=positive, negative=negative)
