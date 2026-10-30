"""Mage Flow edit conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
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
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


class TextEncodeMageFlowEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeMageFlowEdit",
            category="model/conditioning/mage",
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
                InputSpec("vae", _VAE, required=False),
                InputSpec("width", _INT, default=0, widget=NumberWidget(min=0, max=8192, step=16)),
                InputSpec("height", _INT, default=0, widget=NumberWidget(min=0, max=8192, step=16)),
                InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE, required=False),),
                    min_members=0,
                    member_names=tuple(f"image_{index}" for index in range(1, 17)),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
            input_order=(
                "clip",
                "prompt",
                "negative_prompt",
                "vae",
                "images",
                "width",
                "height",
                "batch_size",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        negative_prompt: str = "",
        vae: Any = None,
        images: Mapping[str, object] | None = None,
        width: int = 0,
        height: int = 0,
        batch_size: int = 1,
    ) -> Mapping[str, object]:
        references = [
            materialize_image(image)
            for _, image in sorted(
                (images or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])
            )
            if image is not None
        ]
        if width == 0 or height == 0:
            source_height, source_width = references[0].shape[1:3] if references else (1024, 1024)
            width = width or source_width
            height = height or source_height
        width = max(16, width // 16 * 16)
        height = max(16, height // 16 * 16)

        images_vl: list[Any] = []
        reference_latents: list[Any] = []
        if references:
            initialize_inference_process()
            common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
            for image in references:
                samples = image.movedim(-1, 1)
                image_height, image_width = samples.shape[2:]
                long_edge = max(image_width, image_height)
                if long_edge > 384:
                    scale = 384 / long_edge
                    images_vl.append(
                        common_upscale(
                            samples,
                            max(1, round(image_width * scale)),
                            max(1, round(image_height * scale)),
                            "bicubic",
                            "disabled",
                        ).movedim(1, -1)
                    )
                else:
                    images_vl.append(image)
                if vae is not None:
                    if image_width != width or image_height != height:
                        samples = common_upscale(samples, width, height, "bicubic", "disabled")
                    reference_latents.append(vae.encode(samples.movedim(1, -1)[..., :3]))

        positive: Any = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl))
        negative: Any = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt or " ", images=images_vl)
        )
        if reference_latents:
            values = {"reference_latents": reference_latents}
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

        torch = importlib.import_module("torch")
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": torch.zeros((batch_size, 128, height // 16, width // 16))},
        )
