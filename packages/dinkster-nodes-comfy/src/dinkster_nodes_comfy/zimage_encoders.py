"""Z-Image Omni conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
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
_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


class TextEncodeZImageOmni(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeZImageOmni",
            category="model/conditioning/z-image",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec("image_encoder", _CLIP_VISION, required=False),
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "auto_resize_images",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    advanced=True,
                ),
                InputSpec("vae", _VAE, required=False),
                InputSpec("image1", _IMAGE, required=False),
                InputSpec("image2", _IMAGE, required=False),
                InputSpec("image3", _IMAGE, required=False),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        image_encoder: Any = None,
        auto_resize_images: bool = True,
        vae: Any = None,
        image1: object = None,
        image2: object = None,
        image3: object = None,
    ) -> Mapping[str, object]:
        images = [
            materialize_image(image) for image in (image1, image2, image3) if image is not None
        ]
        fragments: list[str] = []
        template: str | None = None
        if images:
            fragments = ["<|im_start|>user\n<|vision_start|>"]
            fragments.extend("<|vision_end|><|vision_start|>" for _ in range(len(images) - 1))
            fragments.append("<|vision_end|><|im_end|>")
            template = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
        encoded = (
            [image_encoder.encode_image(image) for image in images]
            if image_encoder is not None
            else []
        )
        references: list[Any] = []
        if vae is not None:
            for image in images:
                if auto_resize_images:
                    initialize_inference_process()
                    samples = image.movedim(-1, 1)
                    scale = math.sqrt(1024 * 1024 / (samples.shape[3] * samples.shape[2]))
                    width, height = (
                        round(samples.shape[3] * scale / 8) * 8,
                        round(samples.shape[2] * scale / 8) * 8,
                    )
                    image = (
                        importlib.import_module("dinkster_inference.utils")
                        .common_upscale(samples, width, height, "area", "disabled")
                        .movedim(1, -1)
                    )
                references.append(vae.encode(image))
        conditioning: Any = clip.encode_from_tokens_scheduled(
            clip.tokenize(prompt, llama_template=template)
        )
        embeds = [
            clip.encode_from_tokens_scheduled(clip.tokenize(fragment, llama_template="{}"))[0][0]
            for fragment in fragments
        ]
        if references:
            conditioning = cast(
                "Any",
                conditioning_set_values(
                    conditioning, {"reference_latents": references}, append=True
                ),
            )
        if encoded:
            conditioning = cast(
                "Any",
                conditioning_set_values(
                    conditioning, {"clip_vision_outputs": encoded}, append=True
                ),
            )
        if embeds:
            conditioning = cast(
                "Any",
                conditioning_set_values(
                    conditioning, {"reference_latents_text_embeds": embeds}, append=True
                ),
            )
        return cls.outputs(CONDITIONING=conditioning)
