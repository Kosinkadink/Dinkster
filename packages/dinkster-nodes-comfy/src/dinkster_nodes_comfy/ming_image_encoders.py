"""Ming Image edit conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

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


class TextEncodeMingImageEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeMingImageEdit",
            display_name="Text Encode Ming Image Edit",
            category="model/conditioning/ming image",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec("vae", _VAE, required=False),
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE, required=False),),
                    min_members=0,
                    max_members=8,
                    member_prefix="image_",
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
            input_order=("clip", "vae", "prompt", "images"),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        images: Mapping[str, object] | None = None,
        vae: Any = None,
    ) -> Mapping[str, object]:
        references = [
            materialize_image(image)
            for _, image in sorted(
                (images or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])
            )
            if image is not None
        ]
        conditioning = clip.encode_from_tokens_scheduled(
            clip.tokenize(prompt, images=[image[..., :3] for image in references])
        )
        if vae is None or not references:
            return cls.outputs(CONDITIONING=conditioning)

        initialize_inference_process()
        height, width = references[0].shape[1:3]
        common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
        latents = [
            vae.encode(
                common_upscale(image.movedim(-1, 1), width, height, "bilinear", "disabled").movedim(
                    1, -1
                )
            )
            for image in references
        ]
        return cls.outputs(
            CONDITIONING=conditioning_set_values(
                conditioning, {"reference_latents": latents}, append=True
            )  # type: ignore[call-arg]
        )
