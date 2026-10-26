"""Fibo 2 image conditioning."""

from __future__ import annotations

import importlib
import math
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

TEXT_IMAGES_START = math.exp(1.15) / (9 + math.exp(1.15))
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _has_images(tokens: Any) -> bool:
    return any(
        isinstance(token[0], dict)
        for batches in tokens.values()
        for batch in batches
        for token in batch
    )


def _trained_size(aspect_ratio: float) -> tuple[int, int]:
    area = 1024 * 1024
    sizes = [
        (width, height)
        for width in range(64, 2 * area // 64, 16)
        for height in range(
            max(64, 16 * math.ceil(0.9 * area / width / 16)),
            16 * math.floor(1.1 * area / width / 16) + 1,
            16,
        )
    ]
    return min(
        sizes,
        key=lambda size: (
            round(1 - min(size[0] / size[1] / aspect_ratio, aspect_ratio * size[1] / size[0]), 9),
            abs(size[0] * size[1] - area),
            -size[0],
        ),
    )


class TextEncodeFibo2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeFibo2",
            category="model/conditioning/fibo",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec("prompt", _STRING, widget=StringWidget(multiline=True)),
                InputSpec(
                    "negative_prompt", _STRING, advanced=True, widget=StringWidget(multiline=True)
                ),
                InputSpec("vae", _VAE, required=False),
                InputSpec("width", _INT, default=0, widget=NumberWidget(min=0, max=16384, step=16)),
                InputSpec(
                    "height", _INT, default=0, widget=NumberWidget(min=0, max=16384, step=16)
                ),
                InputSpec(
                    "batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE),),
                    min_members=0,
                    member_names=tuple(f"image_{index}" for index in range(1, 17)),
                ),
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
        references = [
            image[index : index + 1] for image in references for index in range(image.shape[0])
        ]
        default = (
            _trained_size(references[0].shape[2] / references[0].shape[1])
            if references
            else (1024, 1024)
        )
        width, height = (
            max(16, (width or default[0]) // 16 * 16),
            max(16, (height or default[1]) // 16 * 16),
        )
        sizes = (
            [(width, height)]
            if len(references) == 1
            else [_trained_size(image.shape[2] / image.shape[1]) for image in references]
        )
        if references:
            initialize_inference_process()
            common_upscale = importlib.import_module("dinkster_inference.utils").common_upscale
            references = [
                image
                if (image.shape[2], image.shape[1]) == size
                else common_upscale(image.movedim(-1, 1), *size, "lanczos", "center").movedim(1, -1)
                for image, size in zip(references, sizes, strict=True)
            ]
        tokens = clip.tokenize(prompt, images=references)
        positive: Any = clip.encode_from_tokens_scheduled(tokens)
        if not references and _has_images(tokens):
            base = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, text_images=False))
            positive = conditioning_set_values(
                base, {"end_percent": TEXT_IMAGES_START - 1e-6}
            ) + conditioning_set_values(positive, {"start_percent": TEXT_IMAGES_START})
        initialize_inference_process()
        template = importlib.import_module("dinkster_inference.text_encoders.fibo2").TEMPLATE
        negative: Any = clip.encode_from_tokens_scheduled(
            clip.tokenize(template.format(negative_prompt) if negative_prompt.strip() else "")
        )
        if vae is not None and references:
            values = {"reference_latents": [vae.encode(image) for image in references]}
            positive = cast("Any", conditioning_set_values(positive, values, append=True))
            negative = cast("Any", conditioning_set_values(negative, values, append=True))
        torch = importlib.import_module("torch")
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": torch.zeros((batch_size, 128, height // 16, width // 16))},
        )
