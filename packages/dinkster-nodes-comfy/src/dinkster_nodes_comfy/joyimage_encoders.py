"""JoyImage edit conditioning."""

from __future__ import annotations

import importlib
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

BUCKETS_1024 = (
    (512, 1792),
    (512, 1856),
    (512, 1920),
    (512, 1984),
    (512, 2048),
    (576, 1600),
    (576, 1664),
    (576, 1728),
    (576, 1792),
    (640, 1472),
    (640, 1536),
    (640, 1600),
    (704, 1344),
    (704, 1408),
    (704, 1472),
    (768, 1216),
    (768, 1280),
    (768, 1344),
    (832, 1152),
    (832, 1216),
    (896, 1088),
    (896, 1152),
    (960, 1024),
    (960, 1088),
    (1024, 960),
    (1024, 1024),
    (1088, 896),
    (1088, 960),
    (1152, 832),
    (1152, 896),
    (1216, 768),
    (1216, 832),
    (1280, 768),
    (1344, 704),
    (1344, 768),
    (1408, 704),
    (1472, 640),
    (1472, 704),
    (1536, 640),
    (1600, 576),
    (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 512),
    (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
)

_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _bucket(height: int, width: int) -> tuple[int, int]:
    ratio = height / width
    return min(BUCKETS_1024, key=lambda candidate: abs(candidate[0] / candidate[1] - ratio))


def _resize(image: Any) -> Any:
    if image.shape[0] != 1:
        raise ValueError("JoyImage reference inputs must contain one image each")
    initialize_inference_process()
    samples = image.movedim(-1, 1)
    height, width = _bucket(samples.shape[2], samples.shape[3])
    return (
        importlib.import_module("dinkster_inference.utils")
        .common_upscale(samples, width, height, "bilinear", "center")
        .movedim(1, -1)[..., :3]
    )


class TextEncodeJoyImageEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeJoyImageEdit",
            category="model/conditioning/joyimage",
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec("vae", _VAE, required=False),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE),),
                    min_members=0,
                    member_names=tuple(f"image{index}" for index in range(1, 7)),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, clip: Any, prompt: str, vae: Any = None, images: Mapping[str, object] | None = None
    ) -> Mapping[str, object]:
        resized = [
            _resize(materialize_image(image))
            for image in (images or {}).values()
            if image is not None
        ]
        conditioning: Any = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=resized))
        if vae is not None and resized:
            conditioning = cast(
                "Any",
                conditioning_set_values(
                    conditioning,
                    {"reference_latents": [vae.encode(image) for image in resized]},
                    append=True,
                ),
            )
        return cls.outputs(CONDITIONING=conditioning)
