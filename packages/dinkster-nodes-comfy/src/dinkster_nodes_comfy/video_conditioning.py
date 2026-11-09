"""Native Stable Video image conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
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

_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _integer(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    step: int | None = None,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
        advanced=advanced,
    )


class SVD_img2vid_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SVD_img2vid_Conditioning",
            category="model/conditioning/stable video",
            inputs=(
                InputSpec("clip_vision", _CLIP_VISION),
                InputSpec("init_image", _IMAGE),
                InputSpec("vae", _VAE),
                _integer("width", 1024, 16, 16384, step=8),
                _integer("height", 576, 16, 16384, step=8),
                _integer("video_frames", 14, 1, 4096),
                _integer("motion_bucket_id", 127, 1, 1023, advanced=True),
                _integer("fps", 6, 1, 1024),
                InputSpec(
                    "augmentation_level",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    advanced=True,
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
        clip_vision: Any,
        init_image: object,
        vae: Any,
        width: int,
        height: int,
        video_frames: int,
        motion_bucket_id: int,
        fps: int,
        augmentation_level: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        image = materialize_image(init_image)
        output = clip_vision.encode_image(image)
        pooled = output.image_embeds.unsqueeze(0)
        utils = importlib.import_module("dinkster_inference.utils")
        pixels = utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        encoded = vae.encode(encode_pixels)
        metadata = {
            "motion_bucket_id": motion_bucket_id,
            "fps": fps,
            "augmentation_level": augmentation_level,
            "concat_latent_image": encoded,
        }
        return cls.outputs(
            positive=[[pooled, metadata]],
            negative=[
                [
                    torch.zeros_like(pooled),
                    {**metadata, "concat_latent_image": torch.zeros_like(encoded)},
                ]
            ],
            latent={"samples": torch.zeros([video_frames, 4, height // 8, width // 8])},
        )


VIDEO_CONDITIONING_NODES = (SVD_img2vid_Conditioning,)
