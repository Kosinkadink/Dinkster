"""WanDancer image, vision and audio conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    materialize_image,
    materialize_latent,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


class WanDancerVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanDancerVideo",
            category="model/conditioning/wan/dancer",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=480,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=832,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=149,
                    widget=NumberWidget(min=1, max=16384, step=4),
                    doc=(
                        "The number of frames in the generated video. "
                        "Should stay 149 for WanDancer."
                    ),
                ),
                InputSpec(
                    "clip_vision_output",
                    TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE),
                    required=False,
                    doc="The CLIP vision embeds for the first frame.",
                ),
                InputSpec(
                    "clip_vision_output_ref",
                    TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE),
                    required=False,
                    doc="The CLIP vision embeds for the reference image.",
                ),
                InputSpec(
                    "start_image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc="The initial image(s) to be encoded, can be any number of frames.",
                ),
                InputSpec(
                    "mask",
                    TypeExpr.concrete(MASK_TYPE),
                    required=False,
                    doc=(
                        "Image conditioning mask for the start image(s). White is kept, "
                        "black is generated. Used for the local generations."
                    ),
                ),
                InputSpec(
                    "audio_encoder_output",
                    TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE),
                    required=False,
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec(
                    "latent",
                    TypeExpr.concrete(LATENT_TYPE),
                    display_name="latent",
                    doc="Empty latent.",
                ),
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
        start_image: object | None = None,
        mask: object | None = None,
        clip_vision_output: object | None = None,
        clip_vision_output_ref: object | None = None,
        audio_encoder_output: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        latent = torch.zeros(
            (1, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        if start_image is not None:
            start = materialize_image(start_image)[:length]
            start = utils.common_upscale(
                start.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image = torch.zeros(
                (length, height, width, start.shape[-1]), device=start.device, dtype=start.dtype
            )
            image[: start.shape[0]] = start
            encoded = vae.encode(image[:, :, :, :3])
            if mask is None:
                concat_mask = torch.ones(
                    (1, 1, latent.shape[2], encoded.shape[-2], encoded.shape[-1]),
                    device=start.device,
                    dtype=start.dtype,
                )
                concat_mask[:, :, : ((start.shape[0] - 1) // 4) + 1] = 0.0
            else:
                concat_mask = 1 - materialize_image(mask)[:length].unsqueeze(0)
                concat_mask = utils.common_upscale(
                    concat_mask, encoded.shape[-2], encoded.shape[-1], "nearest-exact", "disabled"
                )
                concat_mask = torch.cat(
                    [
                        torch.repeat_interleave(concat_mask[:, 0:1], repeats=4, dim=1),
                        concat_mask[:, 1:],
                    ],
                    dim=1,
                )
                concat_mask = concat_mask.view(
                    1, concat_mask.shape[1] // 4, 4, encoded.shape[-2], encoded.shape[-1]
                ).transpose(1, 2)
            values = {"concat_latent_image": encoded, "concat_mask": concat_mask}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        if clip_vision_output is not None:
            values = {
                "clip_vision_output": materialize_clip_vision_output(clip_vision_output),
                "clip_vision_output_ref": (
                    materialize_clip_vision_output(clip_vision_output_ref)
                    if clip_vision_output_ref is not None
                    else None
                ),
            }
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        if audio_encoder_output is not None:
            audio = materialize_latent(audio_encoder_output)
            values = {
                "audio_embed": audio["audio_feature"],
                "fps": audio["fps"],
                "audio_inject_scale": audio.get("audio_inject_scale", 1.0),
            }
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})
