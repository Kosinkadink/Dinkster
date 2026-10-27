"""Native Wan camera-conditioned image-to-video inputs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
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


class WanCameraImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanCameraImageToVideo",
            category="model/conditioning/wan/camera",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=832,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=480,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=81,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
                InputSpec(
                    "clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE), required=False
                ),
                InputSpec("start_image", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec(
                    "camera_conditions",
                    TypeExpr.concrete(WAN_CAMERA_EMBEDDING_TYPE),
                    required=False,
                ),
            ),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
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
        start_image: object | None = None,
        clip_vision_output: object | None = None,
        camera_conditions: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        formats = importlib.import_module("dinkster_inference.latent_formats")
        shape = (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8)
        latent = torch.zeros(shape, device=management.intermediate_device())
        concat_latent = formats.Wan21().process_out(
            torch.zeros(shape, device=management.intermediate_device())
        )

        if start_image is not None:
            utils = importlib.import_module("dinkster_inference.utils")
            image = materialize_image(start_image)
            image = utils.common_upscale(
                image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            encoded = vae.encode(image[:, :, :, :3])
            concat_latent[:, :, : encoded.shape[2]] = encoded[:, :, : concat_latent.shape[2]]
            mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
            mask[:, :, : image.shape[0] + 3] = 0.0
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            values = {"concat_latent_image": concat_latent, "concat_mask": mask}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)

        if camera_conditions is not None:
            values = {"camera_conditions": materialize_latent(camera_conditions)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)

        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)

        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


WAN_CAMERA_CONDITIONING_NODES = (WanCameraImageToVideo,)
