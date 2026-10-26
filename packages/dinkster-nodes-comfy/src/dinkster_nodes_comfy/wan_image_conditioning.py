"""Native Wan image and frame conditioning nodes."""

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
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    materialize_image,
)

from .conditioning import conditioning_set_values

_CLIP_VISION_OUTPUT = TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CATEGORY = "model/conditioning/wan"


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _common_upscale(image: Any, width: int, height: int) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(
        1, -1
    )


def _latent(
    torch: Any,
    management: Any,
    batch_size: int,
    channels: int,
    length: int,
    height: int,
    width: int,
) -> Any:
    return torch.zeros(
        (batch_size, channels, ((length - 1) // 4) + 1, height, width),
        device=management.intermediate_device(),
    )


class WanImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanImageToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec(
                    "ref_pad_image",
                    _IMAGE,
                    required=False,
                    doc=(
                        "Fills the padding frames of the image conditioning with this image "
                        "instead "
                        "of gray, anchoring identity without pinning frames (SVI-style anti-drift "
                        "padding, used by models such as ID-V2V)."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        ref_pad_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = _latent(torch, management, batch_size, 16, length, height // 8, width // 8)
        if start_image is not None:
            start = _common_upscale(materialize_image(start_image)[:length], width, height)
            image = (
                torch.ones(
                    (length, height, width, start.shape[-1]), device=start.device, dtype=start.dtype
                )
                * 0.5
            )
            if ref_pad_image is not None:
                ref_pad = _common_upscale(materialize_image(ref_pad_image)[:1], width, height)
                image[:, :, :, :3] = ref_pad[:, :, :, :3].to(device=image.device, dtype=image.dtype)
            image[: start.shape[0]] = start
            encoded = vae.encode(image[:, :, :, :3])
            mask = torch.ones(
                (1, 1, latent.shape[2], encoded.shape[-2], encoded.shape[-1]),
                device=start.device,
                dtype=start.dtype,
            )
            mask[:, :, : ((start.shape[0] - 1) // 4) + 1] = 0.0
            values = {"concat_latent_image": encoded, "concat_mask": mask}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class WanFirstLastFrameToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanFirstLastFrameToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("clip_vision_start_image", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("clip_vision_end_image", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("end_image", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        end_image: object | None = None,
        clip_vision_start_image: object | None = None,
        clip_vision_end_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        spatial_scale = vae.spacial_compression_encode()
        latent = _latent(
            torch,
            management,
            batch_size,
            vae.latent_channels,
            length,
            height // spatial_scale,
            width // spatial_scale,
        )
        start: Any = None
        if start_image is not None:
            start = _common_upscale(materialize_image(start_image)[:length], width, height)
        end: Any = None
        if end_image is not None:
            end = _common_upscale(materialize_image(end_image)[-length:], width, height)
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        if start is not None:
            image[: start.shape[0]] = start
            mask[:, :, : start.shape[0] + 3] = 0.0
        if end is not None:
            image[-end.shape[0] :] = end
            mask[:, :, -end.shape[0] :] = 0.0
        encoded = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        values = {"concat_latent_image": encoded, "concat_mask": mask}
        positive = conditioning_set_values(positive, values)
        negative = conditioning_set_values(negative, values)
        first = (
            materialize_clip_vision_output(clip_vision_start_image)
            if clip_vision_start_image is not None
            else None
        )
        last = (
            materialize_clip_vision_output(clip_vision_end_image)
            if clip_vision_end_image is not None
            else None
        )
        clip_vision_output = first
        if last is not None:
            if clip_vision_output is not None:
                output = importlib.import_module("dinkster_inference.clip_vision").Output()
                output.penultimate_hidden_states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states, last.penultimate_hidden_states],
                    dim=-2,
                )
                clip_vision_output = output
            else:
                clip_vision_output = last
        if clip_vision_output is not None:
            values = {"clip_vision_output": clip_vision_output}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class WanFunInpaintToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanFunInpaintToVideo",
            category="model/conditioning/wan/fun inpaint",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("end_image", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        end_image: object | None = None,
        clip_vision_output: object | None = None,
    ) -> Mapping[str, object]:
        return WanFirstLastFrameToVideo.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            start_image=start_image,
            end_image=end_image,
            clip_vision_start_image=clip_vision_output,
        )
