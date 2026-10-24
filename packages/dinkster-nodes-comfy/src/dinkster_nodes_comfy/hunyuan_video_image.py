"""Hunyuan image-to-video conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    ComboWidget,
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
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CATEGORY = "model/conditioning/hunyuan video"


def _integer(name: str, default: int, minimum: int, *, step: int | None = None) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=16384 if name != "batch_size" else 4096, step=step),
    )


def _resized_start_image(start_image: object, length: int, width: int, height: int) -> Any:
    image = materialize_image(start_image)[:length]
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(
        1, -1
    )


def _resized_old_start_image(start_image: object, length: int, width: int, height: int) -> Any:
    image = materialize_image(start_image)[:length, :, :, :3]
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(
        1, -1
    )


class HunyuanImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HunyuanImageToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 848, 16, step=16),
                _integer("height", 480, 16, step=16),
                _integer("length", 53, 1, step=4),
                _integer("batch_size", 1, 1),
                InputSpec(
                    "guidance_type",
                    _COMBO,
                    default="v1 (concat)",
                    widget=ComboWidget(("v1 (concat)", "v2 (replace)", "custom")),
                    advanced=True,
                ),
                InputSpec("start_image", _IMAGE, required=False),
            ),
            outputs=(OutputSpec("positive", _CONDITIONING), OutputSpec("latent", _LATENT)),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        guidance_type: str,
        start_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        result: dict[str, object] = {"samples": latent}
        if start_image is not None:
            image = _resized_old_start_image(start_image, length, width, height)
            encoded = vae.encode(image)
            mask = torch.ones(
                (1, 1, latent.shape[2], encoded.shape[-2], encoded.shape[-1]),
                device=image.device,
                dtype=image.dtype,
            )
            mask[:, :, : ((image.shape[0] - 1) // 4) + 1] = 0.0
            if guidance_type == "v1 (concat)":
                values = {"concat_latent_image": encoded, "concat_mask": mask}
            elif guidance_type == "v2 (replace)":
                values = {"guiding_frame_index": 0}
                latent[:, :, : encoded.shape[2]] = encoded
                result["noise_mask"] = mask
            elif guidance_type == "custom":
                values = {"ref_latent": encoded}
            else:
                raise ValueError(f"unsupported guidance type: {guidance_type}")
            positive = conditioning_set_values(positive, values)
        return cls.outputs(positive=positive, latent=result)


class HunyuanVideo15ImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HunyuanVideo15ImageToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 848, 16, step=16),
                _integer("height", 480, 16, step=16),
                _integer("length", 33, 1, step=4),
                _integer("batch_size", 1, 1),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
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
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object | None = None,
        clip_vision_output: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            [batch_size, 32, ((length - 1) // 4) + 1, height // 16, width // 16],
            device=management.intermediate_device(),
        )
        if start_image is not None:
            image = _resized_start_image(start_image, length, width, height)
            encoded = vae.encode(image[:, :, :, :3])
            concat = torch.zeros(
                (latent.shape[0], 32, latent.shape[2], latent.shape[3], latent.shape[4]),
                device=management.intermediate_device(),
            )
            concat[:, :, : encoded.shape[2], :, :] = encoded
            mask = torch.ones(
                (1, 1, latent.shape[2], concat.shape[-2], concat.shape[-1]),
                device=image.device,
                dtype=image.dtype,
            )
            mask[:, :, : ((image.shape[0] - 1) // 4) + 1] = 0.0
            values = {"concat_latent_image": concat, "concat_mask": mask}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class HunyuanVideo15SuperResolution(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HunyuanVideo15SuperResolution",
            display_name="Hunyuan Video 1.5 Super Resolution",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE, required=False),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("latent", _LATENT),
                InputSpec(
                    "noise_augmentation",
                    _FLOAT,
                    default=0.70,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
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
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        latent: Mapping[str, Any],
        noise_augmentation: float,
        vae: Any | None = None,
        start_image: object | None = None,
        clip_vision_output: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        samples = latent["samples"]
        channels = samples.shape[1]
        conditioning_latent = torch.zeros(
            (samples.shape[0], channels * 2 + 2, *samples.shape[-3:]),
            device=management.intermediate_device(),
        )
        conditioning_latent[:, channels + 1 : 2 * channels + 1] = samples
        conditioning_latent[:, 2 * channels + 1] = 1
        if start_image is not None:
            image = materialize_image(start_image)
            utils = importlib.import_module("dinkster_inference.utils")
            image = utils.common_upscale(
                image.movedim(-1, 1),
                samples.shape[-1] * 16,
                samples.shape[-2] * 16,
                "bilinear",
                "center",
            ).movedim(1, -1)
            if vae is None:
                raise ValueError("vae is required when start_image is provided")
            encoded = vae.encode(image[:, :, :, :3])
            conditioning_latent[:, :channels, : encoded.shape[2]] = encoded
            conditioning_latent[:, channels + 1, 0] = 1
        values = {
            "concat_latent_image": conditioning_latent,
            "noise_augmentation": noise_augmentation,
        }
        positive = conditioning_set_values(positive, values)
        negative = conditioning_set_values(negative, values)
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent=latent)


HUNYUAN_VIDEO_IMAGE_NODES = (
    HunyuanImageToVideo,
    HunyuanVideo15ImageToVideo,
    HunyuanVideo15SuperResolution,
)
