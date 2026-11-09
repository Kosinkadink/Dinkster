"""Native Hunyuan Video and LTXV latent upscale nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_INT,
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_LATENT_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _combo(name: str, options: tuple[str, ...], default: str | None = None) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        widget=ComboWidget(tuple(ComboOption(value) for value in options)),
    )


class HunyuanVideo15LatentUpscaleWithModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HunyuanVideo15LatentUpscaleWithModel",
            display_name="Hunyuan Video 1.5 Latent Upscale With Model",
            category="model/latent/hunyuan video",
            inputs=(
                InputSpec("model", _LATENT_UPSCALE_MODEL),
                InputSpec("samples", _LATENT),
                _combo("upscale_method", _UPSCALE_METHODS, "bilinear"),
                InputSpec(
                    "width", _INT, default=1280, widget=NumberWidget(min=0, max=16384, step=8)
                ),
                InputSpec(
                    "height", _INT, default=720, widget=NumberWidget(min=0, max=16384, step=8)
                ),
                _combo("crop", ("disabled", "center")),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        samples: object,
        upscale_method: str,
        width: int,
        height: int,
        crop: str,
    ) -> Mapping[str, object]:
        if width == 0 and height == 0:
            return cls.outputs(LATENT=samples)
        source = materialize_latent(samples)
        latent = source["samples"]
        if width == 0:
            height = max(64, height)
            width = max(64, round(latent.shape[-1] * height / latent.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(latent.shape[-2] * width / latent.shape[-1]))
        else:
            width, height = max(64, width), max(64, height)
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        upscaled = utils.common_upscale(latent, width // 16, height // 16, upscale_method, crop)
        return cls.outputs(LATENT={"samples": model.resample_latent(upscaled).cpu().float()})


class LTXVLatentUpsampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVLatentUpsampler",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", _LATENT),
                InputSpec("upscale_model", _LATENT_UPSCALE_MODEL),
                InputSpec("vae", _VAE),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, samples: object, upscale_model: Any, vae: Any) -> Mapping[str, object]:
        source = materialize_latent(samples)
        latents = source["samples"]
        input_dtype = latents.dtype
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_models_gpu(
            [upscale_model], memory_required=math.prod(latents.shape) * 3000.0
        )
        latents = latents.to(dtype=upscale_model.model_dtype(), device=upscale_model.load_device)
        statistics = vae.first_stage_model.per_channel_statistics
        upsampled = statistics.normalize(upscale_model.model(statistics.un_normalize(latents)))
        result = source.copy()
        result["samples"] = upsampled.to(dtype=input_dtype, device=management.intermediate_device())
        result.pop("noise_mask", None)
        return cls.outputs(LATENT=result)


LATENT_UPSCALE_NODES = (HunyuanVideo15LatentUpscaleWithModel, LTXVLatentUpsampler)
