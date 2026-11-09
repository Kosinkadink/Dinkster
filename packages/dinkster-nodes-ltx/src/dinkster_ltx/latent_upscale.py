from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    initialize_inference_process,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)

_LATENT_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)


class LTXVLatentUpsampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="LTXVLatentUpsampler",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", _LATENT),
                InputSpec("upscale_model", _LATENT_UPSCALE_MODEL),
                InputSpec("vae", _VAE),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )
        return first_party_schema(schema)

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
