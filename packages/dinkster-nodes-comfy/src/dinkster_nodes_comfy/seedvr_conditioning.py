"""Native SeedVR2 latent conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_SEEDVR2_LATENT_CHANNELS = 16
_SEEDVR2_INVALID_MODEL_MSG_PREFIX = (
    "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
)
_ATTR_MISSING = object()


def _resolve_seedvr2_diffusion_model(model: Any) -> Any:
    inner = getattr(model, "model", _ATTR_MISSING)
    if inner is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input has no 'model' attribute "
            f"(got type {type(model).__name__})."
        )
    if inner is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input.model is None "
            f"(input type {type(model).__name__})."
        )
    diffusion_model = getattr(inner, "diffusion_model", _ATTR_MISSING)
    if diffusion_model is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model' has no "
            f"'diffusion_model' attribute (got type {type(inner).__name__})."
        )
    if diffusion_model is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model.diffusion_model' "
            f"is None (model.model type {type(inner).__name__})."
        )
    return diffusion_model


class SeedVR2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SeedVR2Conditioning",
            display_name="Apply SeedVR2 Conditioning",
            category="model/conditioning",
            description="Build SeedVR2 positive/negative conditioning from a VAE latent.",
            search_terms=("seedvr2", "upscale", "conditioning"),
            inputs=(
                InputSpec("model", _MODEL, doc="The SeedVR2 model."),
                InputSpec("vae_conditioning", _LATENT, display_name="latent"),
            ),
            outputs=(
                OutputSpec(
                    "positive",
                    _CONDITIONING,
                    display_name="positive",
                    doc="The positive conditioning for sampling.",
                ),
                OutputSpec(
                    "negative",
                    _CONDITIONING,
                    display_name="negative",
                    doc="The negative conditioning for sampling.",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, model: Any, vae_conditioning: object) -> Mapping[str, object]:
        initialize_inference_process()
        latent = materialize_latent(vae_conditioning)["samples"]
        if latent.ndim != 5:
            raise ValueError(
                "SeedVR2Conditioning expects a 5-D VAE latent in Comfy "
                f"channel-first layout; got shape {tuple(latent.shape)}."
            )
        if latent.shape[1] != _SEEDVR2_LATENT_CHANNELS:
            if latent.shape[-1] == _SEEDVR2_LATENT_CHANNELS:
                raise ValueError(
                    "SeedVR2Conditioning expects SeedVR2 VAE latents in Comfy "
                    f"channel-first layout (B, {_SEEDVR2_LATENT_CHANNELS}, T, H, W); "
                    f"got channel-last shape {tuple(latent.shape)}."
                )
            raise ValueError(
                "SeedVR2Conditioning expects SeedVR2 VAE latents with "
                f"{_SEEDVR2_LATENT_CHANNELS} channels; got shape {tuple(latent.shape)}."
            )
        torch = importlib.import_module("torch")
        condition = latent.movedim(1, -1).contiguous()
        diffusion_model = _resolve_seedvr2_diffusion_model(model)
        mask = condition.new_ones(condition.shape[:-1] + (1,))
        condition = torch.cat((condition, mask), dim=-1).movedim(-1, 1)
        positive = [[diffusion_model.positive_conditioning.unsqueeze(0), {"condition": condition}]]
        negative = [[diffusion_model.negative_conditioning.unsqueeze(0), {"condition": condition}]]
        return cls.outputs(positive=positive, negative=negative)
