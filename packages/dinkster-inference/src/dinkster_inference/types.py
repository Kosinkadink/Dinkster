"""Inference-owned value type registration."""

from __future__ import annotations

from collections.abc import Callable

from dinkster_values import TypeRegistry, TypeSpec, register_resident_type

from .conditioning_wire import CONDITIONING_TYPE_ID, register_conditioning_type
from .control_wire import CONTROL_TYPE_ID, register_control_type
from .latent_wire import LATENT_TYPE_ID, register_latent_type
from .sampling_wire import register_sampling_type

MODEL_TYPE_ID = "dinkster.model"
CLIP_TYPE_ID = "dinkster.clip"
CLIP_VISION_TYPE_ID = "dinkster.clip-vision"
VAE_TYPE_ID = "dinkster.vae"
SAMPLER_TYPE_ID = "dinkster.sampler"
SIGMAS_TYPE_ID = "dinkster.sigmas"
GUIDER_TYPE_ID = "dinkster.guider"
NOISE_TYPE_ID = "dinkster.noise"

_HANDLE_TYPE_IDS = (
    MODEL_TYPE_ID,
    CLIP_TYPE_ID,
    CLIP_VISION_TYPE_ID,
    VAE_TYPE_ID,
    SAMPLER_TYPE_ID,
    SIGMAS_TYPE_ID,
    GUIDER_TYPE_ID,
    NOISE_TYPE_ID,
)


def register_inference_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Idempotently register every inference-owned value type."""

    def ensure(type_id: str, register: Callable[[TypeRegistry], TypeSpec]) -> TypeSpec:
        return registry.spec(type_id) if type_id in registry else register(registry)

    def register_handle(type_id: str) -> TypeSpec:
        if type_id in (SAMPLER_TYPE_ID, SIGMAS_TYPE_ID, NOISE_TYPE_ID):
            return register_sampling_type(registry, type_id)
        return ensure(type_id, lambda reg: register_resident_type(reg, type_id))

    return (
        ensure(CONDITIONING_TYPE_ID, register_conditioning_type),
        ensure(LATENT_TYPE_ID, register_latent_type),
        ensure(CONTROL_TYPE_ID, register_control_type),
        *(register_handle(type_id) for type_id in _HANDLE_TYPE_IDS),
    )


__all__ = [
    "CLIP_TYPE_ID",
    "CLIP_VISION_TYPE_ID",
    "GUIDER_TYPE_ID",
    "MODEL_TYPE_ID",
    "NOISE_TYPE_ID",
    "SAMPLER_TYPE_ID",
    "SIGMAS_TYPE_ID",
    "VAE_TYPE_ID",
    "register_inference_types",
]
