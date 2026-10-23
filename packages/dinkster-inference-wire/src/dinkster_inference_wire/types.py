"""Value type registration retained outside the fork-backed runtime."""

from __future__ import annotations

from dinkster_values import TypeRegistry, TypeSpec, register_resident_type

from .conditioning_wire import CONDITIONING_TYPE_ID, register_conditioning_type
from .sampling_wire import register_sampling_type

MODEL_TYPE_ID = "MODEL"
CLIP_TYPE_ID = "CLIP"
CLIP_VISION_TYPE_ID = "CLIP_VISION"
VAE_TYPE_ID = "VAE"
SAMPLER_TYPE_ID = "SAMPLER"
SIGMAS_TYPE_ID = "SIGMAS"
GUIDER_TYPE_ID = "GUIDER"
NOISE_TYPE_ID = "NOISE"

_RESIDENT_TYPE_IDS = (
    MODEL_TYPE_ID,
    CLIP_TYPE_ID,
    CLIP_VISION_TYPE_ID,
    VAE_TYPE_ID,
    GUIDER_TYPE_ID,
)
_SAMPLING_TYPE_IDS = (SAMPLER_TYPE_ID, SIGMAS_TYPE_ID, NOISE_TYPE_ID)


def register_inference_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Idempotently register shared wire and resident value types."""
    registered = [
        registry.spec(CONDITIONING_TYPE_ID)
        if CONDITIONING_TYPE_ID in registry
        else register_conditioning_type(registry)
    ]
    for type_id in _RESIDENT_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_resident_type(registry, type_id)
        )
    for type_id in _SAMPLING_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_sampling_type(registry, type_id)
        )
    return tuple(registered)


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
