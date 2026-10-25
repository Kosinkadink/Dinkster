"""Value type registration retained outside the fork-backed runtime."""

from __future__ import annotations

from dinkster_memory.resident import default_pool
from dinkster_values import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CLIP_VISION_TYPE,
    CONTROL_NET_TYPE,
    FLOATS_TYPE,
    GLIGEN_TYPE,
    GUIDER_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    STYLE_MODEL_TYPE,
    TIMESTEPS_RANGE_TYPE,
    VAE_TYPE,
    TypeRegistry,
    TypeSpec,
    register_resident_type,
)

from .conditioning_wire import CONDITIONING_TYPE_ID, register_conditioning_type
from .hook_values import (
    normalize_clip_hook_owners,
    normalize_hook_owners,
    register_hook_structural_type,
)
from .media_wire import (
    materialize_audio,
    materialize_image,
    materialize_latent,
    register_media_types,
)
from .sampling_wire import register_sampling_type
from .vision_wire import (
    clip_vision_output_fingerprint,
    decode_clip_vision_output,
    encode_clip_vision_output,
    materialize_clip_vision_output,
    validate_clip_vision_output,
)

MODEL_TYPE_ID = MODEL_TYPE
CLIP_TYPE_ID = CLIP_TYPE
CLIP_VISION_TYPE_ID = CLIP_VISION_TYPE
VAE_TYPE_ID = VAE_TYPE
SAMPLER_TYPE_ID = SAMPLER_TYPE
SIGMAS_TYPE_ID = SIGMAS_TYPE
GUIDER_TYPE_ID = GUIDER_TYPE
NOISE_TYPE_ID = NOISE_TYPE

_RESIDENT_TYPE_IDS = (
    MODEL_TYPE_ID,
    MODEL_PATCH_TYPE,
    CLIP_TYPE_ID,
    CLIP_VISION_TYPE_ID,
    VAE_TYPE_ID,
    GUIDER_TYPE_ID,
    CONTROL_NET_TYPE,
    STYLE_MODEL_TYPE,
    GLIGEN_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    HOOKS_TYPE,
    HOOK_KEYFRAMES_TYPE,
)
_SAMPLING_TYPE_IDS = (SAMPLER_TYPE_ID, SIGMAS_TYPE_ID, NOISE_TYPE_ID)
_STRUCTURAL_TYPE_IDS = (TIMESTEPS_RANGE_TYPE, FLOATS_TYPE)


def register_inference_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Idempotently register shared wire and resident value types."""
    table = default_pool()
    registered = [
        registry.spec(CONDITIONING_TYPE_ID)
        if CONDITIONING_TYPE_ID in registry
        else register_conditioning_type(registry, resident_table=table)
    ]
    for type_id in _RESIDENT_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_resident_type(
                registry,
                type_id,
                table=table,
                coerce=(
                    normalize_hook_owners
                    if type_id == HOOKS_TYPE
                    else normalize_clip_hook_owners
                    if type_id == CLIP_TYPE
                    else None
                ),
            )
        )
    for type_id in _SAMPLING_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_sampling_type(registry, type_id)
        )
    for type_id in _STRUCTURAL_TYPE_IDS:
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_hook_structural_type(registry, type_id)
        )
    if CLIP_VISION_OUTPUT_TYPE not in registry:
        registry.register(
            CLIP_VISION_OUTPUT_TYPE,
            encode=encode_clip_vision_output,
            decode=decode_clip_vision_output,
            fingerprint=clip_vision_output_fingerprint,
            validate_encoded=validate_clip_vision_output,
        )
    registered.append(registry.spec(CLIP_VISION_OUTPUT_TYPE))
    registered.extend(register_media_types(registry))
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
    "materialize_audio",
    "materialize_clip_vision_output",
    "materialize_image",
    "materialize_latent",
    "register_inference_types",
]
