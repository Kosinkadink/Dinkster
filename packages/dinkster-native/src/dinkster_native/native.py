"""Native execution value-type registration."""

from __future__ import annotations

from dinkster_assets import (
    ASSET_TYPE,
    SAVE_TARGET_TYPE,
    register_asset_type,
    register_save_target_type,
    resolver_from_env,
)
from dinkster_inference_wire import register_inference_types
from dinkster_inference_wire.hook_values import normalize_clip_hook_owners
from dinkster_values import (
    AUDIO_RECORD_TYPE,
    AUDIO_TYPE,
    CLIP_TYPE,
    IMAGE_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL3D_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    VIDEO_TYPE,
    TypeRegistry,
    register_curve_type,
    register_model3d_type,
)

from .audio import register_audio_type
from .devices import comfy_resident_meta
from .image import (
    register_image_asset_providers,
    register_image_type,
)
from .latent import register_latent_type
from .pool import default_pool
from .resident import register_resident_type
from .video import register_video_type


def _register_resident(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        register_resident_type(
            registry,
            type_id,
            table=default_pool(),
            meta=comfy_resident_meta,
            coerce=normalize_clip_hook_owners if type_id == CLIP_TYPE else None,
        )


def register_native_types(registry: TypeRegistry) -> None:
    """Register the wire and resident types used by native execution."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if AUDIO_RECORD_TYPE not in registry:
        register_asset_type(registry, resolver_from_env(), type_id=AUDIO_RECORD_TYPE)
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)

    for type_id in (
        MODEL_TYPE,
        CLIP_TYPE,
        VAE_TYPE,
        MODEL_PATCH_TYPE,
        INFERENCE_LATENT_OPERATION_TYPE,
    ):
        _register_resident(registry, type_id)

    if LATENT_TYPE not in registry:
        register_latent_type(registry, LATENT_TYPE)

    register_inference_types(registry)
    register_curve_type(registry)
    register_model3d_type(registry, MODEL3D_TYPE)

    for type_id in (IMAGE_TYPE, MASK_TYPE):
        if type_id not in registry:
            register_image_type(registry, type_id)
    if registry.asset_decoder_for(IMAGE_TYPE) is None:
        register_image_asset_providers(registry, IMAGE_TYPE)

    if AUDIO_TYPE not in registry:
        register_audio_type(registry, AUDIO_TYPE)
    if VIDEO_TYPE not in registry:
        register_video_type(registry, VIDEO_TYPE)
