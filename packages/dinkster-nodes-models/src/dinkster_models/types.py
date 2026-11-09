"""Model and hook producers retain their resident and structural codecs."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    CONTROL_NET_TYPE,
    FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    STYLE_MODEL_TYPE,
    TIMESTEPS_RANGE_TYPE,
    VAE_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    register_asset_type,
    register_inference_types,
    register_resident_type,
    resolver_from_env,
)
from dinkster_inference_wire.hook_values import (
    normalize_clip_hook_owners,
    normalize_hook_owners,
    register_hook_structural_type,
)


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    register_inference_types(registry)
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    registered: list[TypeSpec] = []
    for type_id in (
        MODEL_TYPE,
        CLIP_TYPE,
        CLIP_VISION_TYPE,
        CONTROL_NET_TYPE,
        VAE_TYPE,
        STYLE_MODEL_TYPE,
        MODEL_PATCH_TYPE,
        LATENT_UPSCALE_MODEL_TYPE,
        HOOKS_TYPE,
        HOOK_KEYFRAMES_TYPE,
    ):
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_resident_type(
                registry,
                type_id,
                table=default_pool(),
                meta=comfy_resident_meta,
                coerce=normalize_hook_owners
                if type_id == HOOKS_TYPE
                else normalize_clip_hook_owners
                if type_id == CLIP_TYPE
                else None,
            )
        )
    for type_id in (TIMESTEPS_RANGE_TYPE, FLOATS_TYPE):
        registered.append(
            registry.spec(type_id)
            if type_id in registry
            else register_hook_structural_type(registry, type_id)
        )
    return tuple(registered)
