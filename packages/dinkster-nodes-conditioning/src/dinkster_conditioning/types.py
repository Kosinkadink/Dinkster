"""Conditioning and encoder producers retain their existing codecs."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_ENCODER_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    register_asset_type,
    register_inference_types,
    register_resident_type,
    resolver_from_env,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.vision_wire import (
    clip_vision_output_fingerprint,
    decode_clip_vision_output,
    encode_clip_vision_output,
    validate_clip_vision_output,
)
from dinkster_inference_wire.wan_wire import register_wan_type


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    register_inference_types(registry)
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    registered = [
        registry.spec(CONDITIONING_TYPE)
        if CONDITIONING_TYPE in registry
        else register_conditioning_type(registry, resident_table=default_pool())
    ]
    registered.append(
        registry.spec(AUDIO_ENCODER_TYPE)
        if AUDIO_ENCODER_TYPE in registry
        else register_resident_type(
            registry, AUDIO_ENCODER_TYPE, table=default_pool(), meta=comfy_resident_meta
        )
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
    registered.append(
        registry.spec(AUDIO_ENCODER_OUTPUT_TYPE)
        if AUDIO_ENCODER_OUTPUT_TYPE in registry
        else register_wan_type(registry, AUDIO_ENCODER_OUTPUT_TYPE)
    )
    return tuple(registered)
