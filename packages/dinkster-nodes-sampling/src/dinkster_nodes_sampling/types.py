"""Sampling value ownership and existing transport codecs."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    GUIDER_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    decode_latent,
    default_pool,
    encode_latent,
    latent_fingerprint,
    register_resident_type,
    validate_latent_encoded,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.sampling_wire import register_sampling_type


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    if LATENT_TYPE not in registry:
        registry.register(
            LATENT_TYPE,
            encode=encode_latent,
            decode=decode_latent,
            fingerprint=latent_fingerprint(LATENT_TYPE),
            validate_encoded=validate_latent_encoded,
        )
    return (
        registry.spec(LATENT_TYPE),
        registry.spec(CONDITIONING_TYPE)
        if CONDITIONING_TYPE in registry
        else register_conditioning_type(registry, resident_table=default_pool()),
        *(
            register_sampling_type(registry, type_id)
            for type_id in (SAMPLER_TYPE, SIGMAS_TYPE, NOISE_TYPE)
        ),
        *(
            registry.spec(type_id)
            if type_id in registry
            else register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )
            for type_id in (MODEL_TYPE, GUIDER_TYPE)
        ),
    )
