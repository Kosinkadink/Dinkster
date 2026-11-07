"""Sampling value ownership and existing transport codecs."""

from dinkster_api.v1 import (
    GUIDER_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    register_resident_type,
)
from dinkster_inference_wire.sampling_wire import register_sampling_type


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    return (
        *(
            register_sampling_type(registry, type_id)
            for type_id in (SAMPLER_TYPE, SIGMAS_TYPE, NOISE_TYPE)
        ),
        registry.spec(GUIDER_TYPE)
        if GUIDER_TYPE in registry
        else register_resident_type(
            registry, GUIDER_TYPE, table=default_pool(), meta=comfy_resident_meta
        ),
    )
