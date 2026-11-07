"""Latent operation producers retain worker-resident operations."""

from dinkster_api.v1 import (
    INFERENCE_LATENT_OPERATION_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    register_media_types,
    register_resident_type,
)


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    register_media_types(registry)
    return (
        registry.spec(INFERENCE_LATENT_OPERATION_TYPE)
        if INFERENCE_LATENT_OPERATION_TYPE in registry
        else register_resident_type(
            registry,
            INFERENCE_LATENT_OPERATION_TYPE,
            table=default_pool(),
            meta=comfy_resident_meta,
        ),
    )
