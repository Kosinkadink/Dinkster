"""Single-image upscaling and its resident model value."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    UPSCALE_MODEL_TYPE,
    TypeRegistry,
    TypeSpec,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_media_types,
    register_resident_type,
    resolver_from_env,
)

from .image_model_upscale import ImageUpscaleWithModel
from .upscale_model_loader import UpscaleModelLoader

NODES = (UpscaleModelLoader, ImageUpscaleWithModel)


def register_types(registry: TypeRegistry) -> tuple[TypeSpec, ...]:
    """Keep model contents in the owning worker and carry only resident identity."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    return (
        registry.spec(UPSCALE_MODEL_TYPE)
        if UPSCALE_MODEL_TYPE in registry
        else register_resident_type(
            registry, UPSCALE_MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta
        ),
    )


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
