"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    BACKGROUND_REMOVAL_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_media_types,
    register_resident_type,
    resolver_from_env,
)

from .background_removal import LoadBackgroundRemovalModel, RemoveBackground

NODES = (
    LoadBackgroundRemovalModel,
    RemoveBackground,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    if BACKGROUND_REMOVAL_TYPE not in registry:
        register_resident_type(
            registry, BACKGROUND_REMOVAL_TYPE, table=default_pool(), meta=comfy_resident_meta
        )


inference_nodes(NODES)
