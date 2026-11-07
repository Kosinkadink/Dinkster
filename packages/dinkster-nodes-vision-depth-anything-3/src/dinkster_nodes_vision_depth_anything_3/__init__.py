"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    DA3_GEOMETRY_TYPE,
    DA3_MODEL_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_asset_type,
    register_media_types,
    register_mesh_type,
    register_resident_type,
    register_vision_packet_types,
    resolver_from_env,
)

from .depth_anything_3 import DA3GeometryToMesh, DA3Inference, DA3Render, LoadDA3Model

NODES = (
    DA3GeometryToMesh,
    DA3Inference,
    DA3Render,
    LoadDA3Model,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_mesh_type(registry)
    if DA3_MODEL_TYPE not in registry:
        register_resident_type(
            registry, DA3_MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta
        )
    register_vision_packet_types(registry, type_ids=(DA3_GEOMETRY_TYPE,))


inference_nodes(NODES)
