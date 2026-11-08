"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    MODEL_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_box_types,
    register_media_types,
    register_resident_type,
)

from .rtdetr import DrawBBoxes, RTDETR_detect

NODES = (
    DrawBBoxes,
    RTDETR_detect,
)


def register_types(registry: TypeRegistry) -> None:
    register_media_types(registry)
    register_box_types(registry)
    if MODEL_TYPE not in registry:
        register_resident_type(registry, MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta)


inference_nodes(NODES)
