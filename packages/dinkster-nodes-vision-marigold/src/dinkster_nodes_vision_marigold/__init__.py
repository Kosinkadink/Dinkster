"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    TypeRegistry,
    inference_nodes,
    register_media_types,
)

from .marigold import MarigoldV2PostProcess

NODES = (MarigoldV2PostProcess,)


def register_types(registry: TypeRegistry) -> None:
    register_media_types(registry)


inference_nodes(NODES)
