"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .marigold import MarigoldV2PostProcess

NODES = (MarigoldV2PostProcess,)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
