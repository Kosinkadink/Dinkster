"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .rtdetr import DrawBBoxes, RTDETR_detect

NODES = (
    DrawBBoxes,
    RTDETR_detect,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
