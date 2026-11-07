"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .depth_anything_3 import DA3GeometryToMesh, DA3Inference, DA3Render, LoadDA3Model

NODES = (
    DA3GeometryToMesh,
    DA3Inference,
    DA3Render,
    LoadDA3Model,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
