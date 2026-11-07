"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

from .background_removal import LoadBackgroundRemovalModel, RemoveBackground

NODES = (
    LoadBackgroundRemovalModel,
    RemoveBackground,
)


def register_types(registry: TypeRegistry) -> None:
    register_inference_types(registry)
