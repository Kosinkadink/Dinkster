"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    TypeRegistry,
    register_asset_type,
    register_inference_types,
    resolver_from_env,
)

from .moge import MoGeGeometryToFOV
from .moge_inference import MoGeInference
from .moge_mesh import MoGePointMapToMesh
from .moge_panorama import MoGePanoramaInference
from .moge_render import MoGeRender

NODES = (
    MoGeGeometryToFOV,
    MoGeInference,
    MoGePanoramaInference,
    MoGePointMapToMesh,
    MoGeRender,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_inference_types(registry)
