"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

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
    register_inference_types(registry)
