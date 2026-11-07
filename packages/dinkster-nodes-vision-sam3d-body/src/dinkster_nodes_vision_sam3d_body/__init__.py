"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    TypeRegistry,
    register_asset_type,
    register_inference_types,
    resolver_from_env,
)

from .sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth
from .sam3d_export import BuildPoseFile
from .sam3d_expression import SAM3DBody_FaceExpression
from .sam3d_prediction import SAM3DBody_Predict
from .sam3d_render import SAM3DBody_Render

NODES = (
    BuildPoseFile,
    SAM3DBody_FaceExpression,
    SAM3DBody_Loader,
    SAM3DBody_Predict,
    SAM3DBody_Render,
    SAM3DBody_Smooth,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_inference_types(registry)
