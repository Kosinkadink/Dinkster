"""Manifest entry points for native execution."""

from __future__ import annotations

from dinkster_schema import Node
from dinkster_values import TypeRegistry

from .fork_nodes import FORK_NODES
from .native import register_native_types
from .nodes_model3d import (
    GenerationApplyTextureToMesh,
    GenerationBakeAmbientOcclusion,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeTextureFromVoxel,
    GenerationDecimateMesh,
    GenerationGetMeshInfo,
    GenerationImageCropToMask,
    GenerationMeshToModel3D,
    GenerationPaintMesh,
    GenerationPreviewMask,
    GenerationRemeshMesh,
    GenerationRenderUVAtlas,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationVoxelToMesh,
)

_MESH_NODES: tuple[type[Node], ...] = (
    GenerationImageCropToMask,
    GenerationPreviewMask,
    GenerationVoxelToMesh,
    GenerationGetMeshInfo,
    GenerationRemeshMesh,
    GenerationDecimateMesh,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationPaintMesh,
    GenerationBakeTextureFromVoxel,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeAmbientOcclusion,
    GenerationRenderUVAtlas,
    GenerationApplyTextureToMesh,
    GenerationMeshToModel3D,
)
NATIVE_NODES: tuple[type[Node], ...] = (*FORK_NODES, *_MESH_NODES)
_SCHEMA_PROVIDER_IDS = {
    "dinkster.load_checkpoint",
    "dinkster.load_diffusion_model",
    "dinkster.clip_text_encode",
    "dinkster.empty_latent_image",
    "dinkster.ksampler",
    "dinkster.vae_decode",
    "dinkster.load_clip",
    "dinkster.load_vae",
    "dinkster.empty_minimax_h3_av",
    "dinkster.minimax_h3_t2va_conditioning",
    "dinkster.minimax_h3_image_to_video",
    "dinkster.separate_av_latent",
    "dinkster.vae_decode_audio",
    *(node.schema().node_type for node in _MESH_NODES),
}
ARM_NODES = {
    "native": tuple(
        node for node in NATIVE_NODES if node.schema().node_type in _SCHEMA_PROVIDER_IDS
    )
}


def register_types(registry: TypeRegistry) -> None:
    register_native_types(registry)
