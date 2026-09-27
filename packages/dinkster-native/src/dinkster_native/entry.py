"""Manifest entry points for native execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
ARM_NODES = {"native": NATIVE_NODES}


def combo_choices() -> Mapping[str, Sequence[str]]:
    from dinkster_nodes_generation.nodes import SAMPLER_CHOICES, SCHEDULER_CHOICES

    return {"comfy.samplers": SAMPLER_CHOICES, "comfy.schedulers": SCHEDULER_CHOICES}


def register_types(registry: TypeRegistry) -> None:
    register_native_types(registry)
