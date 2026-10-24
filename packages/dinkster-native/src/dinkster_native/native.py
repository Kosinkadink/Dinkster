"""Fork-backed native node and value-type registration."""

from __future__ import annotations

from collections.abc import Iterable

from dinkster_assets import (
    ASSET_TYPE,
    SAVE_TARGET_TYPE,
    register_asset_type,
    register_save_target_type,
    resolver_from_env,
)
from dinkster_inference_wire import register_inference_types
from dinkster_inference_wire.sampling_wire import register_sampling_type
from dinkster_schema import Node
from dinkster_values import (
    AUDIO_TYPE,
    CLIP_TYPE,
    IMAGE_TYPE,
    INFERENCE_LATENT_OPERATION_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL3D_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    VAE_TYPE,
    VIDEO_TYPE,
    TypeRegistry,
    register_curve_type,
    register_model3d_type,
)

from .audio import register_audio_type
from .devices import comfy_resident_meta
from .fork_nodes import FORK_NODES
from .image import (
    register_image_asset_providers,
    register_image_type,
)
from .latent import register_latent_type
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
from .pool import default_pool
from .resident import register_resident_type
from .video import register_video_type

NATIVE_NODES: tuple[type[Node], ...] = (
    *FORK_NODES,
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

# These schema packs are composed in the host process, while the compatibility
# worker translates ComfyUI application nodes in isolation. Keep their legacy
# names here so the translated twins do not create ambiguous prompt aliases.
FOUNDATION_CLAIMED_V1_NAMES: tuple[str, ...] = (
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PrimitiveBoolean",
    "CreateList",
    "ComfyMathExpression",
)
MEDIA_IO_CLAIMED_V1_NAMES: tuple[str, ...] = (
    "LoadImage",
    "SaveImage",
    "CreateVideo",
    "SaveVideo",
)


def merge_native_nodes(translated: Iterable[type[Node]]) -> tuple[type[Node], ...]:
    """Replace translated nodes claimed by the fork-backed native nodes."""
    claimed = {
        f"comfy.{name}" for name in (*FOUNDATION_CLAIMED_V1_NAMES, *MEDIA_IO_CLAIMED_V1_NAMES)
    }
    for node in NATIVE_NODES:
        schema = node.schema()
        claimed.add(schema.node_type)
        claimed.update(f"comfy.{alias}" for alias in schema.aliases)
    return (
        *(node for node in translated if node.schema().node_type not in claimed),
        *NATIVE_NODES,
    )


def _register_resident(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        register_resident_type(
            registry,
            type_id,
            table=default_pool(),
            meta=comfy_resident_meta,
        )


def register_native_types(registry: TypeRegistry) -> None:
    """Register the wire and resident types used by native execution."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)

    for type_id in (
        MODEL_TYPE,
        CLIP_TYPE,
        VAE_TYPE,
        MODEL_PATCH_TYPE,
        INFERENCE_LATENT_OPERATION_TYPE,
    ):
        _register_resident(registry, type_id)

    if LATENT_TYPE not in registry:
        register_latent_type(registry, LATENT_TYPE)

    for type_id in (SAMPLER_TYPE, SIGMAS_TYPE, NOISE_TYPE):
        register_sampling_type(registry, type_id)

    register_inference_types(registry)
    register_curve_type(registry)
    register_model3d_type(registry, MODEL3D_TYPE)

    for type_id in (IMAGE_TYPE, MASK_TYPE):
        if type_id not in registry:
            register_image_type(registry, type_id)
    if registry.asset_decoder_for(IMAGE_TYPE) is None:
        register_image_asset_providers(registry, IMAGE_TYPE)

    if AUDIO_TYPE not in registry:
        register_audio_type(registry, AUDIO_TYPE)
    if VIDEO_TYPE not in registry:
        register_video_type(registry, VIDEO_TYPE)
