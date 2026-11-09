"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    inference_nodes,
    register_media_types,
    register_mesh_type,
    register_resident_type,
    register_shape_subdivides_type,
    register_splat_type,
    register_voxel_type,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type

from .latent_model_specific import EmptyTrellis2LatentStructure
from .pixal_conditioning import Pixal3DConditioning, Pixal3DMultiViewConditioning
from .trellis import Trellis2TextureStage, VaeDecodeTextureTrellis
from .trellis_conditioning import Trellis2Conditioning
from .trellis_shape import Trellis2ShapeStage, VaeDecodeShapeTrellis
from .trellis_structure import VaeDecodeStructureTrellis2
from .trellis_upsample import Trellis2UpsampleStage
from .triposplat import (
    TripoSplatConditioning,
    TripoSplatPreprocessImage,
    TripoSplatSamplingPreview,
    VAEDecodeTripoSplat,
)

NODES = (
    EmptyTrellis2LatentStructure,
    Pixal3DConditioning,
    Pixal3DMultiViewConditioning,
    Trellis2Conditioning,
    Trellis2ShapeStage,
    Trellis2TextureStage,
    Trellis2UpsampleStage,
    TripoSplatConditioning,
    TripoSplatPreprocessImage,
    TripoSplatSamplingPreview,
    VAEDecodeTripoSplat,
    VaeDecodeShapeTrellis,
    VaeDecodeStructureTrellis2,
    VaeDecodeTextureTrellis,
)


def register_types(registry: TypeRegistry) -> None:
    register_media_types(registry)
    register_mesh_type(registry)
    register_splat_type(registry)
    register_voxel_type(registry)
    register_shape_subdivides_type(registry)
    if CONDITIONING_TYPE not in registry:
        register_conditioning_type(registry, resident_table=default_pool())
    for type_id in (CLIP_VISION_TYPE, MODEL_TYPE, VAE_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )


inference_nodes(NODES)
