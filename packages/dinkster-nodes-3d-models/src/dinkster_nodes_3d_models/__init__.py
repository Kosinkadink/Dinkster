"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import TypeRegistry, register_inference_types

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
    register_inference_types(registry)
