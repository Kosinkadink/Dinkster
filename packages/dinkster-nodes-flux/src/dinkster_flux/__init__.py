from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .cache import FluxKVCache
from .flux_conditioning import (
    CLIPTextEncodeFlux,
    FluxDisableGuidance,
    FluxGuidance,
    FluxKontextMultiReferenceLatentMethod,
)
from .flux_image_layout import Flux3LayoutPrompt, FluxKontextImageScale
from .image_control_patches import USOStyleReference
from .latent_model_specific import EmptyFlux2LatentImage
from .sigma_schedulers import Flux2Scheduler

NODES = (
    FluxKVCache,
    CLIPTextEncodeFlux,
    FluxGuidance,
    FluxDisableGuidance,
    FluxKontextMultiReferenceLatentMethod,
    FluxKontextImageScale,
    Flux3LayoutPrompt,
    USOStyleReference,
    EmptyFlux2LatentImage,
    Flux2Scheduler,
)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
