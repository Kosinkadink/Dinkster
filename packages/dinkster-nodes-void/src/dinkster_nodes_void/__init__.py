"""First-party void model nodes."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    NOISE_TYPE,
    OPTICAL_FLOW_TYPE,
    SAMPLER_TYPE,
    VAE_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    default_pool,
    register_media_types,
    register_resident_type,
)
from dinkster_inference_wire.conditioning_wire import register_conditioning_type
from dinkster_inference_wire.sampling_wire import register_sampling_type

from .void_conditioning import (
    VOIDInpaintConditioning,
    VOIDQuadmaskPreprocess,
    VOIDWarpedNoiseSource,
)
from .void_sampler import VOIDSampler
from .void_warped_noise import VOIDWarpedNoise

NODES = (
    VOIDInpaintConditioning,
    VOIDQuadmaskPreprocess,
    VOIDSampler,
    VOIDWarpedNoise,
    VOIDWarpedNoiseSource,
)


def register_types(registry: TypeRegistry) -> None:
    """Install consumed canonical codecs in this pack's isolated worker."""
    register_media_types(registry)
    if CONDITIONING_TYPE not in registry:
        register_conditioning_type(registry, resident_table=default_pool())
    for type_id in (NOISE_TYPE, SAMPLER_TYPE):
        register_sampling_type(registry, type_id)
    for type_id in (VAE_TYPE, OPTICAL_FLOW_TYPE):
        if type_id not in registry:
            register_resident_type(
                registry, type_id, table=default_pool(), meta=comfy_resident_meta
            )


__all__ = ["NODES", "register_types"]
