"""Latent-upscale consumer using the model producer's codecs."""

from dinkster_api.v1 import TypeRegistry
from dinkster_models.types import register_types

from tests.first_party_runtime.test_latent_upscale_loader import LatentUpscaleConsumer

NODES = (LatentUpscaleConsumer,)


def register_inference_types(registry: TypeRegistry) -> None:
    with registry.registering_pack("dinkster-models", (), allow_bare=True):
        register_types(registry)


__all__ = ["NODES", "register_inference_types"]
