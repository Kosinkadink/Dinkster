"""Latent-upscale consumer owning its public inference types."""

from dinkster_api.v1 import register_inference_types
from test_latent_upscale_loader import LatentUpscaleConsumer

NODES = (LatentUpscaleConsumer,)

__all__ = ["NODES", "register_inference_types"]
