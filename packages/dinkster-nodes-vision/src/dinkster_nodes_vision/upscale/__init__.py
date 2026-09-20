"""ESRGAN-family execution provider for Dinkster's stable upscale-model schema."""

from .nodes import UPSCALE_PROVIDER_NODES, UpscaleWithModel, register_types

__all__ = ["UPSCALE_PROVIDER_NODES", "UpscaleWithModel", "register_types"]
