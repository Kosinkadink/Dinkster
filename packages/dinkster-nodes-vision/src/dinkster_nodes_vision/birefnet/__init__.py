"""BiRefNet execution provider for Dinkster's stable foreground-matte schema."""

from .nodes import BIREFNET_PROVIDER_NODES, ImageMatte, register_types

__all__ = ["BIREFNET_PROVIDER_NODES", "ImageMatte", "register_types"]
