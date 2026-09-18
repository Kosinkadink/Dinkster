"""Depth Anything V2 execution provider for Dinkster's stable depth schema."""

from .nodes import DEPTH_ANYTHING_V2_PROVIDER_NODES, ModelDepthPreprocessor, register_types

__all__ = [
    "DEPTH_ANYTHING_V2_PROVIDER_NODES",
    "ModelDepthPreprocessor",
    "register_types",
]
