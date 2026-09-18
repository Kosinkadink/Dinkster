"""Depth Anything 3 execution provider for Dinkster's stable depth schema."""

from .nodes import DEPTH_ANYTHING_V3_PROVIDER_NODES, ModelDepthPreprocessor, register_types

__all__ = [
    "DEPTH_ANYTHING_V3_PROVIDER_NODES",
    "ModelDepthPreprocessor",
    "register_types",
]
