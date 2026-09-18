"""DETR execution provider for Dinkster's stable object detection schema."""

from .nodes import DETR_PROVIDER_NODES, DetectObjects, register_types

__all__ = [
    "DETR_PROVIDER_NODES",
    "DetectObjects",
    "register_types",
]
