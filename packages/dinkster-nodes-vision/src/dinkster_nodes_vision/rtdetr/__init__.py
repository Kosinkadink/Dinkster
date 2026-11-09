"""RT-DETR execution provider for Dinkster's stable object detection schema."""

from .nodes import RTDETR_PROVIDER_NODES, DetectObjects, register_types

__all__ = ["RTDETR_PROVIDER_NODES", "DetectObjects", "register_types"]
