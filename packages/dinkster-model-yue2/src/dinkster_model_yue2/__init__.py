"""YuE2 music generation as a pack-owned model family."""

from .declarations import YUE2_FAMILY, register_inference
from .nodes import YUE2_MODEL_NODE_IDS, YUE2_MODEL_NODES

__all__ = [
    "YUE2_FAMILY",
    "YUE2_MODEL_NODE_IDS",
    "YUE2_MODEL_NODES",
    "register_inference",
]
