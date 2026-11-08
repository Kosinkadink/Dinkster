from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

from .text_generation import TextGenerate

NODES = (TextGenerate,)

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
