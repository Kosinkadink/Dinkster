from dinkster_api.v1 import register_inference_types
from test_noise_torch import _Consumer, _Source

NODES = (_Source, _Consumer)

__all__ = ["NODES", "register_inference_types"]
