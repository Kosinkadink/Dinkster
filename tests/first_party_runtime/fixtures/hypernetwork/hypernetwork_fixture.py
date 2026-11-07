"""Model producer and attention consumer with their own value registration."""

from dinkster_api.v1 import register_inference_types
from test_hypernetwork_torch import HYPERNETWORK_FIXTURES

__all__ = ["HYPERNETWORK_FIXTURES", "register_inference_types"]
