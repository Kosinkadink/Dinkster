"""Model-patch consumer with its own public inference type registration."""

from dinkster_api.v1 import register_inference_types
from test_model_patch_loader_torch import MODEL_PATCH_FIXTURES

__all__ = ["MODEL_PATCH_FIXTURES", "register_inference_types"]
