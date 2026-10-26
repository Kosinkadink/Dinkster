"""Checkpoint-hook fixture with independent public inference type registration."""

from dinkster_api.v1 import register_inference_types
from test_checkpoint_hooks_torch import CHECKPOINT_HOOK_FIXTURES

__all__ = ["CHECKPOINT_HOOK_FIXTURES", "register_inference_types"]
