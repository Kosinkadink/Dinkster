"""LoRA-hook fixture with independent public type registration."""

from dinkster_api.v1 import register_inference_types
from test_hook_lora import LORA_HOOK_FIXTURES

__all__ = ["LORA_HOOK_FIXTURES", "register_inference_types"]
