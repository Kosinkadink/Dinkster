"""LoRA-hook fixture with independent public type registration."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    TypeRegistry,
    register_asset_type,
    register_inference_types,
    resolver_from_env,
)
from test_hook_lora import LORA_HOOK_FIXTURES


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_inference_types(registry)


__all__ = ["LORA_HOOK_FIXTURES", "register_types"]
