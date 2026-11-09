"""LoRA-hook fixture using the model producer's codecs."""

from dinkster_api.v1 import TypeRegistry
from dinkster_models.types import register_types

from tests.first_party_runtime.test_hook_lora import LORA_HOOK_FIXTURES


def register_inference_types(registry: TypeRegistry) -> None:
    with registry.registering_pack("dinkster-models", (), allow_bare=True):
        register_types(registry)


__all__ = ["LORA_HOOK_FIXTURES", "register_inference_types"]
