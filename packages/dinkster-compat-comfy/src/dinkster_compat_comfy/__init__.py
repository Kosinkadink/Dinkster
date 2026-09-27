"""ComfyUI application-node compatibility."""

from .adapters import (
    COMFY_INPUT_ADAPTERS,
    adapt_save_image_inputs,
    make_load_checkpoint_adapter,
    make_load_clip_adapter,
    make_load_diffusion_model_adapter,
    make_load_dual_clip_adapter,
    make_load_image_adapter,
    make_load_latent_adapter,
    make_load_lora_adapter,
    make_load_model_patch_adapter,
    make_load_vae_adapter,
    make_load_vision_adapter,
    make_model_asset_inputs_adapter,
)
from .devices import comfy_resident_meta, torch_vram_telemetry, vram_telemetry_snapshot
from .legacy import LegacyPackReport, load_legacy_pack, load_legacy_packs
from .native import NATIVE_NODES, merge_native_nodes, register_native_types
from .pool import ResidentPool, VaeSource, default_pool, memory_consumers
from .prompt import (
    InputAdapter,
    PromptProblem,
    PromptTranslationError,
    extract_prompt,
    translate_prompt,
)
from .resident import (
    DEFAULT_RESIDENT_V1_TYPES,
    ResidencyTable,
    ResidentLookupError,
    register_resident_type,
    resident_resource_id,
)
from .translate import (
    COMFY_TYPE_PREFIX,
    CompatError,
    CompatTranslation,
    comfy_type_id,
    translate_mappings,
    translate_node,
    translate_type,
)
from .translate_v3 import translate_v3_schema, translate_v3_type

__all__ = [
    "COMFY_INPUT_ADAPTERS",
    "COMFY_TYPE_PREFIX",
    "DEFAULT_RESIDENT_V1_TYPES",
    "NATIVE_NODES",
    "CompatError",
    "CompatTranslation",
    "InputAdapter",
    "LegacyPackReport",
    "PromptProblem",
    "PromptTranslationError",
    "ResidencyTable",
    "ResidentLookupError",
    "ResidentPool",
    "VaeSource",
    "adapt_save_image_inputs",
    "comfy_type_id",
    "comfy_resident_meta",
    "default_pool",
    "extract_prompt",
    "load_legacy_pack",
    "load_legacy_packs",
    "make_load_checkpoint_adapter",
    "make_load_clip_adapter",
    "make_load_diffusion_model_adapter",
    "make_load_dual_clip_adapter",
    "make_load_image_adapter",
    "make_load_latent_adapter",
    "make_load_lora_adapter",
    "make_load_model_patch_adapter",
    "make_load_vae_adapter",
    "make_load_vision_adapter",
    "make_model_asset_inputs_adapter",
    "memory_consumers",
    "merge_native_nodes",
    "register_native_types",
    "register_resident_type",
    "resident_resource_id",
    "torch_vram_telemetry",
    "translate_mappings",
    "translate_node",
    "translate_prompt",
    "translate_type",
    "translate_v3_schema",
    "translate_v3_type",
    "vram_telemetry_snapshot",
]
