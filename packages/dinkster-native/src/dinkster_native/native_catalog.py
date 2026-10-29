"""Catalog for execution providers that do not depend on the ComfyUI runtime."""

from types import MappingProxyType

# Runtime symbols and schema IDs share one registration without importing bodies.
NATIVE_SCHEDULING_NODE_TYPES = MappingProxyType(
    {
        "NativeCreateHookLora": "dinkster.create_hook_lora",
        "NativeCreateHookKeyframe": "dinkster.create_hook_keyframe",
        "NativeSetHookKeyframes": "dinkster.set_hook_keyframes",
        "NativeConditioningTimestepsRange": "dinkster.conditioning_timesteps_range",
        "NativeConditioningSetPropertiesAndCombine": (
            "dinkster.conditioning_set_properties_and_combine"
        ),
        "NativePairConditioningSetProperties": "dinkster.pair_conditioning_set_properties",
    }
)
NATIVE_SCHEDULING_SOURCE_NODE_NAMES = MappingProxyType(
    {
        "NativeCreateHookLora": "CreateHookLora",
        "NativeCreateHookKeyframe": "CreateHookKeyframe",
        "NativeSetHookKeyframes": "SetHookKeyframes",
        "NativeConditioningTimestepsRange": "ConditioningTimestepsRange",
        "NativeConditioningSetPropertiesAndCombine": "ConditioningSetPropertiesAndCombine",
        "NativePairConditioningSetProperties": "PairConditioningSetProperties",
    }
)

COMFY_RUNTIME_NODE_IDS: frozenset[str] = frozenset()
