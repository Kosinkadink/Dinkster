from __future__ import annotations

import importlib
from typing import Any

from ..family_registry import encode_component_text
from ..family_registry import load_component as load_registered_component
from ..native_residency import NativeComponentHandle


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    expected_family = importlib.import_module("dinkster_inference").ANIMA_CONFIG.family_id
    try:
        handle = load_registered_component(value, name, role)
    except TypeError as error:
        raise TypeError(f"{name} must be a native Anima {role or 'component'} component") from error
    recipe = handle.recipe
    if recipe is None or recipe.family_id != expected_family:
        raise TypeError(f"{name} must be a native Anima {role or 'component'} component")
    return handle


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:AnimaTextRuntime",
        carrier="dinkster_inference_torch:anima_conditioning_to_carrier",
        role="qwen3_06b",
    )
