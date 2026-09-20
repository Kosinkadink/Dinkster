from __future__ import annotations

import importlib
from typing import Any

from ..family_registry import encode_component_text
from ..family_registry import load_component as load_registered_component
from ..native_residency import NativeComponentHandle
from .kl import CodecAdapter as KLCodecAdapter


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    expected = role or "text"
    try:
        handle = load_registered_component(value, name, role)
    except TypeError as error:
        raise TypeError(f"{name} must be a native Flux2 {expected} component") from error
    recipe = handle.recipe
    assert recipe is not None
    inference = importlib.import_module("dinkster_inference")
    bound_role = recipe.sources[0].role
    valid = (
        recipe.family_id == inference.FLUX2_SHARED_COMPONENT_FAMILY_ID and bound_role == "vae"
        if role == "vae"
        else inference.FLUX2_TEXT_ROLE_BY_FAMILY.get(recipe.family_id) == bound_role
    )
    if not valid:
        raise TypeError(f"{name} must be a native Flux2 {expected} component")
    return handle


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:Flux2TextRuntime",
        carrier="dinkster_inference_torch:basic_conditioning_to_carrier",
    )


class CodecAdapter(KLCodecAdapter):
    def __init__(self, value: object) -> None:
        super().__init__(load_component(value, "vae", "vae"), "Flux2")
