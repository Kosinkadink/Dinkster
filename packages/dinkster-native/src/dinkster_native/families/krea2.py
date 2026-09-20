from __future__ import annotations

from typing import Any

from ..family_registry import encode_component_text


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:Krea2TextRuntime",
        carrier="dinkster_inference_torch:basic_conditioning_to_carrier",
        role="qwen3vl_4b",
    )
