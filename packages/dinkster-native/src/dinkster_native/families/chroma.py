from __future__ import annotations

from typing import Any

from ..family_registry import encode_component_text, load_component
from .kl import CodecAdapter as KLCodecAdapter


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:ChromaTextRuntime",
        carrier="dinkster_inference_torch:basic_conditioning_to_carrier",
        role="t5xxl",
        t5_options=True,
        carrier_within_stage=True,
    )


class CodecAdapter(KLCodecAdapter):
    def __init__(self, value: object) -> None:
        super().__init__(load_component(value, "vae", "vae"), "Chroma")

    def decode_latent(self, latent: Any) -> Any:
        return super().decode_latent(latent).to("cpu")
