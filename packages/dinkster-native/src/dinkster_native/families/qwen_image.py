from __future__ import annotations

import importlib
from typing import Any

from ..family_registry import encode_component_text, load_component
from ..native_residency import NativeComponentHandle


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:QwenImageTextRuntime",
        carrier="dinkster_inference_torch:qwen_image_conditioning_to_carrier",
        role="qwen2_5_vl_7b",
    )


class CodecAdapter:
    def __init__(self, value: object) -> None:
        self._handle = load_component(value, "vae", "vae")
        self._resource_identity = self._handle.resource_identity
        self.descriptor = importlib.import_module("dinkster_inference").WAN21_CODEC
        self.load_device = self._handle.load_device

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self._handle

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self) -> Any:
        return self._handle.stage(clear_cache_after=True)

    def decode_latent(self, latent: Any) -> Any:
        runtime = importlib.import_module("dinkster_inference_torch").WanVAECodecRuntime(
            self._handle.component,
            layered=latent.ndim == 5 and latent.shape[2] > 1,
        )
        return runtime.decode_latent(latent)

    def encode_content(self, content: Any) -> Any:
        if content.ndim == 4:
            runtime = importlib.import_module("dinkster_inference_torch").WanVAECodecRuntime(
                self._handle.component
            )
            return runtime.encode_content(content)
        raise ValueError("Qwen Image content must have shape [batch,3,height,width]")
