from __future__ import annotations

import importlib
from typing import Any, cast

from ..family_registry import encode_component_text, load_component
from ..native_residency import NativeComponentHandle


def is_runtime_family(family_id: str) -> bool:
    return family_id in ("dinkster.wan21", "dinkster.wan22")


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:Wan21TextRuntime",
        carrier="dinkster_inference_torch:wan21_text_conditioning_to_carrier",
        role="umt5xxl",
        t5_options=True,
        omit_empty_t5_options=True,
    )


class CodecAdapter:
    sequence_content = True

    def __init__(self, value: object) -> None:
        self._handle = load_component(value, "vae", "vae")
        recipe = self._handle.recipe
        assert recipe is not None
        self._resource_identity = self._handle.resource_identity
        self.descriptor = importlib.import_module("dinkster_inference").WAN21_CODEC
        self.load_device = self._handle.load_device
        native = importlib.import_module("dinkster_native.native_arm")
        self._compute_dtype = native._torch_dtype(native._torch(), recipe.knobs.vae_dtype)

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
        module = cast("Any", self._handle.component)
        decoded = module.decode(latent.to(dtype=self._compute_dtype))
        if decoded.ndim != 5 or decoded.shape[1] != self.descriptor.content_channels:
            raise ValueError("Wan 2.1 VAE decode must return [batch,3,frames,height,width]")
        torch = importlib.import_module("dinkster_native.native_arm")._torch()
        return ((decoded.to(decoded.device, dtype=torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)

    def encode_content(self, content: Any) -> Any:
        if content.ndim != 5 or content.shape[1] != self.descriptor.content_channels:
            raise ValueError("Wan 2.1 content must have shape [batch,3,frames,height,width]")
        module = cast("Any", self._handle.component)
        return module.encode((content * 2.0 - 1.0).to(dtype=self._compute_dtype))
