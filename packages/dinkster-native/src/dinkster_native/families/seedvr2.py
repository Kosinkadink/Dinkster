from __future__ import annotations

import importlib
import logging
from typing import Any

from ..native_residency import NativeComponentHandle

log = logging.getLogger(__name__)


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    inference = importlib.import_module("dinkster_inference")
    handle = inference.require_inference_component_handle(value, name)
    recipe = getattr(handle, "recipe", None)
    if recipe is not None and recipe.runtime_identity != handle.resource_identity:
        raise TypeError(f"{name} component identity does not match its recipe")
    return handle


class CodecAdapter:
    sequence_content = True
    accepts_batched_video = True
    accepts_image_batch_latent = True
    manages_input_device = True

    def __init__(self, value: object) -> None:
        self._handle = load_component(value, "vae", "vae")
        recipe = getattr(self._handle, "recipe", None)
        dtype = None if recipe is None else recipe.knobs.vae_dtype
        if recipe is None:
            log.warning("SeedVR2 codec component has no reconstruction recipe; using VAE dtype")
        native = importlib.import_module("dinkster_native.native_arm")
        self._runtime = importlib.import_module("dinkster_inference_torch").SeedVR2CodecRuntime(
            self._handle.component,
            compute_dtype=None if dtype is None else native._torch_dtype(native._torch(), dtype),
        )
        self._resource_identity = self._handle.resource_identity
        self.descriptor = self._runtime.codec.descriptor
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
        return self._handle.stage()

    def decode_latent(self, latent: Any) -> Any:
        return self._runtime.decode_latent(latent)

    def decode_latent_tiled(
        self, latent: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.decode_latent_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._runtime.encode_content(content)

    def encode_content_tiled(
        self, content: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.encode_content_tiled(content, tile=tile, overlap=overlap)
