from __future__ import annotations

import importlib
from typing import Any

from ..family_registry import load_component as load_registered_component
from ..native_residency import NativeComponentHandle


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    expected_family = importlib.import_module("dinkster_inference").MINIMAX_MUSIC3_CONFIG.family_id
    try:
        handle = load_registered_component(value, name, role)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a native MiniMax Music 3 {role or 'component'} component"
        ) from error
    recipe = handle.recipe
    if recipe is None or recipe.family_id != expected_family:
        raise TypeError(f"{name} must be a native MiniMax Music 3 {role or 'component'} component")
    return handle


class CodecAdapter:
    def __init__(self, value: object) -> None:
        self._handle = load_component(value, "vae", "vae")
        self._resource_identity = self._handle.resource_identity
        self._plugin = importlib.import_module("dinkster_inference_torch").minimax_music3_dav_codec(
            self._handle.component
        )
        self.descriptor = self._plugin.descriptor
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
        return self._plugin.decode(latent)

    def decode_latent_tiled(
        self, latent: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        torch = importlib.import_module("dinkster_native.native_arm")._torch()
        return self._plugin.decode_tiled(
            latent,
            tile=tile,
            overlap=overlap,
            output_device="cpu",
            dtype=torch.float32,
        )
