from __future__ import annotations

import importlib
from typing import Any, cast

from ..family_registry import load_component
from ..native_residency import NativeComponentHandle


class CodecAdapter:
    sequence_content = True

    def __init__(self, value: object) -> None:
        value = load_component(value, "vae", "vae")
        recipe = value.recipe
        assert recipe is not None
        descriptor = importlib.import_module("dinkster_native.native_arm")._component_descriptor(
            recipe.family_id
        )
        latent = descriptor.family.latent
        streams: dict[str, Any] = dict(getattr(latent, "streams", ()))
        if streams:
            latent = streams.get("video")
        if latent is None or latent.dimensions != 3 or not latent.temporal_causal:
            raise TypeError("vae requires a declared causal video latent geometry")
        self._handle = value
        self._resource_identity = value.resource_identity
        module = cast("Any", value.component)
        for method in ("encode", "decode"):
            if not callable(getattr(module, method, None)):
                raise TypeError(f"vae component requires callable {method}")
        config = getattr(module, "config", None)
        for field, expected in (
            ("latent_channels", latent.channels),
            ("spatial_ratio", latent.spatial_downscale),
            ("temporal_ratio", latent.temporal_downscale),
        ):
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0 or dimension != expected:
                raise TypeError(f"vae config.{field} must match declared video latent geometry")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime_type = (
            inference_torch.LTXAVVideoCodecRuntime
            if "audio" in streams
            else inference_torch.LTXVVideoCodecRuntime
        )
        native = importlib.import_module("dinkster_native.native_arm")
        self._runtime = runtime_type(
            module, compute_dtype=native._torch_dtype(native._torch(), recipe.knobs.vae_dtype)
        )
        self.descriptor = self._runtime.codec.descriptor
        for field in ("channels", "spatial_downscale", "temporal_downscale"):
            if getattr(self.descriptor.latent, field) != getattr(latent, field):
                raise TypeError(f"vae codec requires matching declared latent.{field}")
        self.load_device = value.load_device

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
        return self._runtime.decode_latent(latent)

    def decode_latent_tiled(
        self, latent: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.codec.decode_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._runtime.encode_content(content)

    def encode_content_tiled(
        self, content: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.codec.encode_tiled(content, tile=tile, overlap=overlap)
