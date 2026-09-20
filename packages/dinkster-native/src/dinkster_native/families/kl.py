from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any, cast

from ..native_residency import NativeComponentHandle


class CodecAdapter:
    def __init__(self, handle: NativeComponentHandle, family_name: str) -> None:
        self._handle = handle
        self._family_name = family_name
        self._resource_identity = handle.resource_identity
        module = cast("Any", handle.component)
        plugin = importlib.import_module("dinkster_inference_torch").kl_codec_plugin(module)
        self._plugin = replace(plugin, compute_dtype=next(module.parameters()).dtype)
        self.descriptor = self._plugin.descriptor
        self.load_device = handle.load_device

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self._handle

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self, *, memory_required: int = 0) -> Any:
        return self._handle.stage(memory_required=memory_required, clear_cache_after=True)

    def _memory_required(self, value: Any, direction: str) -> int:
        estimator = self._plugin.memory
        if estimator is None:
            return 0
        inference = importlib.import_module("dinkster_inference")
        dtype_name = str(self._plugin.compute_dtype).removeprefix("torch.")
        try:
            dtype = {
                "float16": inference.FLOAT16,
                "bfloat16": inference.BFLOAT16,
                "float32": inference.FLOAT32,
            }[dtype_name]
        except KeyError:
            raise TypeError(f"unsupported component codec dtype {dtype_name!r}") from None
        geometry = inference.TensorGeometry(tuple(value.shape), dtype)
        return (
            estimator.decode_bytes(geometry)
            if direction == "decode"
            else estimator.encode_bytes(geometry)
        )

    def decode_memory_required(self, latent: Any) -> int:
        return self._memory_required(latent, "decode")

    def encode_memory_required(self, content: Any) -> int:
        return self._memory_required(content, "encode")

    def decode_latent(self, latent: Any) -> Any:
        native = importlib.import_module("dinkster_native.native_arm")
        return native._run_direct_vae(
            handle=self._handle,
            value=latent,
            direction="decode",
            operation=self._plugin.decode,
            codec=self._plugin,
        )

    def encode_content(self, content: Any) -> Any:
        if content.ndim != 4:
            raise ValueError(f"{self._family_name} content must have shape [batch,3,height,width]")
        native = importlib.import_module("dinkster_native.native_arm")
        return native._run_direct_vae(
            handle=self._handle,
            value=content,
            direction="encode",
            operation=self._plugin.encode,
            codec=self._plugin,
        )
