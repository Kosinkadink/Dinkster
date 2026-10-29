"""Strict loading for independently supplied Ideogram 4 components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    IDEOGRAM4_CONFIG,
    IDEOGRAM4_TEXT_CONFIG,
    Ideogram4ComponentAssemblyError,
    Ideogram4ComponentRole,
    ideogram4_component_runtime_identity,
    ideogram4_component_uses_fp8_matmul,
    plan_ideogram4_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .ideogram4_conditioner import ideogram4_language_model
from .ideogram4_dit import Ideogram4DiT
from .operations import Operations
from .qwen_image_text import QwenImageLanguageModel

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class Ideogram4LoadedComponent:
    role: Ideogram4ComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
    runtime_identity: str


@dataclass(frozen=True)
class _PinnedSource:
    source: SafetensorsSource
    file: BinaryIO
    asset_digest: str
    asset_size: int

    @property
    def path(self) -> Path:
        return self.source.path

    def keys(self) -> tuple[str, ...]:
        return tuple(self.source.keys())

    def entry(self, key: str) -> WeightEntry:
        return self.source.entry(key)

    def metadata(self) -> Mapping[str, str]:
        return self.source.metadata()

    def read_uint8_configuration(self, key: str, *, limit: int = 65_536) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key, limit=limit)


def _build_ideogram4_text(config: object, *, operations: Operations) -> QwenImageLanguageModel:
    if config != IDEOGRAM4_TEXT_CONFIG:
        raise Ideogram4ComponentAssemblyError(
            "Ideogram 4 text plan does not carry the supported profile"
        )
    return ideogram4_language_model(operations=operations)


def _build_ideogram4_diffusion(config: object, *, operations: Operations) -> Ideogram4DiT:
    if config != IDEOGRAM4_CONFIG:
        raise Ideogram4ComponentAssemblyError(
            "Ideogram 4 diffusion plan does not carry the supported profile"
        )
    return Ideogram4DiT(operations=operations)


def load_ideogram4_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Ideogram4ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Ideogram4LoadedComponent:
    """Verify, plan, identity-check, and strict-load one Ideogram 4 component."""

    if type(asset) is not AssetRef:
        raise TypeError("Ideogram 4 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Ideogram 4 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Ideogram 4 compute dtype must be bfloat16 or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Ideogram4ComponentAssemblyError(
            f"Ideogram 4 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Ideogram4ComponentAssemblyError(
                f"Ideogram 4 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Ideogram4ComponentAssemblyError(
                f"Ideogram 4 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_ideogram4_split_component(pinned, role=expected_role, path=path)
        runtime_identity = ideogram4_component_runtime_identity(
            planned, expected_role, identity_dtype
        )
        if runtime_identity != expected_identity:
            raise Ideogram4ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        builder = {
            "diffusion": _build_ideogram4_diffusion,
            "qwen3vl_8b": _build_ideogram4_text,
        }[expected_role]
        module = _load_component(
            planned,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=ideogram4_component_uses_fp8_matmul(expected_role, planned),
            source_file=handle,
            source=source,
        )
    return Ideogram4LoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = ["Ideogram4LoadedComponent", "load_ideogram4_component"]
