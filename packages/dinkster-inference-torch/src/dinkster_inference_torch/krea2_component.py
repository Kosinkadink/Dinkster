"""Strict loading for independently supplied Krea 2 components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    Krea2ComponentAssemblyError,
    Krea2ComponentRole,
    krea2_component_runtime_identity,
    plan_krea2_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.krea2_text import KREA2_TEXT_CONFIG
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .krea2_conditioner import krea2_language_model
from .krea2_dit import Krea2DiT
from .operations import Operations
from .qwen_image_text import QwenImageLanguageModel

if TYPE_CHECKING:
    from .attention import AttentionKernel, AttentionRole

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class Krea2LoadedComponent:
    """One independently verified and strict-loaded Krea 2 component."""

    role: Krea2ComponentRole
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


def _build_krea2_text(
    config: object,
    *,
    operations: Operations,
    attention_kernel: AttentionKernel | None = None,
) -> QwenImageLanguageModel:
    if config != KREA2_TEXT_CONFIG:
        raise Krea2ComponentAssemblyError(
            "Krea 2 text plan does not carry the exact supported profile"
        )
    if attention_kernel is None:
        return krea2_language_model(operations=operations)
    return krea2_language_model(operations=operations, attention_kernel=attention_kernel)


def realize_krea2_component(
    plan: ComponentPlan[object],
    *,
    compute_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    fp8_matmul: bool = False,
    attention_kernels: Mapping[AttentionRole, AttentionKernel] | None = None,
) -> torch.nn.Module:
    """Construct the selected architecture from its validated weight plan."""
    builder = {
        "diffusion": (
            Krea2DiT
            if attention_kernels is None
            else partial(Krea2DiT, attention_kernel=attention_kernels["flux"])
        ),
        "qwen3vl_4b": (
            _build_krea2_text
            if attention_kernels is None
            else partial(_build_krea2_text, attention_kernel=attention_kernels["qwen"])
        ),
    }[plan.component]
    return _load_component(
        plan,
        builder,
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
        source_file=source_file,
        source=source,
    )


def load_krea2_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Krea2ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Krea2LoadedComponent:
    """Verify, plan, identity-check, and strict-load one Krea 2 component."""

    if type(asset) is not AssetRef:
        raise TypeError("Krea 2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Krea 2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Krea 2 component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Krea2ComponentAssemblyError(
            f"Krea 2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Krea2ComponentAssemblyError(
                f"Krea 2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Krea2ComponentAssemblyError(
                f"Krea 2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_krea2_split_component(pinned, role=expected_role, path=path)
        runtime_identity = krea2_component_runtime_identity(planned, expected_role, identity_dtype)
        if runtime_identity != expected_identity:
            raise Krea2ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = realize_krea2_component(
            planned,
            compute_dtype=compute_dtype,
            source_file=handle,
            source=source,
        )
    return Krea2LoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = [
    "Krea2LoadedComponent",
    "load_krea2_component",
    "realize_krea2_component",
]
