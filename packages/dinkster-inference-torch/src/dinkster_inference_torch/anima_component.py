"""Strict loading for independently supplied Anima components."""

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
    AnimaComponentAssemblyError,
    AnimaComponentRole,
    anima_component_runtime_identity,
    plan_anima_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .anima_model import AnimaModel
from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .qwen_text import QwenTextModel

if TYPE_CHECKING:
    from .attention import AttentionKernel, AttentionRole

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class AnimaLoadedComponent:
    """One independently verified and strict-loaded Anima component."""

    role: AnimaComponentRole
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


def realize_anima_component(
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
            AnimaModel
            if attention_kernels is None
            else partial(
                AnimaModel,
                attention_kernel=attention_kernels["flux"],
                adapter_attention_kernel=attention_kernels["qwen"],
            )
        ),
        "qwen3_06b": (
            QwenTextModel
            if attention_kernels is None
            else partial(QwenTextModel, attention_kernel=attention_kernels["qwen"])
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


def load_anima_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: AnimaComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> AnimaLoadedComponent:
    """Verify, plan, identity-check, and strict-load one Anima component."""

    if type(asset) is not AssetRef:
        raise TypeError("Anima component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Anima component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Anima component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise AnimaComponentAssemblyError(
            f"Anima {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise AnimaComponentAssemblyError(
                f"Anima {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise AnimaComponentAssemblyError(
                f"Anima {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_anima_split_component(pinned, role=expected_role, path=path)
        runtime_identity = anima_component_runtime_identity(planned, expected_role, identity_dtype)
        if runtime_identity != expected_identity:
            raise AnimaComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = realize_anima_component(
            planned,
            compute_dtype=compute_dtype,
            source_file=handle,
            source=source,
        )
    return AnimaLoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = [
    "AnimaLoadedComponent",
    "load_anima_component",
    "realize_anima_component",
]
