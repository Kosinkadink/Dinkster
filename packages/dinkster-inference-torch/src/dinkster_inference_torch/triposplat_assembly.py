"""Strict direct-import loading for independently supplied TripoSplat components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    TripoSplatComponentAssemblyError,
    TripoSplatComponentRole,
    plan_triposplat_split_component,
    triposplat_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .dinov3 import DINOv3ViTModel
from .triposplat_decoder import OctreeGaussianDecoder
from .triposplat_model import TripoSplatModel

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class TripoSplatLoadedComponent:
    """One independently verified and strict-loaded TripoSplat component."""

    role: TripoSplatComponentRole
    family_id: str
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

    @property
    def entries(self) -> Mapping[str, WeightEntry]:
        return self.source.entries

    def keys(self) -> tuple[str, ...]:
        return tuple(self.source.keys())

    def entry(self, key: str) -> WeightEntry:
        return self.source.entry(key)

    def metadata(self) -> Mapping[str, str]:
        return self.source.metadata()


def load_triposplat_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: TripoSplatComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> TripoSplatLoadedComponent:
    """Verify, plan, identity-check, and strict-load one TripoSplat component."""

    if type(asset) is not AssetRef:
        raise TypeError("TripoSplat component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("TripoSplat component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError(
            f"TripoSplat {expected_role} compute dtype must be bfloat16, float16, or float32"
        )
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise TripoSplatComponentAssemblyError(
            f"TripoSplat {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise TripoSplatComponentAssemblyError(
                f"TripoSplat {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise TripoSplatComponentAssemblyError(
                f"TripoSplat {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_triposplat_split_component(pinned, role=expected_role, path=path)
        runtime_identity = triposplat_component_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise TripoSplatComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        builder = {
            "dit": TripoSplatModel,
            "dinov3-vision-conditioner": DINOv3ViTModel,
            "gaussian-decoder": OctreeGaussianDecoder,
        }[expected_role]
        module = _load_component(
            planned.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=cast("SafetensorsSource", pinned),
        )
    return TripoSplatLoadedComponent(
        expected_role, planned.family_id, module, planned.plan, runtime_identity
    )


__all__ = [
    "TripoSplatLoadedComponent",
    "load_triposplat_component",
]
