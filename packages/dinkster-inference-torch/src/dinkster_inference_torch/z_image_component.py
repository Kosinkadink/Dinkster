"""Strict loading for independently supplied latent Z-Image diffusion."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    ZImageComponentAssemblyError,
    plan_z_image_split_component,
    z_image_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry
from dinkster_inference.z_image import ZImageConfig

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .z_image import ZImage

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class ZImageLoadedComponent:
    """One independently verified and strict-loaded Z-Image diffusion component."""

    role: str
    module: ZImage
    plan: ComponentPlan[ZImageConfig]
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


def load_z_image_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: str,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> ZImageLoadedComponent:
    """Verify, plan, identity-check, and strict-load latent Z-Image diffusion."""
    if type(asset) is not AssetRef:
        raise TypeError("Z-Image component asset must be an AssetRef")
    if expected_role != "diffusion":
        raise ZImageComponentAssemblyError("Z-Image component role must be diffusion")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Z-Image component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Z-Image compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise ZImageComponentAssemblyError(
            f"Z-Image diffusion artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise ZImageComponentAssemblyError(
                f"Z-Image diffusion artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise ZImageComponentAssemblyError(
                "Z-Image diffusion byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_z_image_split_component(pinned, role=expected_role, path=path)
        runtime_identity = z_image_component_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise ZImageComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = _load_component(
            planned,
            ZImage,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
    return ZImageLoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = ["ZImageLoadedComponent", "load_z_image_component"]
