"""Strict direct-import loading for supported Wav2Vec2 profiles."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    Wav2Vec2ComponentAssemblyError,
    Wav2Vec2Config,
    plan_wav2vec2_component,
    wav2vec2_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .wav2vec2 import Wav2Vec2Model

_IDENTITY_DTYPES = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class LoadedWav2Vec2:
    """One independently verified and strict-loaded Wav2Vec2 component."""

    module: Wav2Vec2Model
    plan: ComponentPlan[Wav2Vec2Config]
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

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key)


def load_wav2vec2_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> LoadedWav2Vec2:
    """Verify, plan, identity-check, and strict-load one Wav2Vec2 profile."""

    if type(asset) is not AssetRef:
        raise TypeError("Wav2Vec2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Wav2Vec2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Wav2Vec2 compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Wav2Vec2ComponentAssemblyError(
            f"Wav2Vec2 artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Wav2Vec2ComponentAssemblyError(
                f"Wav2Vec2 artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Wav2Vec2ComponentAssemblyError(
                "Wav2Vec2 byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        plan = plan_wav2vec2_component(pinned, path=path)
        identity = wav2vec2_component_runtime_identity(plan, identity_dtype)
        if identity != expected_identity:
            raise Wav2Vec2ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, constructed {identity!r}"
            )
        module = _load_component(
            plan,
            Wav2Vec2Model,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
    if module.config != plan.config:
        raise Wav2Vec2ComponentAssemblyError("loaded Wav2Vec2 component changed profile")
    return LoadedWav2Vec2(module, plan, identity)


__all__ = ["LoadedWav2Vec2", "load_wav2vec2_component"]
