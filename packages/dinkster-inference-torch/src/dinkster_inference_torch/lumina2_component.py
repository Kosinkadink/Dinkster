"""Strict loading for independently supplied Lumina2 components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    Lumina2ComponentAssemblyError,
    Lumina2ComponentRole,
    lumina2_component_runtime_identity,
    plan_lumina2_artifact_components,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .autoencoder_kl import AutoencoderKL
from .gemma_text import GemmaTextModel
from .gemma_tokenizer import LUMINA2_TOKENIZER_ATTRIBUTE, LUMINA2_TOKENIZER_BYTE_CAP
from .z_image import ZImage

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class Lumina2LoadedComponent:
    """One independently verified and strict-loaded Lumina2 component."""

    role: Lumina2ComponentRole
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


def load_lumina2_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Lumina2ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Lumina2LoadedComponent:
    """Verify, plan, identity-check, and strict-load one Lumina2 component."""
    if type(asset) is not AssetRef:
        raise TypeError("Lumina2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Lumina2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Lumina2 component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Lumina2ComponentAssemblyError(
            f"Lumina2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Lumina2ComponentAssemblyError(
                f"Lumina2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Lumina2ComponentAssemblyError(
                f"Lumina2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        components = dict(plan_lumina2_artifact_components(pinned, path=path))
        planned = components.get(expected_role)
        if planned is None:
            raise Lumina2ComponentAssemblyError(
                f"Lumina2 {expected_role} loader cannot consume artifact roles "
                f"{tuple(components)!r}"
            )
        runtime_identity = lumina2_component_runtime_identity(
            planned, expected_role, identity_dtype
        )
        if runtime_identity != expected_identity:
            raise Lumina2ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        builder = {
            "diffusion": ZImage,
            "gemma2_2b": GemmaTextModel,
            "vae": AutoencoderKL,
        }[expected_role]
        module = _load_component(
            planned,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
        if expected_role == "gemma2_2b":
            tokenizer_key = next(
                (
                    key
                    for key in ("spiece_model", "text_encoders.spiece_model")
                    if key in pinned.keys()
                ),
                None,
            )
            if tokenizer_key is None:
                raise Lumina2ComponentAssemblyError(
                    "Lumina2 gemma2_2b source has no embedded spiece_model"
                )
            module.__dict__[LUMINA2_TOKENIZER_ATTRIBUTE] = pinned.read_uint8_configuration(
                tokenizer_key, limit=LUMINA2_TOKENIZER_BYTE_CAP
            )
    return Lumina2LoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = [
    "LUMINA2_TOKENIZER_ATTRIBUTE",
    "LUMINA2_TOKENIZER_BYTE_CAP",
    "Lumina2LoadedComponent",
    "load_lumina2_component",
]
