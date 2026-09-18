"""Strict direct-import loading for independently supplied Flux2 components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    Flux2ComponentAssemblyError,
    Flux2ComponentRole,
    TekkenBpe,
    flux2_component_runtime_identity,
    load_flux2_tekken_bpe,
    plan_flux2_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .autoencoder_kl import AutoencoderKL
from .flux import Flux
from .module_residency import declare_residency_materialization_ceilings
from .operations import ResidencyRouted
from .qwen_text import QwenTextModel

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}

#: Instance-dict slot carrying the tekken tokenizer extracted from the
#: dev text-encoder file (or the byte-identical vendored blob when the
#: file omits it). The dev prompt runtime requires it.
FLUX2_TEKKEN_ATTRIBUTE = "_dinkster_flux2_tekken"

# Byte cap for the checkpoint-embedded tekken_model tokenizer payload.
TEKKEN_MODEL_BYTE_CAP = 64 * 1024 * 1024


@dataclass(frozen=True)
class Flux2LoadedComponent:
    """One independently verified and strict-loaded Flux2 component."""

    role: Flux2ComponentRole
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

    def read_uint8_configuration(self, key: str, *, limit: int = 65_536) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key, limit=limit)


def load_flux2_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Flux2ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Flux2LoadedComponent:
    """Verify, plan, identity-check, and strict-load one Flux2 component."""

    if type(asset) is not AssetRef:
        raise TypeError("Flux2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Flux2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError(
            f"Flux2 {expected_role} compute dtype must be bfloat16, float16, or float32"
        )
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Flux2ComponentAssemblyError(
            f"Flux2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Flux2ComponentAssemblyError(
                f"Flux2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Flux2ComponentAssemblyError(
                f"Flux2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_flux2_split_component(pinned, role=expected_role, path=path)
        runtime_identity = flux2_component_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise Flux2ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        builder = {
            "diffusion": Flux,
            "mistral3_24b": QwenTextModel,
            "qwen3_8b": QwenTextModel,
            "qwen3_4b": QwenTextModel,
            "vae": AutoencoderKL,
        }[expected_role]
        module = _load_component(
            planned.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=cast("SafetensorsSource", pinned),
        )
        for routed in module.modules():
            if not isinstance(routed, ResidencyRouted):
                continue
            direct = tuple(routed.named_parameters(recurse=False, remove_duplicate=False)) + tuple(
                routed.named_buffers(recurse=False, remove_duplicate=False)
            )
            declare_residency_materialization_ceilings(
                routed,
                {
                    name: routed.residency_materialization_dtype(stored).itemsize
                    for name, stored in direct
                    if stored.is_floating_point() or stored.is_complex()
                },
            )
        if expected_role == "mistral3_24b":
            # The embedded tekken_model is a full tokenizer model
            # (~19 MB in the published checkpoint), far above the
            # default configuration cap for scalar settings.
            tokenizer = (
                TekkenBpe.from_tekken_bytes(
                    pinned.read_uint8_configuration("tekken_model", limit=TEKKEN_MODEL_BYTE_CAP)
                )
                if "tekken_model" in pinned.keys()
                else load_flux2_tekken_bpe()
            )
            module.__dict__[FLUX2_TEKKEN_ATTRIBUTE] = tokenizer
    return Flux2LoadedComponent(
        expected_role, planned.family_id, module, planned.plan, runtime_identity
    )


__all__ = [
    "FLUX2_TEKKEN_ATTRIBUTE",
    "Flux2LoadedComponent",
    "load_flux2_component",
]
