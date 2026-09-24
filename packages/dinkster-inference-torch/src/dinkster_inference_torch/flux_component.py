"""Strict loading and sampling for classic Flux diffusion components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    FLUX_DEV,
    FLUX_SCHNELL,
    AttentionPolicy,
    AttentionRouteToken,
    Conditioning,
    FluxComponentAssemblyError,
    FluxComponentRole,
    ModelFamily,
    flux_component_runtime_identity,
    plan_flux_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.flux import FluxConfig
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionRole, AttentionStatus, resolve_role_attention
from .flux import Flux
from .schedules import torch_scheduler_registry
from .solvers import torch_sampler_registry
from .wiring import (
    FluxRuntime,
    WiringError,
    _flux_sigma_space,  # pyright: ignore[reportPrivateUsage]
)

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class FluxLoadedComponent:
    role: FluxComponentRole
    module: Flux
    plan: ComponentPlan[FluxConfig]
    runtime_identity: str
    attention_status: AttentionStatus


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


def _family(config: FluxConfig) -> ModelFamily:
    return FLUX_DEV if config.guidance_embed else FLUX_SCHNELL


def load_flux_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: FluxComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backend: AttentionRole,
) -> FluxLoadedComponent:
    """Verify, plan, identity-check, and strict-load one classic Flux model."""
    if type(asset) is not AssetRef:
        raise TypeError("classic Flux component asset must be an AssetRef")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("classic Flux compute dtype must be bfloat16, float16, or float32")
    attention = resolve_role_attention(
        attention_backend,
        attention_policy,
        attention_route_token,
    )
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise FluxComponentAssemblyError(
            f"classic Flux artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise FluxComponentAssemblyError(
                f"classic Flux artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise FluxComponentAssemblyError(
                "classic Flux byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_flux_split_component(pinned, role=expected_role, path=path)
        runtime_identity = flux_component_runtime_identity(
            planned,
            identity_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
        if runtime_identity != expected_identity:
            raise FluxComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = _load_component(
            planned,
            partial(Flux, attention_kernel=attention.kernel),
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
    return FluxLoadedComponent(
        expected_role,
        module,
        planned,
        runtime_identity,
        attention.status,
    )


@dataclass(frozen=True)
class _FluxDiffusionAssembly:
    diffusion: Flux
    family: ModelFamily
    dtype: torch.dtype
    attention_status: Mapping[AttentionRole, AttentionStatus]

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.dtype if component == "diffusion" else None


class FluxDiffusionRuntime(FluxRuntime):
    """Classic Flux sampling over separately encoded conditioning."""

    def __init__(
        self,
        diffusion: Flux,
        family: ModelFamily,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        attention_status: AttentionStatus,
    ) -> None:
        self.assembled = _FluxDiffusionAssembly(
            diffusion,
            family,
            compute_dtype,
            {"flux": attention_status},
        )
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._receipt_identity = None
        self._space = _flux_sigma_space(family)
        self._samplers = torch_sampler_registry()
        self._schedulers = torch_scheduler_registry()
        self._guidance = None

    @property
    def retained_offload_storage_components(self) -> frozenset[str]:
        return frozenset()

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        del text
        del hidden_layer
        del min_padding
        del min_length
        raise WiringError("classic Flux diffusion component carries no text encoder")

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        raise WiringError("classic Flux diffusion component carries no codec")

    def encode_content(self, content: torch.Tensor) -> torch.Tensor:
        del content
        raise WiringError("classic Flux diffusion component carries no codec")


def flux_component_runtime(
    loaded: FluxLoadedComponent,
    identity: str,
    dtype: torch.dtype,
) -> FluxDiffusionRuntime:
    return FluxDiffusionRuntime(
        loaded.module,
        _family(loaded.plan.config),
        runtime_identity=identity,
        compute_dtype=dtype,
        attention_status=loaded.attention_status,
    )


__all__ = [
    "FluxDiffusionRuntime",
    "FluxLoadedComponent",
    "flux_component_runtime",
    "load_flux_component",
]
