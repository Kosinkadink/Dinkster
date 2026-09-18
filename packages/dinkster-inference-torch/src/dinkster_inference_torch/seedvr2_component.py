"""Strict loading for independently supplied SeedVR2 components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    SeedVR2ComponentAssemblyError,
    SeedVR2ComponentRole,
    SeedVR2Config,
    SeedVR2VAEConfig,
    plan_seedvr2_split_component,
    seedvr2_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .operations import Operations, bound_compute_dtype
from .seedvr2_dit import NaDiT
from .seedvr2_vae import VideoAutoencoderKLWrapper

if TYPE_CHECKING:
    from .attention import AttentionKernel, AttentionRole

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class SeedVR2LoadedComponent:
    role: SeedVR2ComponentRole
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

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.source.read_uint8_configuration_from_file(self.file, key)


def _build_diffusion(
    config: SeedVR2Config,
    *,
    operations: Operations,
    attention_kernel: AttentionKernel | None = None,
) -> NaDiT:
    builder = (
        NaDiT if attention_kernel is None else partial(NaDiT, attention_kernel=attention_kernel)
    )
    return builder(
        norm_eps=config.norm_eps,
        num_layers=config.layers,
        mlp_type=config.mlp_type,
        vid_dim=config.width,
        heads=config.heads,
        mm_layers=config.separate_layers,
        rope_type=config.rope_type,
        rope_dim=config.rope_dim,
        vid_out_norm="rms" if config.vid_out_norm else None,
        image_model="seedvr2",
        operations=operations,
    )


def _build_vae(
    config: SeedVR2VAEConfig,
    *,
    operations: Operations,
    attention_kernel: AttentionKernel | None = None,
) -> VideoAutoencoderKLWrapper:
    if config.temporal_scale != 4 or config.spatial_scale != 8:
        raise SeedVR2ComponentAssemblyError("SeedVR2 VAE plan has unsupported scale factors")
    builder = (
        VideoAutoencoderKLWrapper
        if attention_kernel is None
        else partial(VideoAutoencoderKLWrapper, attention_kernel=attention_kernel)
    )
    return builder(
        spatial_downsample_factor=config.spatial_scale,
        temporal_downsample_factor=config.temporal_scale,
        operations=operations,
    )


def realize_seedvr2_component(
    plan: ComponentPlan[object],
    *,
    compute_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    fp8_matmul: bool = False,
    attention_kernels: Mapping[AttentionRole, AttentionKernel] | None = None,
) -> torch.nn.Module:
    builder = _build_diffusion if plan.component == "diffusion" else _build_vae
    if attention_kernels is not None:
        builder = partial(
            builder,
            attention_kernel=attention_kernels["flux" if plan.component == "diffusion" else "vae"],
        )
    module = _load_component(
        plan,
        builder,
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
        source_file=source_file,
        source=source,
        # Seed diffusion has no stored-weight patch path requiring pre-rounded weights.
        preserve_equal_width_cast_storage=plan.component == "diffusion" and not plan.quant,
    )
    if plan.component == "diffusion":
        diffusion = cast(NaDiT, module)
        for name in ("positive_conditioning", "negative_conditioning"):
            diffusion.register_buffer(name, getattr(diffusion, name).clone(), persistent=False)
        # Detach non-casting source views from mappings scanned by load-time conversion.
        if not plan.quant:
            for owner in diffusion.modules():
                if bound_compute_dtype(owner) == compute_dtype:
                    continue
                for name, parameter in tuple(owner.named_parameters(recurse=False)):
                    if parameter.is_floating_point() and parameter.dtype != compute_dtype:
                        setattr(
                            owner,
                            name,
                            torch.nn.Parameter(
                                parameter.detach().clone(),
                                requires_grad=parameter.requires_grad,
                            ),
                        )
                for name, buffer in tuple(owner.named_buffers(recurse=False)):
                    if buffer.is_floating_point() and buffer.dtype != compute_dtype:
                        setattr(owner, name, buffer.clone())
    return module


def load_seedvr2_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: SeedVR2ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> SeedVR2LoadedComponent:
    if type(asset) is not AssetRef:
        raise TypeError("SeedVR2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("SeedVR2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("SeedVR2 component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise SeedVR2ComponentAssemblyError(
            f"SeedVR2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise SeedVR2ComponentAssemblyError(
                f"SeedVR2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise SeedVR2ComponentAssemblyError(
                f"SeedVR2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_seedvr2_split_component(pinned, role=expected_role, path=path)
        runtime_identity = seedvr2_component_runtime_identity(
            planned, expected_role, identity_dtype
        )
        if runtime_identity != expected_identity:
            raise SeedVR2ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = realize_seedvr2_component(
            planned,
            compute_dtype=compute_dtype,
            source_file=handle,
            source=source,
        )
    return SeedVR2LoadedComponent(expected_role, module, planned, runtime_identity)


__all__ = ["SeedVR2LoadedComponent", "load_seedvr2_component", "realize_seedvr2_component"]
