"""Strict loading for independently supplied Chroma components."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    ChromaComponentAssemblyError,
    ChromaComponentRole,
    chroma_component_runtime_identity,
    plan_chroma_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.chroma import ChromaConfig, ChromaRadianceConfig
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionKernel, AttentionRole, AttentionStatus, resolve_role_attention
from .autoencoder_kl import AutoencoderKL, kl_codec_plugin
from .checkpoint_runtime import ComponentAssembly
from .chroma import Chroma, ChromaRadiance
from .chroma_runtime import ChromaTextRuntime
from .codecs import CodecPlugin
from .module_residency import declare_residency_materialization_ceilings
from .operations import Operations, ResidencyRouted
from .quant_linear import Fp8Linear, supports_fp8_matmul
from .t5_text import T5TextModel
from .z_image import PixelSpaceCodec

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class ChromaLoadedComponent:
    """One independently verified and strict-loaded Chroma component."""

    role: ChromaComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
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


def _build_diffusion(
    config: ChromaConfig | ChromaRadianceConfig,
    *,
    operations: Operations,
    attention_kernel: AttentionKernel,
) -> Chroma | ChromaRadiance:
    if isinstance(config, ChromaRadianceConfig):
        return ChromaRadiance(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
    return Chroma(config, operations=operations, attention_kernel=attention_kernel)


def realize_chroma_component(
    plan: ComponentPlan[object],
    *,
    compute_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    fp8_matmul: bool = False,
    attention_kernels: Mapping[AttentionRole, AttentionKernel],
) -> torch.nn.Module:
    builders: Mapping[str, tuple[Callable[..., torch.nn.Module], AttentionRole]] = {
        "diffusion": (_build_diffusion, "flux"),
        "t5xxl": (T5TextModel, "t5"),
        "vae": (AutoencoderKL, "vae"),
    }
    builder, attention_role = builders[plan.component]
    module = _load_component(
        plan,
        partial(builder, attention_kernel=attention_kernels[attention_role]),
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
        source_file=source_file,
        source=source,
    )
    if plan.component == "t5xxl":
        module.requires_grad_(False)
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
    return module


def checkpoint_text_runtime(assembled: ComponentAssembly) -> ChromaTextRuntime | None:
    text = assembled.components.get("t5xxl")
    return None if text is None else ChromaTextRuntime(cast(T5TextModel, text))


def checkpoint_codec(assembled: ComponentAssembly) -> CodecPlugin | PixelSpaceCodec | None:
    if isinstance(assembled.diffusion, ChromaRadiance):
        dtype = assembled.compute_dtype("diffusion")
        assert dtype is not None
        return PixelSpaceCodec(compute_dtype=dtype)
    vae = assembled.components.get("vae")
    if vae is None:
        return None
    return replace(
        kl_codec_plugin(cast(AutoencoderKL, vae)), compute_dtype=assembled.compute_dtype("vae")
    )


def load_chroma_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: ChromaComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
    load_device: torch.device | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backend: AttentionRole,
) -> ChromaLoadedComponent:
    """Verify, plan, identity-check, and strict-load one Chroma component."""

    if type(asset) is not AssetRef:
        raise TypeError("Chroma component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Chroma component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Chroma component compute dtype must be bfloat16, float16, or float32")
    attention = resolve_role_attention(
        attention_backend,
        attention_policy,
        attention_route_token,
    )
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise ChromaComponentAssemblyError(
            f"Chroma {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise ChromaComponentAssemblyError(
                f"Chroma {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise ChromaComponentAssemblyError(
                f"Chroma {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_chroma_split_component(pinned, role=expected_role, path=path)
        runtime_identity = chroma_component_runtime_identity(
            planned,
            expected_role,
            identity_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
        if runtime_identity != expected_identity:
            raise ChromaComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = realize_chroma_component(
            planned,
            compute_dtype=compute_dtype,
            source_file=handle,
            source=source,
            attention_kernels={attention_backend: attention.kernel},
        )
        if (
            expected_role == "diffusion"
            and load_device is not None
            and any(quant.config is not None for quant in planned.quant.values())
            and supports_fp8_matmul(load_device)
        ):
            for layer, quant in planned.quant.items():
                if quant.config is None:
                    continue
                submodule = module.get_submodule(layer)
                if isinstance(submodule, Fp8Linear) and not submodule.full_precision_matmul:
                    submodule.bind_fp8_matmul(True)
    return ChromaLoadedComponent(
        expected_role,
        module,
        planned,
        runtime_identity,
        attention.status,
    )


__all__ = [
    "ChromaLoadedComponent",
    "load_chroma_component",
]
