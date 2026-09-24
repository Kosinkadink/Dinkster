"""Strict direct-import loading for independently supplied Wan components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    WAN21_CAUSAL_AR_1_3B,
    WAN21_HUMO_17B,
    WAN22_S2V_14B,
    Conditioning,
    PromptTokenizer,
    Wan21ComponentAssemblyError,
    Wan21StandaloneComponentPlan,
    Wan21StandaloneComponentRole,
    Wan21VAEConfig,
    plan_wan21_split_component,
    plan_wan22_split_component,
    wan21_component_runtime_identity,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.wan21 import WAN22_WANDANCER_14B
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .operations import Operations
from .sources import load_tensors_from_file
from .t5_text import T5TextEncoder, T5TextModel
from .umt5_tokenizer import Umt5SentencePieceTokenizer
from .wan21_causal import Wan21CausalModel
from .wan21_humo import Wan21HumoModel
from .wan21_model import Wan21Model
from .wan21_vae import WanVAE
from .wan21_vae import WanVAEConfig as TorchWanVAEConfig
from .wan22_dancer import Wan22DancerModel
from .wan22_s2v import Wan22S2VModel

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}

_WAN21_TOKENIZER_ATTRIBUTE = "_dinkster_wan21_spiece_model"


@dataclass(frozen=True)
class Wan21LoadedComponent:
    """One independently verified and strict-loaded Wan component."""

    role: Wan21StandaloneComponentRole
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


def _build_vae(config: Wan21VAEConfig, *, operations: Operations) -> WanVAE:
    return WanVAE(
        TorchWanVAEConfig(
            dim=config.dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temporal_downsample=config.temporal_downsample,
            image_channels=config.image_channels,
            conv_out_channels=config.conv_out_channels,
            dropout=config.dropout,
        ),
        operations=operations,
    )


def load_wan21_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: Wan21StandaloneComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> Wan21LoadedComponent:
    """Verify, plan, identity-check, and strict-load one Wan component."""

    if type(asset) is not AssetRef:
        raise TypeError("Wan component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("Wan component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("Wan component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise Wan21ComponentAssemblyError(
            f"Wan {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise Wan21ComponentAssemblyError(
                f"Wan {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise Wan21ComponentAssemblyError(
                f"Wan {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planner = (
            plan_wan22_split_component
            if expected_identity.startswith("native:dinkster.wan22:")
            else plan_wan21_split_component
        )
        planned = cast(
            "Wan21StandaloneComponentPlan",
            planner(pinned, role=expected_role, path=path),
        )
        runtime_identity = wan21_component_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise Wan21ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        diffusion_config = getattr(planned.component, "config", None)
        diffusion_builder = (
            Wan21CausalModel
            if diffusion_config is WAN21_CAUSAL_AR_1_3B
            else Wan21HumoModel
            if diffusion_config is WAN21_HUMO_17B
            else Wan22S2VModel
            if diffusion_config is WAN22_S2V_14B
            else Wan22DancerModel
            if diffusion_config is WAN22_WANDANCER_14B
            else Wan21Model
        )
        builders = {
            "diffusion": diffusion_builder,
            "umt5xxl": T5TextModel,
            "vae": _build_vae,
        }
        module = _load_component(
            planned.component,
            builders[expected_role],
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
        if expected_role == "umt5xxl":
            tokenizer = load_tensors_from_file(handle, source, (planned.tokenizer_source_key,))[
                planned.tokenizer_source_key
            ]
            module.__dict__[_WAN21_TOKENIZER_ATTRIBUTE] = tokenizer.contiguous().numpy().tobytes()
    return Wan21LoadedComponent(
        expected_role,
        module,
        cast("ComponentPlan[object]", planned.component),
        runtime_identity,
    )


class Wan21TextRuntime:
    """Text-only Wan encoder over an independently resident UMT5 component."""

    def __init__(self, text: T5TextModel) -> None:
        tokenizer_model = text.__dict__.get(_WAN21_TOKENIZER_ATTRIBUTE)
        if type(tokenizer_model) is not bytes or not tokenizer_model:
            raise Wan21ComponentAssemblyError("Wan UMT5 component carries no tokenizer model")
        tokenizer = Umt5SentencePieceTokenizer(tokenizer_model)
        self.text = text
        self._encoder = T5TextEncoder(text)
        self._tokenizer = PromptTokenizer(encode_word=tokenizer.encode)

    def encode_text(
        self,
        text: str,
        *,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        spans = self._tokenizer.tokenize(text)
        if min_padding is None and min_length is None:
            return self._encoder.encode(spans)
        return self._encoder.encode(spans, min_padding=min_padding, min_length=min_length)


__all__ = [
    "Wan21LoadedComponent",
    "Wan21TextRuntime",
    "load_wan21_component",
]
