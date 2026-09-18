"""Strict direct-import loading for LTX-2 components."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO, cast

import torch
from dinkster_assets import AssetError, AssetRef
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    LTX_TEXT_STACK_FEATURES,
    LTXAV_19B_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V25_VAE_CONFIG,
    AttentionPolicy,
    AttentionRouteToken,
    ComponentPlan,
    LTXAVAudioCodecAssemblyError,
    LTXAVAudioCodecPlan,
    LTXAVComponentAssemblyError,
    LTXAVStandaloneComponentRole,
    LTXVocoderBWEConfig,
    LTXVocoderConfig,
    ltxav_audio_codec_runtime_identity,
    ltxav_component_runtime_identity,
    ltxav_component_uses_fp8_matmul,
    plan_ltxav_split_audio_codec,
    plan_ltxav_split_component,
)
from dinkster_inference.devices import DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionRole, resolve_role_attention
from .gemma_text import GemmaTextModel, LtxDualTextProjection
from .ltx_audio_vae import LTXAudioVAE, LTXVocoder, LTXVocoderWithBWE
from .ltx_connector import LtxTextConnectors
from .ltx_diffusion_vae import LTXDiffusionVideoVAE
from .ltx_duration import LTXDurationHead
from .ltx_upsampler import LTXLatentUpsampler
from .ltx_video_vae import LTXVideoVAE
from .ltxav_model import LTXAVModel
from .operations import Operations
from .quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from .sources import load_tensors_from_file

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}
_ATTENTION_ROLES: Mapping[LTXAVStandaloneComponentRole, AttentionRole] = {
    "diffusion": "flux",
    "gemma3_12b": "qwen",
    "gemma4_12b": "qwen",
    "connectors": "flux",
}


class LTXAVAudioCodec(torch.nn.Module):
    """One residency unit containing the paired audio VAE and vocoder."""

    def __init__(
        self,
        audio_vae: LTXAudioVAE,
        vocoder: LTXVocoder | LTXVocoderWithBWE,
    ) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.vocoder = vocoder


class LTXAVGemmaComponent(torch.nn.Module):
    """One Gemma residency unit with its immutable tokenizer payload."""

    def __init__(self, model: GemmaTextModel, tokenizer_model: bytes) -> None:
        super().__init__()
        if not tokenizer_model:
            raise ValueError("LTX-2 Gemma component requires tokenizer model bytes")
        self.model = model
        self.tokenizer_model = tokenizer_model


@dataclass(frozen=True)
class LTXAVLoadedAudioCodec:
    """One independently verified and strict-loaded LTX-2 audio codec."""

    module: LTXAVAudioCodec
    plan: LTXAVAudioCodecPlan
    runtime_identity: str


@dataclass(frozen=True)
class LTXAVLoadedComponent:
    """One independently verified and strict-loaded LTX-2 component."""

    role: LTXAVStandaloneComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
    runtime_identity: str
    tokenizer_model: bytes | None = None


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


def load_ltxav_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: LTXAVStandaloneComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> LTXAVLoadedComponent:
    """Verify, plan, identity-check, and strict-load one component."""
    if type(asset) is not AssetRef:
        raise TypeError("LTX-2 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("LTX-2 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("LTX-2 component compute dtype must be bfloat16, float16, or float32")
    if expected_role == "duration_head" and compute_dtype is not torch.float32:
        raise TypeError("LTX-2 duration head requires float32 compute")
    attention_role = _ATTENTION_ROLES.get(expected_role)
    attention_kernel = (
        None
        if attention_role is None
        else resolve_role_attention(
            attention_role,
            attention_policy,
            attention_route_token,
        ).kernel
    )
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise LTXAVComponentAssemblyError(
            f"LTX-2 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise LTXAVComponentAssemblyError(
                f"LTX-2 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise LTXAVComponentAssemblyError(
                f"LTX-2 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        planned = plan_ltxav_split_component(
            _PinnedSource(source, handle, asset.digest, asset.size),
            role=expected_role,
            path=path,
        )
        runtime_identity = ltxav_component_runtime_identity(
            planned,
            identity_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
        if runtime_identity != expected_identity:
            raise LTXAVComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        component = planned.component
        tokenizer_model = None
        builder: Callable[..., torch.nn.Module]
        if expected_role == "diffusion":
            assert attention_kernel is not None
            builder = partial(LTXAVModel, attention_kernel=attention_kernel)
        elif expected_role in ("gemma3_12b", "gemma4_12b"):
            assert attention_kernel is not None
            builder = partial(GemmaTextModel, attention_kernel=attention_kernel)
            tokenizer = load_tensors_from_file(
                handle,
                source,
                (planned.tokenizer_source_key,),
            )[planned.tokenizer_source_key]
            if tokenizer.dtype != torch.uint8 or tokenizer.ndim != 1 or tokenizer.numel() == 0:
                raise LTXAVComponentAssemblyError(
                    f"{expected_role}: {planned.tokenizer_source_key} payload must be "
                    "nonempty rank-1 uint8"
                )
            tokenizer_model = tokenizer.contiguous().numpy().tobytes()
        elif expected_role == "text_projection":
            if component.config == "single_linear":

                def build_projection(_config: object, *, operations: Operations) -> torch.nn.Module:
                    return operations.linear(
                        LTX_TEXT_STACK_FEATURES,
                        LTXAV_19B_CONFIG.caption_channels,
                        bias=False,
                    )
            else:

                def build_projection(_config: object, *, operations: Operations) -> torch.nn.Module:
                    return LtxDualTextProjection(
                        LTX_TEXT_STACK_FEATURES,
                        LTXAV_22B_V23_CONFIG.cross_attention_dim,
                        LTXAV_22B_V23_CONFIG.audio_cross_attention_dim,
                        operations=operations,
                    )

            builder = build_projection
        elif expected_role == "connectors":
            assert attention_kernel is not None
            builder = partial(LtxTextConnectors, attention_kernel=attention_kernel)
        elif expected_role == "duration_head":
            builder = LTXDurationHead
        elif expected_role == "latent_upscaler":
            builder = LTXLatentUpsampler
        elif component.config is LTXAV_22B_V25_VAE_CONFIG:
            builder = LTXDiffusionVideoVAE
        else:
            builder = LTXVideoVAE
        module = _load_component(
            component,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=ltxav_component_uses_fp8_matmul(expected_role, component),
            source_file=handle,
            source=source,
            transform=(
                (lambda _module, state: {key: value.float() for key, value in state.items()})
                if expected_role == "duration_head"
                else None
            ),
        )
        if expected_role in ("gemma3_12b", "gemma4_12b"):
            for layer in module.modules():
                if isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear):
                    layer.compute_dtype = torch.float32
                    layer.full_precision_matmul = True
        if expected_role in ("gemma3_12b", "gemma4_12b"):
            assert tokenizer_model is not None
            module = LTXAVGemmaComponent(cast("GemmaTextModel", module), tokenizer_model)
    return LTXAVLoadedComponent(
        expected_role,
        module,
        cast("ComponentPlan[object]", component),
        runtime_identity,
        tokenizer_model,
    )


def load_ltxav_audio_codec(
    path: Path,
    *,
    asset: AssetRef,
    expected_identity: str,
) -> LTXAVLoadedAudioCodec:
    """Verify, plan, identity-check, and strict-load one audio codec."""
    if type(asset) is not AssetRef:
        raise TypeError("LTX-2 audio codec asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("LTX-2 audio codec requires an expected identity")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise LTXAVAudioCodecAssemblyError(
            f"LTX-2 audio codec artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise LTXAVAudioCodecAssemblyError(
                f"LTX-2 audio codec artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise LTXAVAudioCodecAssemblyError(
                "LTX-2 audio codec byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        planned = plan_ltxav_split_audio_codec(
            _PinnedSource(source, handle, asset.digest, asset.size),
            path=path,
        )
        runtime_identity = ltxav_audio_codec_runtime_identity(planned)
        if runtime_identity != expected_identity:
            raise LTXAVAudioCodecAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        audio_vae = _load_component(
            planned.audio_vae,
            LTXAudioVAE,
            compute_dtype=torch.float32,
            fp8_matmul=False,
            source_file=handle,
            source=source,
            storage_dtype_follows_compute=True,
        )
        if type(planned.vocoder.config) is LTXVocoderBWEConfig:
            vocoder = _load_component(
                cast("ComponentPlan[LTXVocoderBWEConfig]", planned.vocoder),
                LTXVocoderWithBWE,
                compute_dtype=torch.float32,
                fp8_matmul=False,
                source_file=handle,
                source=source,
                storage_dtype_follows_compute=True,
            )
        else:
            vocoder = _load_component(
                cast("ComponentPlan[LTXVocoderConfig]", planned.vocoder),
                LTXVocoder,
                compute_dtype=torch.float32,
                fp8_matmul=False,
                source_file=handle,
                source=source,
                storage_dtype_follows_compute=True,
            )
    return LTXAVLoadedAudioCodec(
        LTXAVAudioCodec(audio_vae, vocoder),
        planned,
        runtime_identity,
    )


__all__ = [
    "LTXAVAudioCodec",
    "LTXAVGemmaComponent",
    "LTXAVLoadedAudioCodec",
    "LTXAVLoadedComponent",
    "load_ltxav_audio_codec",
    "load_ltxav_component",
]
