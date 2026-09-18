"""Strict direct-import loading for classic LTX-Video components."""

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
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    T5_XXL_LTXV_PROFILE,
    Conditioning,
    ConditioningCarrier,
    LTXVComponentAssemblyError,
    LTXVStandaloneComponentRole,
    PromptTokenizer,
    load_t5_spm,
    ltxv_component_runtime_identity,
    plan_ltxv_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .ltx_model import LTXVModel
from .ltx_video_vae import LTXVideoVAE
from .t5_text import T5TextEncoder, T5TextModel

if TYPE_CHECKING:
    from .attention import AttentionKernel, AttentionRole
    from .checkpoint_runtime import ComponentAssembly

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class LTXVLoadedComponent:
    """One independently verified and strict-loaded LTX-Video component."""

    role: LTXVStandaloneComponentRole
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


def realize_ltxv_component(
    plan: ComponentPlan[object],
    *,
    compute_dtype: torch.dtype,
    source_file: BinaryIO | None = None,
    source: SafetensorsSource | None = None,
    fp8_matmul: bool = False,
    attention_kernels: Mapping[AttentionRole, AttentionKernel] | None = None,
) -> torch.nn.Module:
    builder = {
        "diffusion": (
            LTXVModel
            if attention_kernels is None
            else partial(LTXVModel, attention_kernel=attention_kernels["flux"])
        ),
        "t5xxl": (
            T5TextModel
            if attention_kernels is None
            else partial(T5TextModel, attention_kernel=attention_kernels["t5"])
        ),
        "vae": LTXVideoVAE,
    }[plan.component]
    return _load_component(
        plan,
        builder,
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
        source_file=source_file,
        source=source,
    )


def load_ltxv_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: LTXVStandaloneComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
) -> LTXVLoadedComponent:
    """Verify, plan, identity-check, and strict-load one split component."""
    if type(asset) is not AssetRef:
        raise TypeError("LTX-Video component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("LTX-Video component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("LTX-Video component compute dtype must be bfloat16, float16, or float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise LTXVComponentAssemblyError(
            f"LTX-Video {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise LTXVComponentAssemblyError(
                f"LTX-Video {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise LTXVComponentAssemblyError(
                f"LTX-Video {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        planned = plan_ltxv_split_component(
            _PinnedSource(source, handle, asset.digest, asset.size),
            role=expected_role,
            path=path,
        )
        runtime_identity = ltxv_component_runtime_identity(planned, identity_dtype)
        if runtime_identity != expected_identity:
            raise LTXVComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        module = realize_ltxv_component(
            cast("ComponentPlan[object]", planned.component),
            compute_dtype=compute_dtype,
            source_file=handle,
            source=source,
        )
    return LTXVLoadedComponent(
        expected_role,
        module,
        cast("ComponentPlan[object]", planned.component),
        runtime_identity,
    )


class LTXVTextRuntime:
    """Text-only classic LTX-Video encoder over an independent T5 component."""

    def __init__(self, text: T5TextModel) -> None:
        self.text = text
        self._encoder = T5TextEncoder(
            text,
            profile=T5_XXL_LTXV_PROFILE,
            attention_masked=True,
            zero_out_masked=False,
        )
        self._tokenizer = PromptTokenizer(encode_word=load_t5_spm().encode)

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

    @staticmethod
    def text_conditioning_carrier(value: Conditioning[torch.Tensor]) -> ConditioningCarrier:
        from .ltxv_runtime import ltxv_text_conditioning_to_carrier

        return ltxv_text_conditioning_to_carrier(value)


def checkpoint_text_runtime(assembled: ComponentAssembly) -> LTXVTextRuntime | None:
    text = assembled.components.get("t5xxl")
    return None if text is None else LTXVTextRuntime(cast(T5TextModel, text))


__all__ = [
    "LTXVLoadedComponent",
    "LTXVTextRuntime",
    "load_ltxv_component",
    "realize_ltxv_component",
]
