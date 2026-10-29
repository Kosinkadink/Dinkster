"""Strict loading for independently supplied MiniMax Music 3 components."""

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
    MEBIBYTE,
    SPECIAL_TOKEN_IDS,
    AttentionPolicy,
    AttentionRouteToken,
    MiniMaxMusic3ComponentAssemblyError,
    MiniMaxMusic3ComponentRole,
    minimax_music3_component_runtime_identity,
    plan_minimax_music3_split_component,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32, DType
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
from dinkster_inference.weights import WeightEntry
from tokenizers import Tokenizer

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionRole, resolve_role_attention
from .minimax_music3_dav import MiniMaxMusic3Dav
from .minimax_music3_model import MiniMaxMusic3DiT
from .minimax_music3_text import MiniMaxMusic3TextModel
from .sources import load_tensors_from_file

_IDENTITY_DTYPES: Mapping[torch.dtype, DType] = {
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float32: FLOAT32,
}


@dataclass(frozen=True)
class MiniMaxMusic3LoadedComponent:
    role: MiniMaxMusic3ComponentRole
    module: torch.nn.Module
    plan: ComponentPlan[object]
    runtime_identity: str
    tokenizer: Tokenizer | None = None


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


def _load_tokenizer(
    handle: BinaryIO,
    source: SafetensorsSource,
) -> Tokenizer:
    encoded = load_tensors_from_file(handle, source, ("tokenizer_json",))["tokenizer_json"]
    if encoded.dtype != torch.uint8 or encoded.ndim != 1 or encoded.numel() > 16 * MEBIBYTE:
        raise MiniMaxMusic3ComponentAssemblyError(
            "MiniMax Music 3 tokenizer_json must be bounded rank-1 uint8 data"
        )
    tokenizer = Tokenizer.from_str(encoded.numpy().tobytes().decode("utf-8"))
    for token, expected in SPECIAL_TOKEN_IDS.items():
        if tokenizer.token_to_id(token) != expected:
            raise MiniMaxMusic3ComponentAssemblyError(
                f"MiniMax Music 3 tokenizer mismatch for {token}"
            )
    return tokenizer


def load_minimax_music3_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: MiniMaxMusic3ComponentRole,
    expected_identity: str,
    compute_dtype: torch.dtype,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backend: AttentionRole | None = None,
) -> MiniMaxMusic3LoadedComponent:
    if type(asset) is not AssetRef:
        raise TypeError("MiniMax Music 3 component asset must be an AssetRef")
    if type(expected_identity) is not str or not expected_identity:
        raise ValueError("MiniMax Music 3 component requires an expected identity")
    identity_dtype = _IDENTITY_DTYPES.get(compute_dtype)
    if identity_dtype is None:
        raise TypeError("MiniMax Music 3 compute dtype must be bfloat16, float16, or float32")
    if expected_role == "text" and compute_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("MiniMax Music 3 text compute must use bfloat16 or float32")
    if expected_role == "vae" and compute_dtype is not torch.float32:
        raise TypeError("MiniMax Music 3 DAV compute must use float32")
    try:
        handle = asset.open()
    except (AssetError, OSError) as error:
        raise MiniMaxMusic3ComponentAssemblyError(
            f"MiniMax Music 3 {expected_role} artifact is unavailable: {error}"
        ) from error
    with handle:
        try:
            size = os.fstat(handle.fileno()).st_size
        except OSError as error:
            raise MiniMaxMusic3ComponentAssemblyError(
                f"MiniMax Music 3 {expected_role} artifact is unavailable: {error}"
            ) from error
        if size != asset.size:
            raise MiniMaxMusic3ComponentAssemblyError(
                f"MiniMax Music 3 {expected_role} byte size differs from its asset metadata"
            )
        source = load_safetensors_header_from_file(handle, path=path)
        pinned = _PinnedSource(source, handle, asset.digest, asset.size)
        planned = plan_minimax_music3_split_component(pinned, role=expected_role, path=path)
        runtime_identity = minimax_music3_component_runtime_identity(
            planned,
            expected_role,
            identity_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
        if runtime_identity != expected_identity:
            raise MiniMaxMusic3ComponentAssemblyError(
                f"expected component identity {expected_identity!r}, "
                f"constructed {runtime_identity!r}"
            )
        tokenizer = _load_tokenizer(handle, source) if expected_role == "text" else None
        if expected_role == "diffusion":
            if attention_backend is None:
                raise ValueError("MiniMax Music 3 diffusion requires an attention backend")
            attention = resolve_role_attention(
                attention_backend, attention_policy, attention_route_token
            ).kernel
            builder = partial(MiniMaxMusic3DiT, attention_kernel=attention)
        elif expected_role == "text":
            if attention_backend is None:
                raise ValueError("MiniMax Music 3 text requires an attention backend")
            attention = resolve_role_attention(
                attention_backend, attention_policy, attention_route_token
            ).kernel
            builder = partial(MiniMaxMusic3TextModel, attention_kernel=attention)
        else:
            builder = MiniMaxMusic3Dav
        module = _load_component(
            planned,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source_file=handle,
            source=source,
        )
    return MiniMaxMusic3LoadedComponent(expected_role, module, planned, runtime_identity, tokenizer)


__all__ = [
    "MiniMaxMusic3LoadedComponent",
    "load_minimax_music3_component",
]
