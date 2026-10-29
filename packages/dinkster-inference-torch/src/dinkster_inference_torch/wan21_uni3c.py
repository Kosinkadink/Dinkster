"""Native execution for the maintained Wan 2.1 Uni3C patch."""

from __future__ import annotations

import ctypes
import math
import sys
import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import WAN21_UNI3C, PercentRange, Wan21Uni3CConfig

from .attention import AttentionKernel, select_attention
from .flux import EmbedND
from .operations import INITLESS, CastOperations, Operations
from .wan21_model import WanFeedForward, WanSelfAttention

_DEFAULT_ATTENTION = select_attention("flux").kernel


class Wan21Uni3CBindingError(ValueError):
    """A Uni3C resource or prepared render no longer matches its identity."""


class Wan21Uni3CLayerNormZero(torch.nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        *,
        eps: float = 1e-5,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = operations.linear(conditioning_dim, 3 * embedding_dim)
        self.norm = operations.layer_norm(embedding_dim, eps=eps)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        normalized = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return normalized, gate[:, None, :]


class Wan21Uni3CAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        *,
        time_embed_dim: int,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.norm1 = Wan21Uni3CLayerNormZero(time_embed_dim, dim, operations=operations)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            qk_norm=True,
            eps=1e-6,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm2 = Wan21Uni3CLayerNormZero(time_embed_dim, dim, operations=operations)
        self.ffn = WanFeedForward(
            operations.linear(dim, ffn_dim),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(ffn_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        normalized, attention_gate = self.norm1(x, temb)
        x = x + attention_gate * self.self_attn(normalized, freqs)
        normalized, feed_forward_gate = self.norm2(x, temb)
        return x + feed_forward_gate * self.ffn(normalized)


class Wan21Uni3CMaskCameraEmbedding(torch.nn.Module):
    def __init__(
        self,
        add_channels: int,
        mid_channels: int,
        output_dim: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.mask_proj = torch.nn.Sequential(
            operations.conv3d(
                add_channels,
                mid_channels,
                (4, 8, 8),
                stride=(4, 8, 8),
            ),
            operations.group_norm(
                mid_channels,
                num_groups=mid_channels // 8,
                eps=1e-5,
            ),
            torch.nn.SiLU(),
        )
        self.mask_zero_proj = operations.conv3d(
            mid_channels,
            output_dim,
            (1, 2, 2),
            stride=(1, 2, 2),
        )

    def forward(self, additional: torch.Tensor) -> torch.Tensor:
        additional = F.pad(additional, (0, 0, 0, 0, 3, 0))
        return self.mask_zero_proj(self.mask_proj(additional)).flatten(2).transpose(1, 2)


class Wan21Uni3C(torch.nn.Module):
    """The exact published 20-block Wan 2.1 Uni3C control transformer."""

    def __init__(
        self,
        config: Wan21Uni3CConfig = WAN21_UNI3C,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config is not WAN21_UNI3C:
            raise ValueError("Wan 2.1 Uni3C requires the exact maintained profile")
        self.config = config
        self.controlnet_patch_embedding = CastOperations(torch.float32).conv3d(
            config.input_channels,
            config.patch_width,
            (1, 2, 2),
            stride=(1, 2, 2),
        )
        self.controlnet_mask_embedding = Wan21Uni3CMaskCameraEmbedding(
            config.mask_channels,
            config.mask_hidden_width,
            config.patch_width,
            operations=operations,
        )
        self.proj_in = operations.linear(config.patch_width, config.hidden_width)
        self.controlnet_blocks = torch.nn.ModuleList(
            Wan21Uni3CAttentionBlock(
                config.hidden_width,
                config.ffn_width,
                config.attention_heads,
                time_embed_dim=config.timestep_width,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.layers)
        )
        self.proj_out = torch.nn.ModuleList(
            operations.linear(config.hidden_width, config.output_width)
            for _ in range(config.layers)
        )
        head_dim = config.hidden_width // config.attention_heads
        axis_width = head_dim // 6
        self.rope_embedder = EmbedND(
            dim=head_dim,
            theta=10000,
            axes_dim=(head_dim - 4 * axis_width, 2 * axis_width, 2 * axis_width),
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def rope_encode(
        self,
        time: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros((time, height, width, 3), device=device, dtype=dtype)
        ids[..., 0] += torch.arange(time, device=device, dtype=dtype)[:, None, None]
        ids[..., 1] += torch.arange(height, device=device, dtype=dtype)[None, :, None]
        ids[..., 2] += torch.arange(width, device=device, dtype=dtype)[None, None, :]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    def process_input(
        self,
        control_input: torch.Tensor,
        render_mask: torch.Tensor | None = None,
        camera_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.controlnet_patch_embedding(control_input.float()).to(control_input.dtype)
        time, height, width = hidden.shape[2:]
        freqs = self.rope_encode(
            time,
            height,
            width,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = hidden.flatten(2).transpose(1, 2)
        additional = None
        if render_mask is not None:
            additional = (
                render_mask
                if camera_embedding is None
                else torch.cat((render_mask, camera_embedding), dim=1)
            )
        if additional is not None:
            hidden = hidden + self.controlnet_mask_embedding(additional.to(hidden.dtype))
        return self.proj_in(hidden), freqs

    def forward_block(
        self,
        block_index: int,
        hidden: torch.Tensor,
        temb: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.controlnet_blocks[block_index](hidden, temb, freqs)
        return hidden, self.proj_out[block_index](hidden)


@dataclass(frozen=True, slots=True)
class _TensorSeal:
    name: str
    tensor: torch.Tensor
    storage: torch.UntypedStorage
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    conjugated: bool
    negated: bool


@dataclass(frozen=True, slots=True)
class _ResourceSeal:
    digest: str
    tensors: tuple[_TensorSeal, ...]


_RESOURCE_SEALS: weakref.WeakKeyDictionary[Wan21Uni3C, _ResourceSeal] = weakref.WeakKeyDictionary()


def _resource_tensors(model: Wan21Uni3C) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", tensor) for name, tensor in model.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in model.named_buffers()),
    )


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("Wan 2.1 Uni3C tensors must track mutation versions") from error


def wan21_uni3c_resource_digest(asset_digest: str, compute_dtype: torch.dtype) -> str:
    """Identity for one assembled Uni3C artifact."""
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("Wan 2.1 Uni3C source digest must be canonical blake3 identity")
    if not compute_dtype.is_floating_point:
        raise TypeError("Wan 2.1 Uni3C compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.wan21-uni3c-resource.v1\n")
    hasher.update(f"asset={asset_digest}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def _bind_wan21_uni3c_resource(  # pyright: ignore[reportUnusedFunction]
    model: Wan21Uni3C,
    digest: str,
) -> None:
    if type(model) is not Wan21Uni3C:
        raise TypeError("Wan 2.1 Uni3C resource must be the exact maintained model")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Wan 2.1 Uni3C resource digest must be lowercase BLAKE3")
    if model in _RESOURCE_SEALS:
        raise ValueError("Wan 2.1 Uni3C resource is already bound")
    _RESOURCE_SEALS[model] = _ResourceSeal(
        digest,
        tuple(
            _TensorSeal(
                name,
                tensor,
                tensor.untyped_storage(),
                _tensor_version(tensor),
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.is_conj(),
                tensor.is_neg(),
            )
            for name, tensor in _resource_tensors(model)
        ),
    )


def validate_wan21_uni3c_resource(model: Wan21Uni3C, digest: str) -> None:
    """Require a Uni3C model to retain assembly-proven state."""
    if type(model) is not Wan21Uni3C:
        raise TypeError("Wan 2.1 Uni3C resource must be the exact maintained model")
    from .module_residency import (
        _residency_assignment_generation,  # pyright: ignore[reportPrivateUsage]
        _residency_assignment_version,  # pyright: ignore[reportPrivateUsage]
    )

    seal = _RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            name != expected.name
            or tensor.dtype != expected.dtype
            or tuple(tensor.shape) != expected.shape
            or tuple(tensor.stride()) != expected.stride
            or tensor.storage_offset() != expected.storage_offset
            or tensor.is_conj() != expected.conjugated
            or tensor.is_neg() != expected.negated
            or not (
                (
                    tensor is expected.tensor
                    and tensor.untyped_storage() is expected.storage
                    and _tensor_version(tensor) == expected.version
                )
                or (
                    _residency_assignment_generation(model, name.partition(":")[2], tensor)
                    is not None
                    and _residency_assignment_version(model, name.partition(":")[2], tensor)
                    == _tensor_version(tensor)
                )
            )
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise Wan21Uni3CBindingError(
            "wan21-uni3c-resource-binding-mismatch: model provenance is absent or changed"
        )


def wan21_uni3c_tensor_digest(value: torch.Tensor) -> str:
    """Content identity for a canonical Uni3C render tensor."""
    if type(value) is not torch.Tensor or value.layout is not torch.strided:
        raise TypeError("Wan 2.1 Uni3C render must be an exact strided tensor")
    tensor = value.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(
        tensor.data_ptr(), tensor.numel() * tensor.element_size()
    )
    width = tensor.element_size()
    if sys.byteorder == "big" and width > 1:
        raw = bytearray(raw)
        for start in range(0, len(raw), width):
            raw[start : start + width] = reversed(raw[start : start + width])
    hasher = blake3()
    hasher.update(b"dinkster.wan21-uni3c-render.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in tensor.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(tensor.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class Wan21Uni3CExecution:
    """One prepared, identity-bound Uni3C application."""

    model: Wan21Uni3C
    render_latent: torch.Tensor
    strength: float
    window: PercentRange
    model_digest: str
    render_digest: str

    def __post_init__(self) -> None:
        if type(self.model) is not Wan21Uni3C:
            raise TypeError("Wan 2.1 Uni3C model must be the exact maintained patch")
        if (
            type(self.render_latent) is not torch.Tensor
            or self.render_latent.layout is not torch.strided
            or not self.render_latent.is_floating_point()
            or self.render_latent.ndim != 5
            or self.render_latent.shape[1] != 16
            or any(size < 1 for size in self.render_latent.shape)
        ):
            raise ValueError("Wan 2.1 Uni3C render latent must be floating [batch,16,T,H,W]")
        if type(self.strength) is not float or not math.isfinite(self.strength):
            raise TypeError("Wan 2.1 Uni3C strength must be a finite float")
        if not -10.0 <= self.strength <= 10.0:
            raise ValueError("Wan 2.1 Uni3C strength must be within [-10, 10]")
        if type(self.window) is not PercentRange:
            raise TypeError("Wan 2.1 Uni3C window must be an exact PercentRange")
        validate_wan21_uni3c_resource(self.model, self.model_digest)
        if wan21_uni3c_tensor_digest(self.render_latent) != self.render_digest:
            raise Wan21Uni3CBindingError("Wan 2.1 Uni3C render latent identity changed")


def snapshot_wan21_uni3c_execution(execution: Wan21Uni3CExecution) -> Wan21Uni3CExecution:
    """Validate then own the prepared render latent for one sampling execution."""
    if type(execution) is not Wan21Uni3CExecution:
        raise TypeError("Wan 2.1 Uni3C input must be exact Wan21Uni3CExecution")
    return Wan21Uni3CExecution(
        execution.model,
        execution.render_latent.detach().clone(),
        execution.strength,
        execution.window,
        execution.model_digest,
        execution.render_digest,
    )


__all__ = [
    "Wan21Uni3C",
    "Wan21Uni3CBindingError",
    "Wan21Uni3CExecution",
    "snapshot_wan21_uni3c_execution",
    "validate_wan21_uni3c_resource",
    "wan21_uni3c_resource_digest",
    "wan21_uni3c_tensor_digest",
]
