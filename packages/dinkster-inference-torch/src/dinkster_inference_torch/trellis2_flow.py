"""Native TRELLIS.2 dense and sparse flow transformers.

The module structure matches ComfyUI's TRELLIS.2 state dictionaries.
Sparse values remain in :class:`dinkster_inference.SparseLatent` at the
public boundary and use a transient torch view only while executing a
flow. Both dense structure and sparse shape/texture stages use the
shared optimized attention selection and normal module residency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dinkster_inference import SparseLatent, SparseSupport, Trellis2FlowConfig

from .attention import AttentionKernel, select_attention
from .flux import apply_rope
from .operations import Operations, ResidencyRouted
from .sparse import authenticate_sparse_support, pack_sparse_latent, unpack_sparse_latent

_DEFAULT_ATTENTION = select_attention("flux").kernel


@dataclass(frozen=True)
class _SparseTensor:
    support: SparseSupport[torch.Tensor]
    feats: torch.Tensor

    def replace(self, feats: torch.Tensor) -> _SparseTensor:
        return _SparseTensor(self.support, feats)

    @property
    def batch_map(self) -> torch.Tensor:
        counts = torch.tensor(self.support.batch_counts, device=self.feats.device)
        return torch.repeat_interleave(
            torch.arange(len(self.support.batch_counts), device=self.feats.device), counts
        )


def _padded(value: _SparseTensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    counts = value.support.batch_counts
    maximum = max(counts)
    batch = len(counts)
    if all(count == maximum for count in counts):
        return value.feats.view(batch, maximum, *value.feats.shape[1:]), None
    padded = value.feats.new_zeros((batch, maximum, *value.feats.shape[1:]))
    valid = torch.zeros((batch, maximum), dtype=torch.bool, device=value.feats.device)
    offset = 0
    for index, count in enumerate(counts):
        padded[index, :count] = value.feats[offset : offset + count]
        valid[index, :count] = True
        offset += count
    return padded, valid


def _sparse_attention(
    q: _SparseTensor,
    k: _SparseTensor,
    v: _SparseTensor,
    kernel: AttentionKernel,
) -> _SparseTensor:
    q_padded, q_valid = _padded(q)
    k_padded, k_valid = _padded(k)
    v_padded, _ = _padded(v)
    mask = None
    if k_valid is not None:
        mask = torch.zeros(
            (k_padded.shape[0], 1, 1, k_padded.shape[1]),
            dtype=q_padded.dtype,
            device=q_padded.device,
        )
        mask.masked_fill_(~k_valid[:, None, None, :], -torch.finfo(mask.dtype).max)
    out = kernel(
        q_padded.transpose(1, 2),
        k_padded.transpose(1, 2),
        v_padded.transpose(1, 2),
        mask=mask,
    ).transpose(1, 2)
    return q.replace(out.reshape(-1, *out.shape[2:]) if q_valid is None else out[q_valid])


def _timestep_embedding(timestep: torch.Tensor, channels: int) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
    )
    arguments = timestep[:, None].float() * frequencies[None]
    return torch.cat((arguments.cos(), arguments.sin()), dim=-1)


class Trellis2TimestepEmbedder(torch.nn.Module):
    def __init__(self, config: Trellis2FlowConfig, *, operations: Operations) -> None:
        super().__init__()
        self.channels = config.timestep_channels
        self.mlp = torch.nn.Sequential(
            operations.linear(config.timestep_channels, config.model_channels),
            torch.nn.SiLU(),
            operations.linear(config.model_channels, config.model_channels),
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(_timestep_embedding(timestep, self.channels).to(dtype))


class Trellis2HeadRmsNorm(ResidencyRouted, torch.nn.Module):
    def __init__(self, heads: int, head_channels: int, *, microsoft_split_precision: bool) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.empty(heads, head_channels))
        self.scale = head_channels**0.5
        self.microsoft_split_precision = microsoft_split_precision

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._normalize(value, self.gamma)
        with binding.lease() as lease:
            return self._normalize(value, lease.get("gamma", dtype=self.gamma.dtype))

    def _normalize(self, value: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        if self.microsoft_split_precision:
            return (F.normalize(value.float(), dim=-1) * gamma * self.scale).to(value.dtype)
        return F.rms_norm(value, (value.shape[-1],)) * gamma.to(value)


def _rope_table(coordinates: torch.Tensor, head_channels: int) -> torch.Tensor:
    frequency_channels = head_channels // 2 // 3
    frequencies = (
        torch.arange(frequency_channels, dtype=torch.float32, device=coordinates.device)
        / frequency_channels
    )
    frequencies = 1.0 / (10000.0**frequencies)
    phases = torch.cat(
        [torch.outer(coordinates[:, axis].float(), frequencies) for axis in range(3)], dim=-1
    )
    missing = head_channels // 2 - phases.shape[-1]
    if missing:
        phases = F.pad(phases, (0, missing))
    cosine, sine = phases.cos(), phases.sin()
    return torch.stack(
        (torch.stack((cosine, sine), dim=-1), torch.stack((-sine, cosine), dim=-1)),
        dim=-1,
    )


def _microsoft_rope_table(
    coordinates: torch.Tensor, head_channels: int, *, phases_on_cpu: bool = False
) -> torch.Tensor:
    frequency_channels = head_channels // 2 // 3
    frequencies = torch.arange(frequency_channels, dtype=torch.float32) / frequency_channels
    frequencies = 1.0 / (10000.0**frequencies)
    output_device = coordinates.device
    if phases_on_cpu:
        coordinates = coordinates.cpu()
    else:
        frequencies = frequencies.to(coordinates.device)
    phases = torch.cat(
        [torch.outer(coordinates[:, axis].float(), frequencies) for axis in range(3)], dim=-1
    )
    missing = head_channels // 2 - phases.shape[-1]
    if missing:
        phases = F.pad(phases, (0, missing))
    return torch.polar(torch.ones_like(phases), phases).to(output_device)


def _apply_microsoft_rope(value: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    complex_value = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
    rotated = complex_value * phases.unsqueeze(-2)
    return torch.view_as_real(rotated).reshape(*rotated.shape[:-1], -1).to(value.dtype)


def _apply_sparse_rope(
    q: torch.Tensor, k: torch.Tensor, phases: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rotated, k_rotated = apply_rope(q.unsqueeze(0), k.unsqueeze(0), phases[None, :, None])
    return q_rotated.squeeze(0), k_rotated.squeeze(0)


class Trellis2Attention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: Trellis2FlowConfig,
        *,
        cross: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
        microsoft_split_precision: bool,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.heads = config.num_heads
        self.head_channels = config.head_channels
        self.cross = cross
        self.microsoft_split_precision = microsoft_split_precision
        if cross:
            self.to_q = operations.linear(hidden, hidden)
            self.to_kv = operations.linear(config.condition_channels, hidden * 2)
        else:
            self.to_qkv = operations.linear(hidden, hidden * 3)
        self.q_rms_norm = Trellis2HeadRmsNorm(
            self.heads,
            self.head_channels,
            microsoft_split_precision=microsoft_split_precision,
        )
        self.k_rms_norm = Trellis2HeadRmsNorm(
            self.heads,
            self.head_channels,
            microsoft_split_precision=microsoft_split_precision,
        )
        self.to_out = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def _dense(
        self,
        value: torch.Tensor,
        context: torch.Tensor | None,
        rope: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, rows, _ = value.shape
        if self.cross:
            assert context is not None
            q = self.to_q(value).view(batch, rows, self.heads, self.head_channels)
            key_value = self.to_kv(context).view(
                batch, context.shape[1], 2, self.heads, self.head_channels
            )
            k, v = key_value.unbind(dim=2)
        else:
            query_key_value = self.to_qkv(value).view(
                batch, rows, 3, self.heads, self.head_channels
            )
            q, k, v = query_key_value.unbind(dim=2)
        q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        if rope is not None:
            if self.microsoft_split_precision:
                q = _apply_microsoft_rope(q, rope)
                k = _apply_microsoft_rope(k, rope)
            else:
                q, k = apply_rope(q, k, rope)
        attended = self._attention_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.to_out(attended.transpose(1, 2).reshape(batch, rows, -1))

    def _sparse(
        self,
        value: _SparseTensor,
        context: torch.Tensor | None,
    ) -> _SparseTensor:
        rows = value.feats.shape[0]
        if self.cross:
            assert context is not None
            q = value.replace(self.to_q(value.feats).view(rows, self.heads, self.head_channels))
            key_value = self.to_kv(context).view(
                context.shape[0], context.shape[1], 2, self.heads, self.head_channels
            )
            k_dense, v_dense = key_value.unbind(dim=2)
            q_padded, valid = _padded(q)
            q_dense = self.q_rms_norm(q_padded).transpose(1, 2)
            k_dense = self.k_rms_norm(k_dense).transpose(1, 2)
            attended = self._attention_kernel(q_dense, k_dense, v_dense.transpose(1, 2))
            out = attended.transpose(1, 2)
            out = out.reshape(-1, *out.shape[2:]) if valid is None else out[valid]
            return value.replace(self.to_out(out.reshape(rows, -1)))
        query_key_value = self.to_qkv(value.feats).view(rows, 3, self.heads, self.head_channels)
        q_values, k_values, v_values = query_key_value.unbind(dim=1)
        q_values = self.q_rms_norm(q_values)
        k_values = self.k_rms_norm(k_values)
        coordinates = value.support.coordinates[:, 1:].to(value.feats.device)
        if self.microsoft_split_precision:
            rope = _microsoft_rope_table(coordinates, self.head_channels)
            q_values = _apply_microsoft_rope(q_values, rope)
            k_values = _apply_microsoft_rope(k_values, rope)
        else:
            rope = _rope_table(coordinates, self.head_channels)
            q_values, k_values = _apply_sparse_rope(q_values, k_values, rope)
        attended = _sparse_attention(
            value.replace(q_values),
            value.replace(k_values),
            value.replace(v_values),
            self._attention_kernel,
        )
        return value.replace(self.to_out(attended.feats.reshape(rows, -1)))

    def forward(
        self,
        value: torch.Tensor | _SparseTensor,
        context: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor | _SparseTensor:
        if isinstance(value, _SparseTensor):
            return self._sparse(value, context)
        return self._dense(value, context, rope)


class Trellis2ProjectedAttention(torch.nn.Module):
    def __init__(
        self,
        config: Trellis2FlowConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
        microsoft_split_precision: bool,
    ) -> None:
        super().__init__()
        assert config.projected_channels is not None
        self.cross_attn_block = Trellis2Attention(
            config,
            cross=True,
            operations=operations,
            attention_kernel=attention_kernel,
            microsoft_split_precision=microsoft_split_precision,
        )
        self.proj_linear = operations.linear(config.projected_channels, config.model_channels)

    def forward(
        self,
        value: torch.Tensor | _SparseTensor,
        context: torch.Tensor,
        projected: torch.Tensor,
    ) -> torch.Tensor | _SparseTensor:
        output = self.cross_attn_block(value, context)
        projection = self.proj_linear(projected)
        if isinstance(output, _SparseTensor):
            return output.replace(output.feats + projection.to(output.feats))
        return output + projection.to(output)


class Trellis2Mlp(torch.nn.Module):
    def __init__(self, config: Trellis2FlowConfig, *, operations: Operations) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(config.model_channels, config.mlp_channels),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(config.mlp_channels, config.model_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(value)


class Trellis2Block(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: Trellis2FlowConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
        microsoft_split_precision: bool,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.microsoft_split_precision = microsoft_split_precision
        self.norm1 = operations.layer_norm(hidden, eps=1e-6, elementwise_affine=False)
        self.norm2 = operations.layer_norm(hidden, eps=1e-6)
        self.norm3 = operations.layer_norm(hidden, eps=1e-6, elementwise_affine=False)
        self.self_attn = Trellis2Attention(
            config,
            cross=False,
            operations=operations,
            attention_kernel=attention_kernel,
            microsoft_split_precision=microsoft_split_precision,
        )
        if config.image_attention == "projected":
            self.cross_attn: Trellis2Attention | Trellis2ProjectedAttention = (
                Trellis2ProjectedAttention(
                    config,
                    operations=operations,
                    attention_kernel=attention_kernel,
                    microsoft_split_precision=microsoft_split_precision,
                )
            )
        else:
            self.cross_attn = Trellis2Attention(
                config,
                cross=True,
                operations=operations,
                attention_kernel=attention_kernel,
                microsoft_split_precision=microsoft_split_precision,
            )
        self.mlp = Trellis2Mlp(config, operations=operations)
        self.modulation = torch.nn.Parameter(torch.empty(hidden * 6))

    def _block_modulation(self, modulation: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            block = self.modulation
        else:
            with binding.lease() as lease:
                block = lease.get("modulation", dtype=self.modulation.dtype)
                return self._combine_modulation(block, modulation)
        return self._combine_modulation(block, modulation)

    def _combine_modulation(self, block: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        if self.microsoft_split_precision:
            return (block + modulation).to(modulation.dtype)
        return block.to(modulation) + modulation

    def forward(
        self,
        value: torch.Tensor | _SparseTensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        projected: torch.Tensor | None,
        rope: torch.Tensor | None,
    ) -> torch.Tensor | _SparseTensor:
        features = value.feats if isinstance(value, _SparseTensor) else value
        batch_map = value.batch_map if isinstance(value, _SparseTensor) else None
        mod = self._block_modulation(modulation)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        if batch_map is not None:
            shift_attn, scale_attn, gate_attn = (
                item[batch_map] for item in (shift_attn, scale_attn, gate_attn)
            )
            shift_mlp, scale_mlp, gate_mlp = (
                item[batch_map] for item in (shift_mlp, scale_mlp, gate_mlp)
            )
        else:
            shift_attn, scale_attn, gate_attn = (
                item[:, None] for item in (shift_attn, scale_attn, gate_attn)
            )
            shift_mlp, scale_mlp, gate_mlp = (
                item[:, None] for item in (shift_mlp, scale_mlp, gate_mlp)
            )
        normalized_features = self.norm1(
            features.float() if self.microsoft_split_precision else features
        )
        if self.microsoft_split_precision:
            normalized_features = normalized_features.float().to(features.dtype)
            normalized = normalized_features * (1 + scale_attn) + shift_attn
        else:
            normalized = torch.addcmul(shift_attn, normalized_features, 1 + scale_attn)
        attention_input = (
            value.replace(normalized) if isinstance(value, _SparseTensor) else normalized
        )
        attention = self.self_attn(attention_input, rope=rope)
        attention_features = attention.feats if isinstance(attention, _SparseTensor) else attention
        features = (
            features + attention_features * gate_attn
            if self.microsoft_split_precision
            else torch.addcmul(features, attention_features, gate_attn)
        )
        value = value.replace(features) if isinstance(value, _SparseTensor) else features
        normalized_features = self.norm2(
            features.float() if self.microsoft_split_precision else features
        )
        if self.microsoft_split_precision:
            normalized_features = normalized_features.to(features.dtype)
        cross_input = (
            value.replace(normalized_features)
            if isinstance(value, _SparseTensor)
            else normalized_features
        )
        if isinstance(self.cross_attn, Trellis2ProjectedAttention):
            assert projected is not None
            cross = self.cross_attn(cross_input, context, projected)
        else:
            cross = self.cross_attn(cross_input, context)
        cross_features = cross.feats if isinstance(cross, _SparseTensor) else cross
        features = features + cross_features
        normalized_features = self.norm3(
            features.float() if self.microsoft_split_precision else features
        )
        if self.microsoft_split_precision:
            normalized_features = normalized_features.float().to(features.dtype)
            normalized = normalized_features * (1 + scale_mlp) + shift_mlp
        else:
            normalized = torch.addcmul(shift_mlp, normalized_features, 1 + scale_mlp)
        mlp = self.mlp(normalized)
        features = (
            features + mlp * gate_mlp
            if self.microsoft_split_precision
            else torch.addcmul(features, mlp, gate_mlp)
        )
        return value.replace(features) if isinstance(value, _SparseTensor) else features


class Trellis2FlowModel(torch.nn.Module):
    """One parameterized structure, shape, or texture flow."""

    def __init__(
        self,
        config: Trellis2FlowConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
        compute_dtype: torch.dtype | None = None,
        microsoft_split_precision: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.microsoft_split_precision = microsoft_split_precision
        self.t_embedder = Trellis2TimestepEmbedder(config, operations=operations)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), operations.linear(config.model_channels, config.model_channels * 6)
        )
        self.input_layer = operations.linear(config.in_channels, config.model_channels)
        self.blocks = torch.nn.ModuleList(
            [
                Trellis2Block(
                    config,
                    operations=operations,
                    attention_kernel=attention_kernel,
                    microsoft_split_precision=microsoft_split_precision,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.out_layer = operations.linear(config.model_channels, config.out_channels)

    def forward(
        self,
        latent: torch.Tensor | SparseLatent[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        projected: torch.Tensor | None = None,
    ) -> torch.Tensor | SparseLatent[torch.Tensor]:
        support: SparseSupport[torch.Tensor] | None = None
        input_dtype: torch.dtype
        if type(latent) is SparseLatent:
            support, features = unpack_sparse_latent(latent)
            input_dtype = features.dtype
            value: torch.Tensor | _SparseTensor = _SparseTensor(
                support,
                self.input_layer(features.float() if self.microsoft_split_precision else features),
            )
            rope = None
        else:
            assert type(latent) is torch.Tensor
            if latent.shape[1:] != (32, 16, 16, 16):
                raise ValueError("TRELLIS.2 structure latent must have shape [B,32,16,16,16]")
            features = latent[:, : self.config.in_channels].flatten(2).transpose(1, 2)
            input_dtype = features.dtype
            value = self.input_layer(
                features.float() if self.microsoft_split_precision else features
            )
            one = torch.arange(16, dtype=torch.float32, device=latent.device)
            coordinates = torch.stack(torch.meshgrid(one, one, one, indexing="ij"), dim=-1)
            if self.microsoft_split_precision:
                rope = _microsoft_rope_table(
                    coordinates.reshape(-1, 3),
                    self.config.head_channels,
                    phases_on_cpu=True,
                )
            else:
                rope = _rope_table(coordinates.reshape(-1, 3), self.config.head_channels)
                rope = rope[None, :, None]
        embedding_dtype = torch.float32 if self.microsoft_split_precision else features.dtype
        modulation = self.adaLN_modulation(self.t_embedder(timestep, embedding_dtype))
        if self.microsoft_split_precision:
            assert self.compute_dtype is not None
            value = (
                value.replace(value.feats.to(self.compute_dtype))
                if isinstance(value, _SparseTensor)
                else value.to(self.compute_dtype)
            )
            modulation = modulation.to(self.compute_dtype)
            context = context.to(self.compute_dtype)
            if projected is not None:
                projected = projected.to(self.compute_dtype)
        for block in self.blocks:
            value = block(value, modulation, context, projected, rope)
        value_features = value.feats if isinstance(value, _SparseTensor) else value
        if self.microsoft_split_precision:
            value_features = value_features.to(input_dtype)
        output = self.out_layer(F.layer_norm(value_features, value_features.shape[-1:]))
        if support is not None:
            return pack_sparse_latent(authenticate_sparse_support(support), output)
        assert type(latent) is torch.Tensor
        dense = output.transpose(1, 2).reshape(
            latent.shape[0], self.config.out_channels, 16, 16, 16
        )
        return F.pad(dense, (0, 0, 0, 0, 0, 0, 0, 32 - self.config.out_channels))


__all__ = ["Trellis2FlowModel"]
