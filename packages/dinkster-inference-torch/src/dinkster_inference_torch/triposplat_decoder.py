"""Native torch TripoSplat octree gaussian decoder.

Transcribed from ``comfy/ldm/triposplat/vae.py`` and ``gaussian.py``
@ 36408117. Two cross-attention transformers condition on the denoised
16-channel latent sequence: the octree probability decoder descends an
occupancy octree level by level (systematic resampling distributes the
point budget over predicted child probabilities), then the elastic
gaussian decoder predicts a packed 480-channel feature per sampled
point that unpacks into 32 gaussians (position offsets, DC spherical
harmonics, scaling, rotation, opacity, and a per-gaussian offset
scale).

Randomness is explicit: every draw goes through the caller's CPU
``torch.Generator``, so decodes are reproducible without touching
global RNG (the reference reads the global RNG when no generator is
passed; Dinkster always requires one). Activation into render-ready
splat tensors (:func:`render_splat_tensors`) matches the reference
GaussianModel: softplus scaling over a 0.004 bias with the 3D filter
kernel folded in, sigmoid opacity over a 0.1 bias, a [1, 0, 0, 0]
quaternion bias, the [-0.5, 0.5] cube placement, and the object-frame
to viewer-Y-up axis rotation baked into positions and rotations.

State-dict keys match the published checkpoint exactly
(:func:`dinkster_inference.triposplat.triposplat_gaussian_decoder_layout`);
the hammersley perturbation and base offset scale are persistent
buffers.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from dinkster_inference import TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, ResidencyRouted, bound_compute_dtype
from .ops import cast_weight
from .triposplat_model import (
    TripoSplatMlp,
    TripoSplatMultiHeadRMSNorm,
    pcd_position_embedding,
)

_DEFAULT_ATTENTION = select_attention("vae").kernel
_NORM_EPS = 1e-6
_LEVEL_FREQUENCIES = 256
_LEVEL_MAX_PERIOD = 1024
_PCD_MAX_RES = 10
_PERTURBE_SIZE = 1.5
_OFFSET_SCALE = 0.05
_ROTATION_LR = 0.1
_FILTER_KERNEL_3D = 0.0009
_SCALING_BIAS = 0.004
_OPACITY_BIAS = 0.1
_AABB_OFFSET = -0.5
# Object frame -> viewer Y-up frame, baked into rendered tensors.
_AXIS_TRANSFORM = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


class TripoSplatDecoderConfig(Protocol):
    """The architecture facts the decoder consumes (satisfied by
    :class:`dinkster_inference.TripoSplatGaussianDecoderConfig`)."""

    @property
    def model_channels(self) -> int: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def octree_blocks(self) -> int: ...
    @property
    def gaussian_blocks(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def attention_head_dim(self) -> int: ...
    @property
    def mlp_ratio(self) -> int: ...
    @property
    def gaussians_per_point(self) -> int: ...
    @property
    def feature_channels(self) -> int: ...
    @property
    def max_voxel_level(self) -> int: ...


def _radical_inverse(base: int, n: int) -> float:
    value = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        value += (n % base) * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return value


def hammersley_sequence(dim: int, n: int, num_samples: int) -> list[float]:
    """The reference low-discrepancy point set seeding the per-point
    gaussian offset perturbation."""
    return [n / num_samples] + [_radical_inverse(_PRIMES[i], n) for i in range(dim - 1)]


def sample_probs(
    probs: torch.Tensor, counts: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Systematic resampling: distribute ``counts[r]`` draws across the
    bins of probability row ``r`` (sample_probs @ 36408117). The one
    uniform draw per row comes from the CPU ``generator``."""
    batch_shape = counts.shape
    rows = counts.numel()
    bins = probs.size(-1)
    device = probs.device
    probs = probs.reshape(rows, bins).to(torch.float32).clamp_min(0)
    counts = counts.reshape(rows).to(device=device, dtype=torch.long)

    row_sums = probs.sum(1, keepdim=True)
    probs = torch.where(row_sums == 0, probs.new_tensor(1.0 / bins), probs / row_sums.clamp_min(1))
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)

    max_count = int(counts.max())
    if max_count == 0:
        return counts.new_zeros(*batch_shape, bins)
    clamped = counts.clamp_min(1).float().unsqueeze(1)
    grid = torch.arange(max_count, device=device, dtype=torch.float32).unsqueeze(0)
    u = (torch.rand(rows, 1, generator=generator).to(device) + grid) / clamped
    indices = torch.searchsorted(cdf, u.clamp(max=1.0 - 1e-12)).clamp_max(bins - 1)
    weight = (grid < counts.unsqueeze(1)).to(cdf.dtype)
    out = torch.zeros(rows, bins, dtype=torch.float32, device=device)
    out.scatter_add_(1, indices, weight)
    return out.to(torch.long).view(*batch_shape, bins)


class DecoderSelfAttention(torch.nn.Module):
    """Fused-qkv self-attention with per-head q/k RMS norms
    (MultiHeadAttention type="self" @ 36408117; no rope)."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.heads = config.attention_heads
        self.to_qkv = operations.linear(hidden, hidden * 3)
        self.q_rms_norm = TripoSplatMultiHeadRMSNorm(config.attention_head_dim, self.heads)
        self.k_rms_norm = TripoSplatMultiHeadRMSNorm(config.attention_head_dim, self.heads)
        self.to_out = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.to_qkv(x).reshape(batch, length, 3, self.heads, -1).unbind(dim=2)
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        attended = self._attention_kernel(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return self.to_out(attended.reshape(batch, length, channels))


class DecoderCrossAttention(torch.nn.Module):
    """Cross-attention over the latent sequence with per-head q/k RMS
    norms (MultiHeadAttention type="cross" @ 36408117)."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.heads = config.attention_heads
        self.to_q = operations.linear(hidden, hidden)
        self.to_kv = operations.linear(config.latent_channels, hidden * 2)
        self.q_rms_norm = TripoSplatMultiHeadRMSNorm(config.attention_head_dim, self.heads)
        self.k_rms_norm = TripoSplatMultiHeadRMSNorm(config.attention_head_dim, self.heads)
        self.to_out = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        rows = context.shape[1]
        q = self.to_q(x).reshape(batch, length, self.heads, -1)
        k, v = self.to_kv(context).reshape(batch, rows, 2, self.heads, -1).unbind(dim=2)
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        attended = self._attention_kernel(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return self.to_out(attended.reshape(batch, length, channels))


def level_embedding(t: torch.Tensor, dim: int, max_period: int = _LEVEL_MAX_PERIOD) -> torch.Tensor:
    """Sinusoidal octree-level table (LevelEmbedder @ 36408117):
    float32, a 2*pi angle factor, cos-then-sin."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None] * 2 * torch.pi
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TripoSplatLevelEmbedder(torch.nn.Module):
    def __init__(self, hidden: int, *, operations: Operations) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(_LEVEL_FREQUENCIES, hidden),
            torch.nn.SiLU(),
            operations.linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(level_embedding(t, _LEVEL_FREQUENCIES).to(dtype))


class OctreeCrossBlock(torch.nn.Module):
    """Cross-only modulated block (ModulatedTransformerCrossOnlyBlock
    @ 36408117, share_mod: the decoder-level modulation projection is
    chunked directly; both norms are non-affine)."""

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.norm1 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=False)
        self.norm2 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=False)
        self.cross_attn = DecoderCrossAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.mlp = TripoSplatMlp(hidden, hidden * config.mlp_ratio, hidden, operations=operations)

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = torch.addcmul(shift_msa.unsqueeze(1), self.norm1(x), 1 + scale_msa.unsqueeze(1))
        x = torch.addcmul(x, self.cross_attn(h, context), gate_msa.unsqueeze(1))
        h = torch.addcmul(shift_mlp.unsqueeze(1), self.norm2(x), 1 + scale_mlp.unsqueeze(1))
        return torch.addcmul(x, self.mlp(h), gate_mlp.unsqueeze(1))


class GaussianCrossBlock(torch.nn.Module):
    """Self plus cross block (TransformerCrossBlock @ 36408117): only
    the cross-attention pre-norm is affine; plain residuals."""

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.norm1 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=False)
        self.norm2 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=True)
        self.norm3 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=False)
        self.self_attn = DecoderSelfAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.cross_attn = DecoderCrossAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.mlp = TripoSplatMlp(hidden, hidden * config.mlp_ratio, hidden, operations=operations)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class OctreeProbabilityDecoder(torch.nn.Module):
    """Octree node coordinates -> 8-way child occupancy logits
    (OctreeProbabilityFixedlenDecoder @ 36408117)."""

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.model_channels = hidden
        self.input_layer = operations.linear(hidden, hidden)
        self.l_embedder = TripoSplatLevelEmbedder(hidden, operations=operations)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), operations.linear(hidden, 6 * hidden)
        )
        self.blocks = torch.nn.ModuleList(
            OctreeCrossBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.octree_blocks)
        )
        self.out_proj = operations.linear(hidden, 8)
        self.in_proj = operations.linear(3, hidden)

    def forward(
        self, points: torch.Tensor, levels: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        dtype = cond.dtype
        batch, length, _ = points.shape
        positions = pcd_position_embedding(
            points.reshape(-1, 3), self.model_channels, max_res=_PCD_MAX_RES, schedule="log2"
        )
        h = self.in_proj(points.to(dtype)) + positions.reshape(batch, length, -1).to(dtype)
        h = self.input_layer(h)
        mod = self.adaLN_modulation(self.l_embedder(levels, dtype=dtype))
        for block in self.blocks:
            h = block(h, mod, cond)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(dtype)
        return self.out_proj(h)


class OctreePoints(NamedTuple):
    """Sampled octree leaf points in the unit cube with their
    autoregressive log-probabilities."""

    points: torch.Tensor
    log_probs: torch.Tensor


def sample_octree_points(
    decoder: OctreeProbabilityDecoder,
    cond: torch.Tensor,
    *,
    num_points: int,
    level: int,
    generator: torch.Generator,
    temperature: float = 1.0,
) -> OctreePoints:
    """Autoregressive octree descent (OctreeProbabilityFixedlenDecoder
    .sample @ 36408117): at each level the point budget is
    redistributed over predicted child occupancies by systematic
    resampling; leaves are jittered uniformly inside their voxel."""
    batch = cond.shape[0]
    device = cond.device
    child_offset = torch.tensor(
        [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)],
        dtype=torch.long,
        device=device,
    )
    prev_coords_int = torch.zeros(batch, 1, 3, dtype=torch.long, device=device)
    prev_counts = torch.full((batch, 1), num_points, dtype=torch.long, device=device)
    prev_log_probs = torch.zeros(batch, 1, dtype=torch.float32, device=device)
    batch_indices_range = torch.arange(batch, device=device).unsqueeze(1)

    for current in range(1, level + 1):
        parent_res = 1 << (current - 1)
        res = 1 << current
        parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / parent_res
        res_tensor = torch.full((batch,), res, dtype=torch.long, device=device)
        pred_logits = decoder(parent_coords_norm, res_tensor, cond) / temperature
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
        sampled = sample_probs(pred_probs, prev_counts, generator).flatten(1, 2)
        pred_log_probs = pred_log_probs.flatten(1, 2)
        prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, dim=1)
        child_coords_int = (
            prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]
        ).flatten(1, 2)
        mask = sampled > 0
        max_valid = int(mask.sum(dim=1).max().item())
        scatter_indices = mask.cumsum(dim=1) - 1
        valid_scatter_indices = scatter_indices[mask]
        valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
        next_coords = torch.zeros(batch, max_valid, 3, dtype=child_coords_int.dtype, device=device)
        next_coords[valid_batch_indices, valid_scatter_indices] = child_coords_int[mask]
        next_counts = torch.zeros(batch, max_valid, dtype=sampled.dtype, device=device)
        next_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
        next_log_probs = torch.zeros(batch, max_valid, dtype=prev_log_probs.dtype, device=device)
        next_log_probs[valid_batch_indices, valid_scatter_indices] = (
            prev_log_probs_expanded + pred_log_probs
        )[mask]
        prev_coords_int = next_coords
        prev_counts = next_counts
        prev_log_probs = next_log_probs

    res = 1 << level
    log_probs = torch.repeat_interleave(
        prev_log_probs.flatten(0, 1), prev_counts.flatten(0, 1), dim=0
    ).reshape(batch, num_points)
    coords_int = torch.repeat_interleave(
        prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0
    ).reshape(batch, num_points, -1)
    jitter = torch.rand(coords_int.shape, dtype=torch.float32, generator=generator).to(device)
    points = (coords_int.to(torch.float32) + jitter) / res
    return OctreePoints(points=points, log_probs=log_probs)


class ElasticGaussianDecoder(ResidencyRouted, torch.nn.Module):
    """Sampled octree points -> packed per-point gaussian features
    (ElasticGaussianFixedlenDecoder @ 36408117). The packed layout
    orders xyz, DC features, scaling, rotation, opacity, then the
    per-gaussian offset scale."""

    def __init__(
        self,
        config: TripoSplatDecoderConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        gaussians = config.gaussians_per_point
        if config.feature_channels != gaussians * 15:
            raise ValueError(
                "feature channels must pack 15 values per gaussian"
                f" ({gaussians} * 15 != {config.feature_channels})"
            )
        self.gaussians_per_point = gaussians
        self.model_channels = hidden
        self.input_layer = operations.linear(hidden, hidden)
        self.blocks = torch.nn.ModuleList(
            GaussianCrossBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.gaussian_blocks)
        )
        self.in_proj = operations.linear(3, hidden)
        self.out_proj = operations.linear(hidden, config.feature_channels)
        perturbation = torch.tensor(
            [hammersley_sequence(3, i, gaussians) for i in range(gaussians)],
            dtype=torch.float32,
        )
        perturbation = torch.atanh((perturbation * 2 - 1) / _PERTURBE_SIZE)
        self.register_buffer("points_offset_perturbation", perturbation)
        base = torch.tensor(_OFFSET_SCALE)
        self.register_buffer("base_offset_scale", torch.log(torch.exp(base) - 1.0))

    def offsets(self, features: torch.Tensor) -> torch.Tensor:
        """Per-gaussian position offsets from the packed features
        (_get_offset @ 36408117): a softplus-activated per-gaussian
        scale bounds a tanh-squashed, hammersley-perturbed offset."""
        batch = features.shape[0]
        gaussians = self.gaussians_per_point
        binding = self._offloaded_residency()
        if binding is None:
            perturbation = cast_weight(
                self.get_buffer("points_offset_perturbation"),
                dtype=features.dtype,
                device=features.device,
            )
            base = cast_weight(
                self.get_buffer("base_offset_scale"), dtype=features.dtype, device=features.device
            )
        else:
            with binding.lease() as lease:
                perturbation = lease.get("points_offset_perturbation", dtype=features.dtype)
                base = lease.get("base_offset_scale", dtype=features.dtype)
        scale = F.softplus(
            features[:, :, 14 * gaussians : 15 * gaussians].reshape(batch, -1, gaussians, 1) + base
        )
        offset = features[:, :, : 3 * gaussians].reshape(batch, -1, gaussians, 3)
        offset = offset + perturbation
        offset = torch.tanh(offset) * 0.5 * _PERTURBE_SIZE
        return offset * scale

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        dtype = cond.dtype
        batch, length, _ = points.shape
        positions = pcd_position_embedding(
            points.reshape(-1, 3), self.model_channels, max_res=_PCD_MAX_RES, schedule="log2"
        )
        h = self.in_proj(points.to(dtype)) + positions.reshape(batch, length, -1).to(dtype)
        h = self.input_layer(h)
        for block in self.blocks:
            h = block(h, cond)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(h.dtype)
        return self.out_proj(h)


class SplatTensors(NamedTuple):
    """Render-ready gaussian splat tensors for one batch item:
    activated, world-space, viewer-Y-up (GaussianModel.render_tensors
    @ 36408117)."""

    positions: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    q = torch.zeros((rotation.shape[0], 4), dtype=rotation.dtype, device=rotation.device)
    s = torch.sqrt(torch.clamp(trace + 1, min=0)) * 2
    q[:, 0] = 0.25 * s
    denom = torch.where(s != 0, s, torch.ones_like(s))
    q[:, 1] = (rotation[:, 2, 1] - rotation[:, 1, 2]) / denom
    q[:, 2] = (rotation[:, 0, 2] - rotation[:, 2, 0]) / denom
    q[:, 3] = (rotation[:, 1, 0] - rotation[:, 0, 1]) / denom
    m01 = (
        (rotation[:, 0, 0] >= rotation[:, 1, 1])
        & (rotation[:, 0, 0] >= rotation[:, 2, 2])
        & (s == 0)
    )
    s1 = (
        torch.sqrt(
            torch.clamp(1 + rotation[:, 0, 0] - rotation[:, 1, 1] - rotation[:, 2, 2], min=0)
        )
        * 2
    )
    q[m01, 0] = (rotation[m01, 2, 1] - rotation[m01, 1, 2]) / s1[m01]
    q[m01, 1] = 0.25 * s1[m01]
    q[m01, 2] = (rotation[m01, 0, 1] + rotation[m01, 1, 0]) / s1[m01]
    q[m01, 3] = (rotation[m01, 0, 2] + rotation[m01, 2, 0]) / s1[m01]
    m11 = (
        (rotation[:, 1, 1] > rotation[:, 0, 0])
        & (rotation[:, 1, 1] >= rotation[:, 2, 2])
        & (s == 0)
    )
    s2 = (
        torch.sqrt(
            torch.clamp(1 + rotation[:, 1, 1] - rotation[:, 0, 0] - rotation[:, 2, 2], min=0)
        )
        * 2
    )
    q[m11, 0] = (rotation[m11, 0, 2] - rotation[m11, 2, 0]) / s2[m11]
    q[m11, 1] = (rotation[m11, 0, 1] + rotation[m11, 1, 0]) / s2[m11]
    q[m11, 2] = 0.25 * s2[m11]
    q[m11, 3] = (rotation[m11, 1, 2] + rotation[m11, 2, 1]) / s2[m11]
    m21 = (
        (rotation[:, 2, 2] > rotation[:, 0, 0]) & (rotation[:, 2, 2] > rotation[:, 1, 1]) & (s == 0)
    )
    s3 = (
        torch.sqrt(
            torch.clamp(1 + rotation[:, 2, 2] - rotation[:, 0, 0] - rotation[:, 1, 1], min=0)
        )
        * 2
    )
    q[m21, 0] = (rotation[m21, 1, 0] - rotation[m21, 0, 1]) / s3[m21]
    q[m21, 1] = (rotation[m21, 0, 2] + rotation[m21, 2, 0]) / s3[m21]
    q[m21, 2] = (rotation[m21, 1, 2] + rotation[m21, 2, 1]) / s3[m21]
    q[m21, 3] = 0.25 * s3[m21]
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def render_splat_tensors(
    decoder: ElasticGaussianDecoder, points: torch.Tensor, features: torch.Tensor
) -> tuple[SplatTensors, ...]:
    """Unpack and activate one splat per batch item
    (build_gaussian_models + GaussianModel.render_tensors @ 36408117).
    All outputs are float32 and contiguous on the input device."""
    gaussians = decoder.gaussians_per_point
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [batch, tokens, 3], got {tuple(points.shape)}")
    if (
        features.ndim != 3
        or features.shape[-1] != gaussians * 15
        or (features.shape[:2] != points.shape[:2])
    ):
        raise ValueError(
            f"features must be [batch, tokens, {gaussians * 15}] matching points,"
            f" got {tuple(features.shape)}"
        )
    offsets = decoder.offsets(features)
    scale_bias = _SCALING_BIAS + math.log(-math.expm1(-_SCALING_BIAS))
    opacity_bias = math.log(_OPACITY_BIAS / (1.0 - _OPACITY_BIAS))
    axis = torch.tensor(_AXIS_TRANSFORM, dtype=torch.float32, device=features.device)
    results: list[SplatTensors] = []
    for index in range(features.shape[0]):
        item = features[index]

        def channel(start: int, width: int, item: torch.Tensor = item) -> torch.Tensor:
            span = item[:, start * gaussians : (start + width) * gaussians]
            return span.reshape(-1, gaussians, width).flatten(0, 1).float()

        raw_xyz = (offsets[index] + points[index, :, None, :]).flatten(0, 1).float()
        positions = raw_xyz + _AABB_OFFSET
        sh = channel(3, 3).reshape(-1, 1, 3)
        scaling = F.softplus(channel(6, 3) + scale_bias)
        scales = torch.sqrt(torch.square(scaling) + _FILTER_KERNEL_3D**2)
        rotations = channel(9, 4) * _ROTATION_LR
        rotations[:, 0] = rotations[:, 0] + 1.0
        rotations = _matrix_to_quaternion(axis @ _quaternion_to_matrix(rotations))
        rotations = rotations / torch.linalg.norm(rotations, dim=-1, keepdim=True)
        opacities = torch.sigmoid(channel(13, 1) + opacity_bias)
        positions = positions @ axis.T
        results.append(
            SplatTensors(
                positions=positions.contiguous(),
                scales=scales.contiguous(),
                rotations=rotations.contiguous(),
                opacities=opacities.contiguous(),
                sh=sh.contiguous(),
            )
        )
    return tuple(results)


class OctreeGaussianDecoder(torch.nn.Module):
    """The combined first-stage decoder (OctreeGaussianDecoder
    @ 36408117): octree point sampling followed by elastic gaussian
    decoding into render-ready splat tensors."""

    def __init__(
        self,
        config: TripoSplatDecoderConfig = TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config.model_channels != config.attention_heads * config.attention_head_dim:
            raise ValueError("decoder width must equal attention heads times head dimension")
        self.config = config
        self.octree = OctreeProbabilityDecoder(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.gs = ElasticGaussianDecoder(
            config, operations=operations, attention_kernel=attention_kernel
        )

    @property
    def gaussians_per_point(self) -> int:
        return self.gs.gaussians_per_point

    def decode(
        self,
        latent: torch.Tensor,
        *,
        num_gaussians: int,
        generator: torch.Generator,
        level: int | None = None,
    ) -> tuple[SplatTensors, ...]:
        """Decode the denoised latent sequence into one splat per batch
        item. ``level`` below the full octree depth is cheaper and
        coarser (the live-preview leg); ``num_gaussians`` rounds down
        to whole decoder tokens."""
        if latent.ndim != 3 or latent.shape[-1] != self.config.latent_channels:
            raise ValueError(
                f"latent must be [batch, tokens, {self.config.latent_channels}],"
                f" got {tuple(latent.shape)}"
            )
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be positive")
        level = self.config.max_voxel_level if level is None else level
        if not 1 <= level <= self.config.max_voxel_level:
            raise ValueError(
                f"octree level must be in [1, {self.config.max_voxel_level}], got {level}"
            )
        compute_dtype = bound_compute_dtype(self.octree.input_layer)
        if compute_dtype is not None:
            latent = latent.to(dtype=compute_dtype)
        num_tokens = max(1, num_gaussians // self.gaussians_per_point)
        sampled = sample_octree_points(
            self.octree, latent, num_points=num_tokens, level=level, generator=generator
        )
        features = self.gs(sampled.points, latent)
        return render_splat_tensors(self.gs, sampled.points, features)


__all__ = [
    "DecoderCrossAttention",
    "DecoderSelfAttention",
    "ElasticGaussianDecoder",
    "GaussianCrossBlock",
    "OctreeCrossBlock",
    "OctreeGaussianDecoder",
    "OctreePoints",
    "OctreeProbabilityDecoder",
    "SplatTensors",
    "TripoSplatDecoderConfig",
    "TripoSplatLevelEmbedder",
    "hammersley_sequence",
    "level_embedding",
    "render_splat_tensors",
    "sample_octree_points",
    "sample_probs",
]
