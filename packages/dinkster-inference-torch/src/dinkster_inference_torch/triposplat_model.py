"""Native torch TripoSplat flow denoiser.

Transcribed from ``comfy/ldm/triposplat/model.py`` (LatentSeqMMFlowModel
@ 36408117): a shared-modulation transformer that jointly denoises the
fixed (B, 8192, 16) latent token sequence and a (B, 1, 5) camera token.
Conditioning is a DINOv3 token sequence (cross-attention context is
concatenated, not cross-attended: latent, context, and camera streams
run through one joint self-attention) plus an optional Flux2 VAE
reference-image latent added through a second embedder. Rotary angles
are not positional: every block predicts per-token 3D rotation angles
from its own hidden states (RePo3DRotaryEmbedding), and rope is applied
to queries and keys BEFORE the per-head q/k RMS norms.

State-dict keys match the published checkpoint exactly
(:func:`dinkster_inference.triposplat.triposplat_layout`). The Sobol
anchor embedding is a non-persistent buffer recomputed at construction.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from dinkster_inference import TRIPOSPLAT_CONFIG

from .attention import AttentionKernel, select_attention
from .flux import apply_rope
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("flux").kernel
_NORM_EPS = 1e-6
_REPO_MAX_FREQ = 16.0
_SOBOL_SEED = 123
_POS_EMB_MAX_RES = 16
_TIMESTEP_FREQUENCIES = 256


class TripoSplatModelConfig(Protocol):
    """The architecture facts :class:`TripoSplatModel` consumes
    (satisfied by :class:`dinkster_inference.TripoSplatConfig`)."""

    @property
    def q_token_length(self) -> int: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def model_channels(self) -> int: ...
    @property
    def cond_channels(self) -> int: ...
    @property
    def cond2_channels(self) -> int: ...
    @property
    def num_blocks(self) -> int: ...
    @property
    def num_refiner_blocks(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def attention_head_dim(self) -> int: ...
    @property
    def cam_channels(self) -> int: ...
    @property
    def mlp_ratio(self) -> int: ...
    @property
    def repo_hidden_size(self) -> int: ...


def pcd_position_embedding(
    x: torch.Tensor,
    channels: int,
    *,
    max_res: int = 16,
    schedule: str = "pow2",
) -> torch.Tensor:
    """Sinusoidal embedding of 3D point coordinates
    (PcdAbsolutePositionEmbedder @ 36408117). ``pow2`` embeds the flow
    model's Sobol anchors; ``log2`` embeds octree coordinates inside
    the gaussian decoder. Parameter-free; sin-then-cos, zero-padded up
    to ``channels``."""
    in_channels = x.shape[-1]
    freq_dim = channels // in_channels // 2
    orig_dtype = x.dtype
    x = x.float()
    if schedule == "pow2":
        # The *2 folds this schedule's 2*pi into the shared *pi below.
        base = torch.arange(max_res, dtype=torch.float32, device=x.device)
        res_dim = max(0, freq_dim - max_res)
        if res_dim > 0:
            extra = torch.arange(res_dim, dtype=torch.float32, device=x.device)
            base = torch.cat([base, extra / max(res_dim, 1) * max_res], dim=0)
        freqs = torch.pow(2.0, base[:freq_dim]) * 2.0
    elif schedule == "log2":
        logs = torch.linspace(
            0.0, float(max_res), steps=freq_dim, dtype=torch.float32, device=x.device
        )
        freqs = torch.pow(2.0, logs)
    else:
        raise ValueError(f"unknown pcd embedding schedule {schedule!r}")
    dims = x.shape[:-1]
    out = torch.outer(x.reshape(-1), freqs) * torch.pi
    out = torch.cat([out.sin(), out.cos()], dim=-1).reshape(*dims, -1)
    if out.shape[-1] < channels:
        pad = torch.zeros(*dims, channels - out.shape[-1], device=out.device, dtype=out.dtype)
        out = torch.cat([out, pad], dim=-1)
    return out.to(orig_dtype)


def triposplat_timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """The reference sinusoidal timestep table (TimestepEmbedder
    @ 36408117): float32, cos-then-sin, zero-padded for odd ``dim``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TripoSplatMultiHeadRMSNorm(ResidencyRouted, torch.nn.Module):
    """Per-head RMS norm with a (heads, head_dim) gamma
    (MultiHeadRMSNorm @ 36408117): weight-free rms_norm at eps 1e-6,
    then an elementwise gamma product."""

    def __init__(self, head_dim: int, heads: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.empty(heads, head_dim))

    def _normalize(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=_NORM_EPS) * gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._normalize(x, cast_weight(self.gamma, dtype=x.dtype, device=x.device))
        with binding.lease() as lease:
            return self._normalize(x, lease.get("gamma", dtype=x.dtype))


class TripoSplatRepoRotary(ResidencyRouted, torch.nn.Module):
    """Content-dependent rotary table (RePo3DRotaryEmbedding
    @ 36408117): a gated projection predicts per-token, per-head 3D
    rotation angles; three TRAINED frequency vectors (linspace init,
    then learned) expand them over the head dimension split
    2*(d//6) / 2*(d//6) / remainder. Returns flux-layout rotation
    matrices [B, L, heads, head_dim/2, 2, 2] in float32."""

    def __init__(self, config: TripoSplatModelConfig, *, operations: Operations) -> None:
        super().__init__()
        hidden = config.model_channels
        head_dim = config.attention_head_dim
        self.heads = config.attention_heads
        dim_0 = 2 * (head_dim // 6)
        dim_2 = head_dim - 2 * dim_0
        self.norm = operations.layer_norm(hidden)
        self.gate_map = operations.linear(hidden, config.repo_hidden_size, bias=False)
        self.content_map = operations.linear(hidden, config.repo_hidden_size, bias=False)
        self.final_map = operations.linear(config.repo_hidden_size, 3 * self.heads, bias=False)
        self.freqs_0 = torch.nn.Parameter(torch.empty(dim_0 // 2))
        self.freqs_1 = torch.nn.Parameter(torch.empty(dim_0 // 2))
        self.freqs_2 = torch.nn.Parameter(torch.empty(dim_2 // 2))

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return torch.float32

    def _angles(self, delta_pos: torch.Tensor, freqs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        parts = [delta_pos[..., axis].unsqueeze(-1) * freqs[axis] * torch.pi for axis in range(3)]
        return torch.cat(parts, dim=-1).float()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states)
        feat = F.silu(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        batch, length, _ = out.shape
        delta_pos = out.reshape(batch, length, self.heads, 3)
        binding = self._offloaded_residency()
        if binding is None:
            freqs = tuple(
                cast_weight(parameter, dtype=torch.float32, device=out.device)
                for parameter in (self.freqs_0, self.freqs_1, self.freqs_2)
            )
            angles = self._angles(delta_pos, freqs)
        else:
            with binding.lease() as lease:
                freqs = tuple(lease.get(f"freqs_{axis}", dtype=torch.float32) for axis in range(3))
                angles = self._angles(delta_pos, freqs)
        cos, sin = angles.cos(), angles.sin()
        return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*angles.shape, 2, 2)


class TripoSplatMlp(torch.nn.Module):
    """Linear / tanh-GELU / Linear under a nested ``mlp`` Sequential,
    matching the checkpoint's ``mlp.mlp.{0,2}`` key nesting."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, *, operations: Operations
    ) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(in_channels, hidden_channels),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TripoSplatAttention(torch.nn.Module):
    """Fused-qkv self-attention with rope applied BEFORE the per-head
    q/k RMS norms (RopeMultiHeadAttention @ 36408117, use_rope and
    qk_rms_norm both on in every TripoSplat block)."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: TripoSplatModelConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.heads = config.attention_heads
        self.head_dim = config.attention_head_dim
        self.qkv = operations.linear(hidden, hidden * 3)
        self.q_norm = TripoSplatMultiHeadRMSNorm(self.head_dim, self.heads)
        self.k_norm = TripoSplatMultiHeadRMSNorm(self.head_dim, self.heads)
        self.out = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = apply_rope(q, k, rope)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attended = self._attention_kernel(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return self.out(attended.reshape(batch, length, channels))


class TripoSplatBlock(ResidencyRouted, torch.nn.Module):
    """One joint transformer block (UnifiedTransformerBlock
    @ 36408117). Modulated blocks share the model-level modulation
    projection and add a per-block shift table; unmodulated blocks
    (the context refiner) carry affine norms and plain residuals."""

    def __init__(
        self,
        config: TripoSplatModelConfig,
        *,
        modulated: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        self.modulated = modulated
        self.norm1 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=not modulated)
        self.norm2 = operations.layer_norm(hidden, eps=_NORM_EPS, elementwise_affine=not modulated)
        self.attn = TripoSplatAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.mlp = TripoSplatMlp(hidden, hidden * config.mlp_ratio, hidden, operations=operations)
        if modulated:
            self.shift_table = torch.nn.Parameter(torch.empty(1, 6 * hidden))

    def _modulated_forward(
        self, x: torch.Tensor, mod: torch.Tensor, rope: torch.Tensor, shift_table: torch.Tensor
    ) -> torch.Tensor:
        mod = mod + shift_table
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = torch.addcmul(shift_msa.unsqueeze(1), self.norm1(x), 1 + scale_msa.unsqueeze(1))
        x = torch.addcmul(x, self.attn(h, rope), gate_msa.unsqueeze(1))
        h = torch.addcmul(shift_mlp.unsqueeze(1), self.norm2(x), 1 + scale_mlp.unsqueeze(1))
        return torch.addcmul(x, self.mlp(h), gate_mlp.unsqueeze(1))

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor, mod: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.modulated:
            x = x + self.attn(self.norm1(x), rope)
            return x + self.mlp(self.norm2(x))
        if mod is None:
            raise ValueError("modulated TripoSplat blocks require the shared modulation vector")
        binding = self._offloaded_residency()
        if binding is None:
            shift_table = cast_weight(self.shift_table, dtype=mod.dtype, device=mod.device)
            return self._modulated_forward(x, mod, rope, shift_table)
        with binding.lease() as lease:
            return self._modulated_forward(x, mod, rope, lease.get("shift_table", dtype=mod.dtype))


class TripoSplatTimestepEmbedder(torch.nn.Module):
    """256-frequency sinusoidal timestep MLP (TimestepEmbedder
    @ 36408117); the caller names the compute dtype."""

    def __init__(self, hidden: int, *, operations: Operations) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(_TIMESTEP_FREQUENCIES, hidden),
            torch.nn.SiLU(),
            operations.linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(triposplat_timestep_embedding(t, _TIMESTEP_FREQUENCIES).to(dtype))


class TripoSplatModel(ResidencyRouted, torch.nn.Module):
    """The TripoSplat flow denoiser over explicit latent and camera
    streams (LatentSeqMMFlowModel @ 36408117 unpacks the same pair
    from its nested latent). Returns the denoised (latent, camera)
    tuple."""

    _dinkster_residency_constant_buffers = frozenset({"pos_emb"})
    pos_emb: torch.Tensor

    def __init__(
        self,
        config: TripoSplatModelConfig = TRIPOSPLAT_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        hidden = config.model_channels
        head_dim = config.attention_head_dim
        if hidden != config.attention_heads * head_dim:
            raise ValueError("model width must equal attention heads times head dimension")
        if head_dim < 6 or head_dim % 2:
            raise ValueError("head dimension must be even and at least 6 for three rope axes")
        if (head_dim - 4 * (head_dim // 6)) % 2:
            raise ValueError("head dimension must split into even rope axis widths")
        self.config = config
        self.t_embedder = TripoSplatTimestepEmbedder(hidden, operations=operations)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), operations.linear(hidden, 6 * hidden)
        )
        self.input_layer = operations.linear(config.latent_channels, hidden)
        self.cond_embedder = operations.linear(config.cond_channels, hidden)
        self.cond_embedder2 = operations.linear(config.cond2_channels, hidden)

        # Fixed Sobol 3D anchors embedded once as the latent stream's
        # absolute positions; forced onto CPU so meta-device
        # construction cannot corrupt the deterministic draw.
        with torch.device("cpu"):
            anchors = torch.quasirandom.SobolEngine(
                dimension=3, scramble=True, seed=_SOBOL_SEED
            ).draw(config.q_token_length)
            pos_emb = pcd_position_embedding(
                anchors.unsqueeze(0), hidden, max_res=_POS_EMB_MAX_RES, schedule="pow2"
            )
        self.register_buffer("pos_emb", pos_emb, persistent=False)

        def rotary_layers(count: int) -> torch.nn.ModuleList:
            return torch.nn.ModuleList(
                TripoSplatRepoRotary(config, operations=operations) for _ in range(count)
            )

        def blocks(count: int, *, modulated: bool) -> torch.nn.ModuleList:
            return torch.nn.ModuleList(
                TripoSplatBlock(
                    config,
                    modulated=modulated,
                    operations=operations,
                    attention_kernel=attention_kernel,
                )
                for _ in range(count)
            )

        self.noise_repo_layers = rotary_layers(config.num_refiner_blocks)
        self.context_repo_layers = rotary_layers(config.num_refiner_blocks)
        self.repo_layers = rotary_layers(config.num_blocks)
        self.noise_refiner = blocks(config.num_refiner_blocks, modulated=True)
        self.context_refiner = blocks(config.num_refiner_blocks, modulated=False)
        self.cam_refiner = TripoSplatMlp(config.cam_channels, hidden, hidden, operations=operations)
        self.blocks = blocks(config.num_blocks, modulated=True)
        self.shift_table = torch.nn.Parameter(torch.empty(1, 2, hidden))
        self.out_layer = operations.linear(hidden, config.latent_channels)
        self.cam_out_layer = operations.linear(hidden, config.cam_channels)

    def _validate(
        self,
        latent: torch.Tensor,
        camera: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        reference_latent: torch.Tensor | None,
    ) -> None:
        config = self.config
        expected = (latent.shape[0], config.q_token_length, config.latent_channels)
        if latent.ndim != 3 or tuple(latent.shape) != expected:
            raise ValueError(f"latent must have shape {expected}, got {tuple(latent.shape)}")
        if camera.ndim != 3 or tuple(camera.shape) != (latent.shape[0], 1, config.cam_channels):
            raise ValueError(
                f"camera must have shape ({latent.shape[0]}, 1, {config.cam_channels}),"
                f" got {tuple(camera.shape)}"
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != latent.shape[0]:
            raise ValueError(
                f"timesteps must have shape ({latent.shape[0]},), got {tuple(timesteps.shape)}"
            )
        if (
            context.ndim != 3
            or context.shape[0] != latent.shape[0]
            or (context.shape[2] != config.cond_channels)
        ):
            raise ValueError(
                f"context must be [batch, rows, {config.cond_channels}], got {tuple(context.shape)}"
            )
        if reference_latent is None:
            return
        if (
            reference_latent.ndim != 4
            or reference_latent.shape[0] != latent.shape[0]
            or (reference_latent.shape[1] != config.cond2_channels)
        ):
            raise ValueError(
                f"reference latent must be [batch, {config.cond2_channels}, height, width],"
                f" got {tuple(reference_latent.shape)}"
            )
        if reference_latent.shape[2] * reference_latent.shape[3] > context.shape[1]:
            raise ValueError(
                "reference latent tokens must not exceed the context rows"
                f" ({reference_latent.shape[2] * reference_latent.shape[3]} >"
                f" {context.shape[1]})"
            )

    def forward(
        self,
        latent: torch.Tensor,
        camera: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        reference_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(latent, camera, timesteps, context, reference_latent)
        h_x = self.input_layer(latent)
        h_cond = self.cond_embedder(context)
        if reference_latent is not None:
            # Flatten the reference latent to tokens and front-pad to
            # the context length (the pad covers the DINOv3 class and
            # register prefix), then add through the second embedder.
            tokens = reference_latent.flatten(2).transpose(1, 2)
            tokens = F.pad(tokens, (0, 0, context.shape[1] - tokens.shape[1], 0))
            h_cond = h_cond + self.cond_embedder2(tokens.to(h_cond.dtype))
        t_emb = self.t_embedder(timesteps, dtype=h_x.dtype)
        t_mod = self.adaLN_modulation(t_emb)

        h_x = h_x + self.get_buffer("pos_emb").to(latent.device, latent.dtype)
        for rotary, block in zip(self.noise_repo_layers, self.noise_refiner, strict=True):
            h_x = block(h_x, rotary(h_x), mod=t_mod)
        for rotary, block in zip(self.context_repo_layers, self.context_refiner, strict=True):
            h_cond = block(h_cond, rotary(h_cond))

        h_cam = self.cam_refiner(camera.to(latent.dtype))
        h = torch.cat([h_x, h_cond, h_cam], dim=1)
        for rotary, block in zip(self.repo_layers, self.blocks, strict=True):
            h = block(h, rotary(h), mod=t_mod)

        h_x = F.layer_norm(h[:, : latent.shape[1]].float(), h.shape[-1:]).to(latent.dtype)
        h_cam = F.layer_norm(h[:, -camera.shape[1] :].float(), h.shape[-1:]).to(latent.dtype)

        binding = self._offloaded_residency()
        if binding is None:
            shift_table = cast_weight(self.shift_table, dtype=t_emb.dtype, device=t_emb.device)
        else:
            with binding.lease() as lease:
                shift_table = lease.get("shift_table", dtype=t_emb.dtype)
        shift, scale = (shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        scale = 1 + scale
        h_x = torch.addcmul(shift, h_x, scale)
        h_cam = torch.addcmul(shift, h_cam, scale)
        return self.out_layer(h_x), self.cam_out_layer(h_cam)


__all__ = [
    "TripoSplatAttention",
    "TripoSplatBlock",
    "TripoSplatMlp",
    "TripoSplatModel",
    "TripoSplatModelConfig",
    "TripoSplatMultiHeadRMSNorm",
    "TripoSplatRepoRotary",
    "TripoSplatTimestepEmbedder",
    "pcd_position_embedding",
    "triposplat_timestep_embedding",
]
