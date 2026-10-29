"""Native LTX-Video 2B diffusion transformer.

Faithful standalone port of ``comfy/ldm/lightricks/model.py`` (LTXVModel)
at the current ComfyUI pin b78cec87. State-dict names match the reference.
Guide/keyframe token handling follows the current pin.

Inputs and outputs use ``[B, C, T, H, W]`` with patch size 1: tokens are
row-major over time, height, then width and each token is one 128-channel
latent voxel. ``context`` contains T5-XXL rows. RoPE is the reference's
interleaved variant applied over the full flattened hidden width before
the heads split, from float32 frequency indices (the published 2B configs
carry neither ``rope_type`` nor ``frequencies_precision``, so the
reference takes the interleaved/float32 defaults). Timesteps may be one
scalar per batch row or one value per token (the reference's per-token
denoise path).
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    LTXV_MAX_POS,
    LTXV_THETA,
    LTXV_TIME_PROJ_CHANNELS,
    LTXV_TIMESTEP_MULTIPLIER,
    LTXV_VAE_SCALE_FACTORS,
    LTXVConfig,
)

from .attention import AttentionKernel, select_attention
from .ltx_media import LTXVGuideConditioning
from .operations import INITLESS, Operations, ResidencyRouted
from .quant_linear import linear_input_act

_DEFAULT_ATTENTION = select_attention("flux").kernel


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, channels: int) -> torch.Tensor:
    """The reference's ``get_timestep_embedding`` with
    ``flip_sin_to_cos=True`` and ``downscale_freq_shift=0``: cos rows
    first, float32 throughout."""
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    emb = torch.exp(exponent / half)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)


def _rms(x: torch.Tensor) -> torch.Tensor:
    """The reference's weightless ``comfy.ldm.common_dit.rms_norm``."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class _TimestepEmbedding(torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.linear_1 = operations.linear(LTXV_TIME_PROJ_CHANNELS, hidden_size)
        self.linear_2 = operations.linear(hidden_size, hidden_size)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class _CombinedTimestepEmbedding(torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.timestep_embedder = _TimestepEmbedding(hidden_size, operations=operations)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        projected = _sinusoidal_timestep_embedding(timestep, LTXV_TIME_PROJ_CHANNELS)
        return self.timestep_embedder(projected.to(hidden_dtype))


class AdaLayerNormSingle(torch.nn.Module):
    """PixArt-Alpha adaLN-single: one embedded timestep expanded to
    ``coefficient`` modulation rows by a single linear."""

    def __init__(self, hidden_size: int, coefficient: int, *, operations: Operations) -> None:
        super().__init__()
        self.emb = _CombinedTimestepEmbedding(hidden_size, operations=operations)
        self.linear = operations.linear(hidden_size, coefficient * hidden_size)

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.emb(timestep, hidden_dtype)
        return self.linear(F.silu(embedded)), embedded


class _CaptionProjection(torch.nn.Module):
    def __init__(self, caption_channels: int, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.linear_1 = operations.linear(caption_channels, hidden_size)
        self.linear_2 = operations.linear(hidden_size, hidden_size)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(caption), approximate="tanh"))


class _GELUProjection(torch.nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, *, operations: Operations, bias: bool = True
    ) -> None:
        super().__init__()
        self.proj = operations.linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class _FeedForward(torch.nn.Module):
    """GELU-tanh MLP at the reference's ``ff.net.0.proj`` / ``ff.net.2``
    key positions (slot 1 is the reference's parameterless Dropout)."""

    def __init__(
        self, dim: int, inner_dim: int, *, operations: Operations, bias: bool = True
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            _GELUProjection(dim, inner_dim, operations=operations, bias=bias),
            torch.nn.Identity(),
            operations.linear(inner_dim, dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = cast(_GELUProjection, self.net[0]).proj
        return linear_input_act(self.net[2], projection(x), "gelu_tanh")


def _rope_matrix(
    frequencies: torch.Tensor,
    pad_size: int,
    split: bool,
    heads: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """The reference's per-head ``[cos, -sin; sin, cos]`` RoPE table."""
    cos = frequencies.cos().to(out_dtype)
    sin = frequencies.sin().to(out_dtype)
    if pad_size:
        matrix_pad = pad_size if split else pad_size // 2
        cos = torch.cat((torch.ones_like(cos[:, :, :matrix_pad]), cos), dim=-1)
        sin = torch.cat((torch.zeros_like(sin[:, :, :matrix_pad]), sin), dim=-1)
    batch, tokens, half_width = cos.shape
    cos = cos.reshape(batch, tokens, heads, half_width // heads)
    sin = sin.reshape(batch, tokens, heads, half_width // heads)
    matrix = torch.stack((cos, -sin, sin, cos), dim=-1)
    return matrix.reshape(*matrix.shape[:-1], 2, 2)


def _apply_rope_torch(x: torch.Tensor, matrix: torch.Tensor, split: bool) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1], matrix.shape[2], -1)
    if split:
        pairs = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    else:
        pairs = x.reshape(*x.shape[:-1], -1, 1, 2)
    pairs = pairs.to(matrix.dtype)
    output = matrix[..., 0] * pairs[..., 0] + matrix[..., 1] * pairs[..., 1]
    if split:
        output = output.movedim(-1, -2)
    return output.reshape(original_shape).type_as(x)


def _apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: tuple[torch.Tensor, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix, split = rope
    if torch.is_grad_enabled():
        return _apply_rope_torch(q, matrix, split), _apply_rope_torch(k, matrix, split)
    q_shape = q.shape
    k_shape = k.shape
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    q = q.reshape(q.shape[0], q.shape[1], matrix.shape[2], -1)
    k = k.reshape(k.shape[0], k.shape[1], matrix.shape[2], -1)
    if split:
        q, k = dinkster_kitchen.apply_rope_split_half(q, k, matrix)
    else:
        q, k = dinkster_kitchen.apply_rope(q, k, matrix)
    return q.reshape(q_shape), k.reshape(k_shape)


def _rms_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        return _rms(x) * (1 + scale) + shift
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    return dinkster_kitchen.rms_adaln(x, scale, shift)


class _GuideAttentionMask:
    """Partitioned self-attention masks for generated and guide tokens."""

    def __init__(
        self,
        total_tokens: int,
        guide_start: int,
        tracked_count: int,
        tracked_weights: torch.Tensor,
    ) -> None:
        finfo = torch.finfo(tracked_weights.dtype)
        positive = tracked_weights > 0
        log_weights = torch.full_like(tracked_weights, finfo.min)
        log_weights[positive] = torch.log(tracked_weights[positive].clamp(min=finfo.tiny))
        self.guide_start = guide_start
        self.tracked_count = tracked_count
        self.generated = torch.zeros(
            (1, 1, 1, total_tokens),
            device=tracked_weights.device,
            dtype=tracked_weights.dtype,
        )
        self.generated[:, :, :, guide_start : guide_start + tracked_count] = log_weights.reshape(
            1, 1, 1, -1
        )
        self.guides = torch.zeros(
            (1, 1, tracked_count, total_tokens),
            device=tracked_weights.device,
            dtype=tracked_weights.dtype,
        )
        self.guides[:, :, :, :guide_start] = log_weights.reshape(1, 1, -1, 1)


class LTXAttention(torch.nn.Module):
    """Self- or cross-attention with whole-width RMS q/k norms. RoPE, when
    given, applies to the full flattened hidden width (heads split after),
    exactly as the reference orders it."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        head_dim: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
        gated: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.q_norm = operations.rms_norm(inner_dim, eps=1e-5)
        self.k_norm = operations.rms_norm(inner_dim, eps=1e-5)
        self.to_q = operations.linear(query_dim, inner_dim)
        self.to_k = operations.linear(context_dim, inner_dim)
        self.to_v = operations.linear(context_dim, inner_dim)
        self.to_gate_logits = operations.linear(query_dim, heads) if gated else None
        self.to_out = torch.nn.Sequential(
            operations.linear(inner_dim, query_dim), torch.nn.Identity()
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | _GuideAttentionMask | None = None,
        rope: tuple[torch.Tensor, bool] | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(context))
        v = self.to_v(context)
        if rope is not None:
            q, k = _apply_rope_qk(q, k, rope)
        batch, q_tokens = q.shape[0], q.shape[1]
        q = q.view(batch, q_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        if isinstance(mask, _GuideAttentionMask):
            start = mask.guide_start
            end = start + mask.tracked_count
            attended = torch.empty_like(q)
            if start > 0:
                attended[:, :, :start] = self._attention_kernel(
                    q[:, :, :start], k, v, mask=mask.generated
                )
            attended[:, :, start:end] = self._attention_kernel(
                q[:, :, start:end], k, v, mask=mask.guides
            )
            if end < q_tokens:
                attended[:, :, end:] = self._attention_kernel(q[:, :, end:], k, v)
        else:
            attended = self._attention_kernel(q, k, v, mask=mask)
        merged = attended.transpose(1, 2).reshape(batch, q_tokens, self.heads * self.head_dim)
        if self.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            merged = (
                merged.view(batch, q_tokens, self.heads, self.head_dim) * gates.unsqueeze(-1)
            ).view(batch, q_tokens, self.heads * self.head_dim)
        return self.to_out(merged)


class LTXTransformerBlock(ResidencyRouted, torch.nn.Module):
    """Self-attention, cross-attention, and MLP under six adaLN-single
    modulation rows added to a per-block learned table."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        context_dim: int,
        ffn_dim: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.attn1 = LTXAttention(
            dim,
            dim,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.attn2 = LTXAttention(
            dim,
            context_dim,
            heads,
            head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.ff = _FeedForward(dim, ffn_dim, operations=operations)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(6, dim))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        rope: tuple[torch.Tensor, bool],
        self_attention_mask: _GuideAttentionMask | None = None,
    ) -> torch.Tensor:
        with self.materialized_state("scale_shift_table", device=x.device, dtype=x.dtype) as table:
            rows = table[None, None] + timestep.reshape(x.shape[0], timestep.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = rows.unbind(dim=2)
        x = (
            x
            + self.attn1(
                _rms_adaln(x, scale_msa, shift_msa),
                mask=self_attention_mask,
                rope=rope,
            )
            * gate_msa
        )
        x = x + self.attn2(x, context=context, mask=attention_mask)
        y = _rms(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        return x.addcmul_(self.ff(y), gate_mlp)


def _pixel_coordinates(
    batch: int,
    frames: int,
    height: int,
    width: int,
    causal_fix: bool,
    device: torch.device,
) -> torch.Tensor:
    """Top-left pixel coordinates of every latent token, ``[B, 3, T]``,
    with the causal first-frame temporal fix when configured."""
    grid = torch.meshgrid(
        torch.arange(frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack(grid, dim=0).flatten(1).unsqueeze(0).repeat(batch, 1, 1)
    coords = coords * torch.tensor(LTXV_VAE_SCALE_FACTORS, device=device).view(1, 3, 1)
    if causal_fix:
        coords[:, 0] = (coords[:, 0] + 1 - LTXV_VAE_SCALE_FACTORS[0]).clamp(min=0)
    return coords


def _interleaved_rope(
    fractional_coords: torch.Tensor, dim: int, heads: int, out_dtype: torch.dtype
) -> tuple[torch.Tensor, bool]:
    """Interleaved RoPE matrix from fractional coordinates ``[B, 3, T]``."""
    axes = fractional_coords.shape[1]
    n_elem = 2 * axes
    indices = LTXV_THETA ** torch.linspace(
        0.0,
        1.0,
        dim // n_elem,
        device=fractional_coords.device,
        dtype=torch.float32,
    )
    indices = indices * math.pi / 2
    fractional = torch.stack(
        [fractional_coords[:, i] / LTXV_MAX_POS[i] for i in range(axes)], dim=-1
    )
    freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return _rope_matrix(freqs, dim % n_elem, False, heads, out_dtype), False


def _additive_attention_mask(
    attention_mask: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor | None:
    """Boolean/integer masks become the reference's large-negative additive
    bias; float masks pass through unchanged."""
    if attention_mask is None or torch.is_floating_point(attention_mask):
        return attention_mask
    return (attention_mask - 1).to(dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(dtype).max


def _downsample_guide_mask(mask: torch.Tensor, latent_shape: tuple[int, int, int]) -> torch.Tensor:
    """Reference area-spatial and causal-temporal guide-mask reduction."""
    frames, height, width = latent_shape
    batch, _, pixel_frames, pixel_height, pixel_width = mask.shape
    spatial = F.interpolate(
        mask.permute(0, 2, 1, 3, 4).reshape(batch * pixel_frames, 1, pixel_height, pixel_width),
        size=(height, width),
        mode="area",
    )
    spatial = spatial.reshape(batch, pixel_frames, 1, height, width).permute(0, 2, 1, 3, 4)
    first = spatial[:, :, :1]
    if pixel_frames > 1 and frames > 1:
        remaining = frames - 1
        group = (pixel_frames - 1) // remaining
        if group < 1:
            rest = F.interpolate(
                spatial[:, :, 1:].permute(0, 3, 4, 1, 2).reshape(batch * height * width, 1, -1),
                size=remaining,
                mode="nearest",
            )
            rest = rest.reshape(batch, height, width, 1, remaining).permute(0, 3, 4, 1, 2)
        else:
            rest = spatial[:, :, 1 : 1 + remaining * group]
            rest = rest.reshape(batch, 1, remaining, group, height, width).mean(dim=3)
        reduced = torch.cat((first, rest), dim=2)
    elif frames > 1:
        reduced = first.expand(-1, -1, frames, -1, -1)
    else:
        reduced = first
    return reduced.flatten(1)


def _guide_attention_mask(
    guides: tuple[LTXVGuideConditioning, ...],
    *,
    total_tokens: int,
    guide_grid_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> _GuideAttentionMask | None:
    if not guides or all(
        guide.strength == 1.0 and guide.attention_mask is None for guide in guides
    ):
        return None
    weights: list[torch.Tensor] = []
    offset = 0
    for guide in guides:
        selected = guide_grid_mask[offset : offset + guide.pre_filter_count]
        offset += guide.pre_filter_count
        if not bool(selected.any()):
            continue
        if guide.attention_mask is None:
            current = torch.full(
                (1, guide.pre_filter_count), guide.strength, device=device, dtype=dtype
            )
        else:
            current = _downsample_guide_mask(
                guide.attention_mask.to(device=device, dtype=dtype), guide.latent_shape
            )
            if current.shape[0] > 1:
                if any(not torch.equal(current[0], row) for row in current[1:]):
                    raise ValueError("LTX-Video guide attention masks must match across the batch")
                current = current[:1]
            current = current * guide.strength
        weights.append(current[:, selected])
    if not weights:
        return None
    tracked = torch.cat(weights, dim=1)
    if bool((tracked == 1.0).all()):
        return None
    tracked_count = tracked.shape[1]
    return _GuideAttentionMask(
        total_tokens,
        total_tokens - tracked_count,
        tracked_count,
        tracked,
    )


class LTXVModel(ResidencyRouted, torch.nn.Module):
    """The LTX-Video 2B diffusion transformer."""

    def __init__(
        self,
        config: LTXVConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.patchify_proj = operations.linear(config.in_channels, hidden)
        self.adaln_single = AdaLayerNormSingle(hidden, 6, operations=operations)
        self.caption_projection = _CaptionProjection(
            config.caption_channels, hidden, operations=operations
        )
        self.transformer_blocks = torch.nn.ModuleList(
            LTXTransformerBlock(
                hidden,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                config.ffn_dim,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.num_layers)
        )
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, hidden))
        self.norm_out = operations.layer_norm(hidden, eps=1e-6, elementwise_affine=False)
        self.proj_out = operations.linear(hidden, config.in_channels)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        frame_rate: float = 25.0,
        denoise_mask: torch.Tensor | None = None,
        guides: tuple[LTXVGuideConditioning, ...] = (),
    ) -> torch.Tensor:
        if type(guides) is not tuple or any(
            type(guide) is not LTXVGuideConditioning for guide in guides
        ):
            raise TypeError("LTX-Video guides must be exact LTXVGuideConditioning values")
        batch, _, frames, height, width = x.shape
        raw_tokens = x.flatten(2).transpose(1, 2)
        coords = _pixel_coordinates(
            batch,
            frames,
            height,
            width,
            self.config.causal_temporal_positioning,
            x.device,
        )
        grid_mask: torch.Tensor | None = None
        guide_grid_mask = torch.empty(0, device=x.device, dtype=torch.bool)
        if guides:
            guide_tokens = sum(guide.pre_filter_count for guide in guides)
            if guide_tokens >= raw_tokens.shape[1]:
                raise ValueError("LTX-Video guides must follow a nonempty generated latent span")
            if denoise_mask is None:
                raise ValueError("LTX-Video guides require a denoise mask")
            if (
                denoise_mask.ndim != 5
                or denoise_mask.shape[0] not in (1, batch)
                or denoise_mask.shape[1] < 1
                or tuple(denoise_mask.shape[2:]) != (frames, height, width)
            ):
                raise ValueError("LTX-Video denoise mask must match [B,channels,T,H,W]")
            token_masks = ~(denoise_mask < 0).any(dim=1).flatten(1)
            if token_masks.shape[0] > 1 and any(
                not torch.equal(token_masks[0], row) for row in token_masks[1:]
            ):
                raise ValueError("LTX-Video guide grid masks must match across the batch")
            grid_mask = token_masks[0]
            guide_grid_mask = grid_mask[-guide_tokens:]
            keyframes = torch.cat(tuple(guide.keyframe_indices for guide in guides), dim=2).to(
                device=x.device
            )
            if keyframes.shape[0] == 1 and batch > 1:
                keyframes = keyframes.expand(batch, -1, -1, -1)
            if keyframes.shape[0] != batch:
                raise ValueError("LTX-Video guide coordinate batch must match the latent batch")
            keyframes = keyframes[:, :, guide_grid_mask]
            raw_tokens = raw_tokens[:, grid_mask]
            coords = coords[:, :, grid_mask]
            if keyframes.shape[2] > 0:
                coords[:, :, -keyframes.shape[2] :] = keyframes[..., 0]
            if timestep.ndim > 1:
                timestep = timestep[:, grid_mask]

        tokens = self.patchify_proj(raw_tokens)
        fractional = coords.to(torch.float32)
        fractional[:, 0] = fractional[:, 0] * (1.0 / frame_rate)
        rope = _interleaved_rope(
            fractional,
            self.config.hidden_size,
            self.config.num_attention_heads,
            x.dtype,
        )

        scaled = timestep * LTXV_TIMESTEP_MULTIPLIER
        modulation, embedded = self.adaln_single(scaled.flatten(), x.dtype)
        modulation = modulation.view(batch, -1, modulation.shape[-1])
        embedded = embedded.view(batch, -1, embedded.shape[-1])

        projected_context = self.caption_projection(context).view(batch, -1, tokens.shape[-1])
        mask = _additive_attention_mask(attention_mask, x.dtype)
        self_attention_mask = _guide_attention_mask(
            guides,
            total_tokens=tokens.shape[1],
            guide_grid_mask=guide_grid_mask,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.transformer_blocks:
            tokens = block(
                tokens,
                projected_context,
                mask,
                modulation,
                rope,
                self_attention_mask,
            )

        with self.materialized_state(
            "scale_shift_table", device=tokens.device, dtype=tokens.dtype
        ) as table:
            rows = table[None, None] + embedded[:, :, None]
        shift, scale = rows[:, :, 0], rows[:, :, 1]
        tokens = self.norm_out(tokens) * (1 + scale) + shift
        tokens = self.proj_out(tokens)
        if grid_mask is not None:
            full = torch.zeros(
                (batch, frames * height * width, tokens.shape[2]),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            full[:, grid_mask] = tokens
            tokens = full
        return tokens.transpose(1, 2).reshape(batch, -1, frames, height, width)


__all__ = [
    "AdaLayerNormSingle",
    "LTXAttention",
    "LTXTransformerBlock",
    "LTXVModel",
]
