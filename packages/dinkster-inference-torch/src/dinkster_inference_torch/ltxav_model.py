"""Native LTX-2 audio-video diffusion transformer.

State-dict names match ComfyUI's LTXAVModel. LTX-2 19B consumes
text-side connector output; the 22B profiles own asymmetric gated
connectors in this model and consume dual-projection output. The
LTX-2.5 profile additionally omits video feed-forward biases and can
carry the learned keyframe marker.

Video latents are ``[B, C, T, H, W]`` with patch size 1; audio latents
are ``[B, 8, T, 16]`` mel patches whose tokens flatten channel by
frequency (``T`` may be zero, which skips every audio and cross-modal
leg). ``context`` concatenates the video and audio text lanes. RoPE is
the reference's split variant from float64 frequency indices (published
AV checkpoints pin ``rope_type: split`` and ``frequencies_precision:
float64``); cross-modal attention rotates queries and keys with separate
time-only tables. Timesteps may be one scalar per batch row or one value
per token, independently per stream; per-token video rows compress to
one row per frame exactly where the reference does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from dinkster_inference import (
    LTXAV_AUDIO_CHANNELS,
    LTXAV_AUDIO_FREQUENCY_BINS,
    LTXAV_AUDIO_HOP_LENGTH,
    LTXAV_AUDIO_LATENT_DOWNSAMPLE,
    LTXAV_AUDIO_MAX_POS,
    LTXAV_AUDIO_SAMPLE_RATE,
    LTXV_MAX_POS,
    LTXV_THETA,
    LTXV_TIMESTEP_MULTIPLIER,
    LTXV_VAE_SCALE_FACTORS,
    LTXAVConfig,
    LTXGeneratedKeyframes,
)

from .attention import AttentionKernel, select_attention
from .ltx_connector import LtxEmbeddingsConnector
from .ltx_model import (
    AdaLayerNormSingle,
    _additive_attention_mask,  # pyright: ignore[reportPrivateUsage]
    _apply_rope_torch,  # pyright: ignore[reportPrivateUsage]
    _CaptionProjection,  # pyright: ignore[reportPrivateUsage]
    _FeedForward,  # pyright: ignore[reportPrivateUsage]
    _rms,  # pyright: ignore[reportPrivateUsage]
    _rms_adaln,  # pyright: ignore[reportPrivateUsage]
    _rope_matrix,  # pyright: ignore[reportPrivateUsage]
)
from .operations import INITLESS, Operations, ResidencyRouted

_DEFAULT_ATTENTION = select_attention("flux").kernel


def pack_av_latents(
    latents: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """The reference's single-tensor wire format for multi-stream latents
    (comfy/utils.py pack_latents): every stream flattens to ``[B, 1, -1]``
    and streams concatenate on the last axis."""
    shapes = [tuple(latent.shape) for latent in latents]
    packed = torch.cat([latent.reshape(latent.shape[0], 1, -1) for latent in latents], dim=-1)
    return packed, shapes


def unpack_av_latents(packed: torch.Tensor, shapes: Sequence[Sequence[int]]) -> list[torch.Tensor]:
    """Inverse of :func:`pack_av_latents` (comfy/utils.py unpack_latents):
    a single recorded shape passes the packed tensor through unchanged."""
    if len(shapes) <= 1:
        return [packed]
    streams: list[torch.Tensor] = []
    for shape in shapes:
        cut = math.prod(shape[1:])
        streams.append(packed[:, :, :cut].reshape([packed.shape[0], *shape[1:]]))
        packed = packed[:, :, cut:]
    return streams


class _CompressedTimestep:
    """Per-frame compressed modulation rows (the reference's
    CompressedTimestep): when every patch of a frame shares one timestep,
    only the first row per frame is stored, and rows re-expand across the
    frame's patches after the learned-table add."""

    __slots__ = ("batch_size", "data", "feature_dim", "num_frames", "patches_per_frame")

    def __init__(
        self, rows: torch.Tensor, patches_per_frame: int | None, *, per_frame: bool = False
    ) -> None:
        self.batch_size, count, self.feature_dim = rows.shape
        if per_frame:
            if patches_per_frame is None:
                raise ValueError("per-frame timestep rows need patches_per_frame")
            self.patches_per_frame = patches_per_frame
            self.num_frames = count
            self.data = rows
        elif (
            patches_per_frame is not None
            and count >= patches_per_frame
            and count % patches_per_frame == 0
        ):
            self.patches_per_frame = patches_per_frame
            self.num_frames = count // patches_per_frame
            self.data = rows.view(
                self.batch_size, self.num_frames, patches_per_frame, self.feature_dim
            )[:, :, 0, :].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = count
            self.data = rows

    def expand(self) -> torch.Tensor:
        if self.patches_per_frame == 1:
            return self.data
        expanded = self.data.unsqueeze(2).expand(
            self.batch_size, self.num_frames, self.patches_per_frame, self.feature_dim
        )
        return expanded.reshape(self.batch_size, -1, self.feature_dim)

    def rows(self, table: torch.Tensor, indices: slice) -> tuple[torch.Tensor, ...]:
        """The selected table rows added to the stored (compressed) rows,
        expanded back to per-token width (the reference's
        expand_for_computation)."""
        count = table.shape[0]
        values = table[indices][None, None].to(device=self.data.device, dtype=self.data.dtype)
        if self.patches_per_frame == 1:
            per_token = self.data.reshape(self.batch_size, self.data.shape[1], count, -1)
            return (values + per_token[:, :, indices]).unbind(dim=2)
        per_frame = self.data.reshape(self.batch_size, self.num_frames, count, -1)
        return tuple(
            value.unsqueeze(2)
            .expand(self.batch_size, self.num_frames, self.patches_per_frame, value.shape[-1])
            .reshape(self.batch_size, -1, value.shape[-1])
            for value in (values + per_frame[:, :, indices]).unbind(dim=2)
        )


def _ada_values(
    table: torch.Tensor,
    batch: int,
    timestep: torch.Tensor | _CompressedTimestep,
    indices: slice = slice(None),
) -> tuple[torch.Tensor, ...]:
    """Modulation rows: learned table rows plus the adaLN-single rows for
    the selected slice, one tuple entry per row."""
    if isinstance(timestep, _CompressedTimestep):
        return timestep.rows(table, indices)
    values = table[indices][None, None].to(device=timestep.device, dtype=timestep.dtype)
    per_token = timestep.reshape(batch, timestep.shape[1], table.shape[0], -1)
    return (values + per_token[:, :, indices]).unbind(dim=2)


def _apply_split_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: tuple[torch.Tensor, bool],
    k_rope: tuple[torch.Tensor, bool] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix, _ = rope
    if torch.is_grad_enabled():
        return (
            _apply_rope_torch(q, matrix, True),
            _apply_rope_torch(k, matrix if k_rope is None else k_rope[0], True),
        )
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    q_shape = q.shape
    k_shape = k.shape
    q = q.reshape(q.shape[0], q.shape[1], matrix.shape[2], -1)
    k_matrix = matrix if k_rope is None else k_rope[0]
    k = k.reshape(k.shape[0], k.shape[1], k_matrix.shape[2], -1)
    if k_rope is None and q.shape == k.shape:
        q, k = comfy_kitchen.apply_rope_split_half(q, k, matrix)
    else:
        q = comfy_kitchen.apply_rope_split_half1(q, matrix)
        k = comfy_kitchen.apply_rope_split_half1(k, k_matrix)
    return q.reshape(q_shape), k.reshape(k_shape)


class LTXAVAttention(torch.nn.Module):
    """Self-, text-cross-, or modality-cross-attention with whole-width
    RMS q/k norms and split RoPE. Cross-modal attention rotates keys with
    their own table (``k_rope``); the reference applies the rotation on
    the flattened width before the heads split, which is the same values
    in a different layout."""

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
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, bool] | None = None,
        k_rope: tuple[torch.Tensor, bool] | None = None,
        value_passthrough: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context
        v = self.to_v(context)
        batch, q_tokens = x.shape[0], x.shape[1]
        if value_passthrough:
            merged = v
        else:
            q = self.q_norm(self.to_q(x))
            k = self.k_norm(self.to_k(context))
            if rope is not None:
                q, k = _apply_split_rope_qk(q, k, rope, k_rope)
            q = q.view(batch, q_tokens, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            attended = self._attention_kernel(q, k, v, mask=mask)
            merged = attended.transpose(1, 2).reshape(batch, q_tokens, self.heads * self.head_dim)
        if self.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            merged = (
                merged.view(batch, q_tokens, self.heads, self.head_dim) * gates.unsqueeze(-1)
            ).view(batch, q_tokens, self.heads * self.head_dim)
        return self.to_out(merged)


def _freq_indices(dim: int, axes: int, device: torch.device) -> torch.Tensor:
    """The reference's float64 geometric frequency ladder
    (generate_freq_grid_np), cast to float32."""
    count = dim // (2 * axes)
    powered = LTXV_THETA ** torch.linspace(0.0, 1.0, count, dtype=torch.float64, device=device)
    return (powered * math.pi / 2).to(torch.float32)


def _split_rope(
    coords: torch.Tensor,
    dim: int,
    heads: int,
    max_pos: Sequence[float],
    use_middle: bool,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Per-head split-RoPE matrix ``[B, T, H, dim / (2 * H), 2, 2]``
    from start/end coordinate pairs ``[B, axes, T, 2]``: per-axis
    geometric frequencies over positions mapped to [-1, 1], front-padded
    with the identity rotation up to half the width."""
    axes = coords.shape[1]
    grid = (coords[..., 0] + coords[..., 1]) / 2.0 if use_middle else coords[..., 0]
    indices = _freq_indices(dim, axes, coords.device)
    fractional = torch.stack([grid[:, i] / max_pos[i] for i in range(axes)], dim=-1)
    freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    pad = dim // 2 - freqs.shape[-1]
    return _rope_matrix(freqs, pad, True, heads, out_dtype), True


def _video_coordinates(
    batch: int,
    frames: int,
    height: int,
    width: int,
    causal_fix: bool,
    device: torch.device,
) -> torch.Tensor:
    """Start/end pixel coordinates of every latent token,
    ``[B, 3, T, 2]`` int64, with the causal first-frame temporal fix when
    configured."""
    grid = torch.meshgrid(
        torch.arange(frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    start = torch.stack(grid, dim=0).flatten(1)
    coords = torch.stack((start, start + 1), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    coords = coords * torch.tensor(LTXV_VAE_SCALE_FACTORS, device=device).view(1, 3, 1, 1)
    if causal_fix:
        coords[:, 0] = (coords[:, 0] + 1 - LTXV_VAE_SCALE_FACTORS[0]).clamp(min=0)
    return coords


def _audio_times(start: int, end: int, device: torch.device) -> torch.Tensor:
    """Second offsets of audio latent frames (the reference
    AudioPatchifier's causal mel-frame alignment at hop-length
    granularity)."""
    mel = torch.arange(start, end, dtype=torch.float32, device=device)
    mel = mel * LTXAV_AUDIO_LATENT_DOWNSAMPLE
    mel = (mel + 1 - LTXAV_AUDIO_LATENT_DOWNSAMPLE).clip(min=0)
    return mel * LTXAV_AUDIO_HOP_LENGTH / LTXAV_AUDIO_SAMPLE_RATE


def _audio_coordinates(batch: int, length: int, device: torch.device) -> torch.Tensor:
    """Start/end times in seconds of every audio token, ``[B, 1, T, 2]``
    float32."""
    start = _audio_times(0, length, device).unsqueeze(0).expand(batch, -1).unsqueeze(1)
    end = _audio_times(1, length + 1, device).unsqueeze(0).expand(batch, -1).unsqueeze(1)
    return torch.stack((start, end), dim=-1)


def _prompt_timestep(
    adaln: AdaLayerNormSingle | None,
    scaled: torch.Tensor,
    batch: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Shift/scale rows for adaLN-modulated text cross-attention, driven
    by the per-batch-row maximum timestep (the reference's
    compute_prompt_timestep)."""
    if adaln is None:
        return None
    flat = (
        scaled.max(dim=1, keepdim=True).values.flatten() if scaled.dim() > 1 else scaled.flatten()
    )
    rows, _ = adaln(flat, dtype)
    return rows.view(batch, 1, rows.shape[-1])


def _text_cross_attention(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: LTXAVAttention,
    table: torch.Tensor,
    prompt_table: torch.Tensor | None,
    timestep: torch.Tensor | _CompressedTimestep,
    prompt_timestep: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """One stream's text cross-attention: plain (ungated) when the model
    has no prompt tables, otherwise query and key/value modulation plus a
    gate from the stream table's last three rows (the reference's
    apply_cross_attention_adaln)."""
    if prompt_table is None:
        return attn(_rms(x), context=context, mask=mask)
    assert prompt_timestep is not None
    shift_q, scale_q, gate = _ada_values(table, x.shape[0], timestep, slice(6, 9))
    rows = prompt_table[None, None].to(device=x.device, dtype=x.dtype) + prompt_timestep.reshape(
        x.shape[0], prompt_timestep.shape[1], 2, -1
    )
    shift_kv, scale_kv = rows.unbind(dim=2)
    modulated = _rms_adaln(x, scale_q, shift_q)
    return attn(modulated, context=context * (1 + scale_kv) + shift_kv, mask=mask) * gate


class LTXAVTransformerBlock(ResidencyRouted, torch.nn.Module):
    """Paired video and audio self-attention, text cross-attention,
    cross-modal attention, and MLPs, each modulated by adaLN-single rows
    added to per-block learned tables. Every audio and cross-modal leg
    skips when the audio stream is empty."""

    prompt_scale_shift_table: torch.nn.Parameter | None
    audio_prompt_scale_shift_table: torch.nn.Parameter | None

    def __init__(
        self,
        config: LTXAVConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        audio = config.audio_hidden_size
        self.attn1 = LTXAVAttention(
            hidden,
            hidden,
            config.num_attention_heads,
            config.attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.attn2 = LTXAVAttention(
            hidden,
            config.cross_attention_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.audio_attn1 = LTXAVAttention(
            audio,
            audio,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.audio_attn2 = LTXAVAttention(
            audio,
            config.audio_cross_attention_dim,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.audio_to_video_attn = LTXAVAttention(
            hidden,
            audio,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.video_to_audio_attn = LTXAVAttention(
            audio,
            hidden,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            operations=operations,
            attention_kernel=attention_kernel,
            gated=config.gated_attention,
        )
        self.ff = _FeedForward(
            hidden,
            config.ffn_dim,
            operations=operations,
            bias=config.ff_bias,
        )
        self.audio_ff = _FeedForward(
            audio,
            config.audio_ffn_dim,
            operations=operations,
            bias=config.audio_ff_bias,
        )
        rows = config.adaln_rows
        self.scale_shift_table = torch.nn.Parameter(torch.empty(rows, hidden))
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(rows, audio))
        if config.cross_attention_adaln:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, hidden))
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio))
        else:
            self.register_parameter("prompt_scale_shift_table", None)
            self.register_parameter("audio_prompt_scale_shift_table", None)
        self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio))
        self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, hidden))

    def forward(
        self,
        vx: torch.Tensor,
        ax: torch.Tensor,
        v_context: torch.Tensor,
        a_context: torch.Tensor,
        attention_mask: torch.Tensor | None,
        v_timestep: _CompressedTimestep,
        a_timestep: torch.Tensor,
        v_rope: tuple[torch.Tensor, bool],
        a_rope: tuple[torch.Tensor, bool],
        v_cross_rope: tuple[torch.Tensor, bool],
        a_cross_rope: tuple[torch.Tensor, bool],
        v_cross_scale_shift: _CompressedTimestep,
        a_cross_scale_shift: torch.Tensor,
        v_cross_gate: _CompressedTimestep,
        a_cross_gate: torch.Tensor,
        v_prompt_timestep: torch.Tensor | None,
        a_prompt_timestep: torch.Tensor | None,
        self_attention_passthrough: bool = False,
        a2v_cross_attention: bool = True,
        v2a_cross_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = vx.shape[0]
        run_audio = ax.numel() > 0

        shift, scale = _ada_values(self.scale_shift_table, batch, v_timestep, slice(0, 2))
        attended = self.attn1(
            _rms_adaln(vx, scale, shift),
            rope=v_rope,
            value_passthrough=self_attention_passthrough,
        )
        gate = _ada_values(self.scale_shift_table, batch, v_timestep, slice(2, 3))[0]
        vx = torch.addcmul(vx, attended, gate)
        vx = vx + _text_cross_attention(
            vx,
            v_context,
            self.attn2,
            self.scale_shift_table,
            self.prompt_scale_shift_table,
            v_timestep,
            v_prompt_timestep,
            attention_mask,
        )

        if run_audio:
            shift, scale = _ada_values(self.audio_scale_shift_table, batch, a_timestep, slice(0, 2))
            attended = self.audio_attn1(
                _rms(ax) * (1 + scale) + shift,
                rope=a_rope,
                value_passthrough=self_attention_passthrough,
            )
            gate = _ada_values(self.audio_scale_shift_table, batch, a_timestep, slice(2, 3))[0]
            ax = torch.addcmul(ax, attended, gate)
            ax = ax + _text_cross_attention(
                ax,
                a_context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                self.audio_prompt_scale_shift_table,
                a_timestep,
                a_prompt_timestep,
                attention_mask,
            )

            ax_norm = _rms(ax) if a2v_cross_attention or v2a_cross_attention else None

            if a2v_cross_attention:
                assert ax_norm is not None
                a_scale, a_shift = _ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4], batch, a_cross_scale_shift
                )[:2]
                v_scale, v_shift = _ada_values(
                    self.scale_shift_table_a2v_ca_video[:4], batch, v_cross_scale_shift
                )[:2]
                attended = self.audio_to_video_attn(
                    _rms_adaln(vx, v_scale, v_shift),
                    context=ax_norm * (1 + a_scale) + a_shift,
                    rope=v_cross_rope,
                    k_rope=a_cross_rope,
                )
                gate = _ada_values(self.scale_shift_table_a2v_ca_video[4:], batch, v_cross_gate)[0]
                vx = torch.addcmul(vx, attended, gate)

            if v2a_cross_attention:
                assert ax_norm is not None
                a_scale, a_shift = _ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4], batch, a_cross_scale_shift
                )[2:4]
                v_scale, v_shift = _ada_values(
                    self.scale_shift_table_a2v_ca_video[:4], batch, v_cross_scale_shift
                )[2:4]
                attended = self.video_to_audio_attn(
                    ax_norm * (1 + a_scale) + a_shift,
                    context=_rms_adaln(vx, v_scale, v_shift),
                    rope=a_cross_rope,
                    k_rope=v_cross_rope,
                )
                gate = _ada_values(self.scale_shift_table_a2v_ca_audio[4:], batch, a_cross_gate)[0]
                ax = torch.addcmul(ax, attended, gate)

        shift, scale = _ada_values(self.scale_shift_table, batch, v_timestep, slice(3, 5))
        projected = self.ff(_rms_adaln(vx, scale, shift))
        gate = _ada_values(self.scale_shift_table, batch, v_timestep, slice(5, 6))[0]
        vx = torch.addcmul(vx, projected, gate)

        if run_audio:
            shift, scale = _ada_values(self.audio_scale_shift_table, batch, a_timestep, slice(3, 5))
            projected = self.audio_ff(_rms(ax) * (1 + scale) + shift)
            gate = _ada_values(self.audio_scale_shift_table, batch, a_timestep, slice(5, 6))[0]
            ax = torch.addcmul(ax, projected, gate)

        return vx, ax


class LTXAVModel(ResidencyRouted, torch.nn.Module):
    """The LTX-2 audio-video diffusion transformer."""

    keyframes_abs_pos_embedding: torch.nn.Parameter | None
    prompt_adaln_single: AdaLayerNormSingle | None
    audio_prompt_adaln_single: AdaLayerNormSingle | None

    def __init__(
        self,
        config: LTXAVConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        audio = config.audio_hidden_size
        rows = config.adaln_rows
        self.patchify_proj = operations.linear(config.in_channels, hidden)
        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = torch.nn.Parameter(torch.empty(1, hidden))
        else:
            self.register_parameter("keyframes_abs_pos_embedding", None)
        self.adaln_single = AdaLayerNormSingle(hidden, rows, operations=operations)
        if config.cross_attention_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(hidden, 2, operations=operations)
            self.audio_prompt_adaln_single = AdaLayerNormSingle(audio, 2, operations=operations)
        else:
            self.prompt_adaln_single = None
            self.audio_prompt_adaln_single = None
        if config.caption_proj_before_connector:
            assert config.video_connector is not None and config.audio_connector is not None
            self.caption_projection = None
            self.audio_caption_projection = None
            self.video_embeddings_connector = LtxEmbeddingsConnector(
                config.video_connector,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            self.audio_embeddings_connector = LtxEmbeddingsConnector(
                config.audio_connector,
                operations=operations,
                attention_kernel=attention_kernel,
            )
        else:
            self.caption_projection = _CaptionProjection(
                config.caption_channels, hidden, operations=operations
            )
            self.audio_caption_projection = _CaptionProjection(
                config.caption_channels, audio, operations=operations
            )
            self.video_embeddings_connector = None
            self.audio_embeddings_connector = None
        self.audio_patchify_proj = operations.linear(config.audio_in_channels, audio)
        self.audio_adaln_single = AdaLayerNormSingle(audio, rows, operations=operations)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            hidden, 4, operations=operations
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(hidden, 1, operations=operations)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            audio, 4, operations=operations
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(audio, 1, operations=operations)
        self.transformer_blocks = torch.nn.ModuleList(
            LTXAVTransformerBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_layers)
        )
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, hidden))
        self.norm_out = operations.layer_norm(hidden, eps=1e-6, elementwise_affine=False)
        self.proj_out = operations.linear(hidden, config.in_channels)
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio))
        self.audio_norm_out = operations.layer_norm(audio, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = operations.linear(audio, config.audio_in_channels)

    def preprocess_text_embeds(self, context: torch.Tensor) -> torch.Tensor:
        """Apply the profile's text projections and connector towers."""
        config = self.config
        expected = (
            config.cross_attention_dim + config.audio_cross_attention_dim
            if config.caption_proj_before_connector
            else 2 * config.caption_channels
        )
        if context.shape[-1] != expected:
            if config.caption_proj_before_connector:
                raise ValueError(
                    "LTXAV context must carry the video and audio dual projection"
                    f" concatenated ({expected} wide); got {context.shape[-1]}"
                )
            raise ValueError(
                "LTXAV context must carry the video and audio text projections"
                f" concatenated ({expected} wide); got {context.shape[-1]}."
                " Unprocessed text embeddings (the reference's deprecated"
                " embeddings-connector path) are not supported."
            )
        if config.caption_proj_before_connector:
            assert self.video_embeddings_connector is not None
            assert self.audio_embeddings_connector is not None
            video, audio = torch.split(
                context,
                [config.cross_attention_dim, config.audio_cross_attention_dim],
                dim=-1,
            )
            video = self.video_embeddings_connector(video)
            audio = self.audio_embeddings_connector(audio)
        else:
            assert self.caption_projection is not None
            assert self.audio_caption_projection is not None
            video, audio = torch.split(
                context,
                [config.caption_channels, config.caption_channels],
                dim=-1,
            )
            video = self.caption_projection(video)
            audio = self.audio_caption_projection(audio)
        return torch.cat((video, audio), dim=-1)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        frame_rate: float = 25.0,
        denoise_mask: torch.Tensor | None = None,
        ref_audio_tokens: torch.Tensor | None = None,
        stg_self_attn_blocks: frozenset[int] = frozenset(),
        a2v_cross_attention: bool = True,
        v2a_cross_attention: bool = True,
        generated_keyframes: LTXGeneratedKeyframes | None = None,
        context_preprocessed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        batch, _, frames, height, width = video.shape
        audio_length = audio.shape[2]

        has_spatial_mask = False
        if denoise_mask is not None:
            for frame in range(denoise_mask.shape[2]):
                frame_masks = denoise_mask[:, 0, frame].flatten(1)
                if bool((frame_masks.amin(dim=1) != frame_masks.amax(dim=1)).any()):
                    has_spatial_mask = True
                    break

        tokens = self.patchify_proj(video.flatten(2).transpose(1, 2))
        audio_tokens = audio.permute(0, 2, 1, 3).reshape(
            batch, audio_length, config.audio_in_channels
        )
        a_coords = _audio_coordinates(batch, audio_length, video.device)

        ref_length = 0
        if ref_audio_tokens is not None:
            ref_tokens = ref_audio_tokens.to(dtype=audio_tokens.dtype, device=audio_tokens.device)
            if ref_tokens.shape[0] < batch:
                ref_tokens = ref_tokens.expand(batch, -1, -1)
            ref_length = ref_tokens.shape[1]
            time_per_latent = (
                LTXAV_AUDIO_HOP_LENGTH * LTXAV_AUDIO_LATENT_DOWNSAMPLE / LTXAV_AUDIO_SAMPLE_RATE
            )
            ref_start = _audio_times(0, ref_length, video.device)
            ref_end = _audio_times(1, ref_length + 1, video.device)
            offset = ref_end[-1].item() + time_per_latent
            ref_start = (ref_start - offset).unsqueeze(0).expand(batch, -1).unsqueeze(1)
            ref_end = (ref_end - offset).unsqueeze(0).expand(batch, -1).unsqueeze(1)
            ref_coords = torch.stack((ref_start, ref_end), dim=-1)
            audio_tokens = torch.cat((ref_tokens, audio_tokens), dim=1)
            a_coords = torch.cat((ref_coords.to(a_coords), a_coords), dim=2)
        audio_tokens = self.audio_patchify_proj(audio_tokens)

        v_patches_per_frame = None if has_spatial_mask else height * width
        scaled = timestep * LTXV_TIMESTEP_MULTIPLIER
        per_frame_path = (
            v_patches_per_frame is not None
            and (timestep.numel() // batch) % v_patches_per_frame == 0
        )
        if per_frame_path:
            assert v_patches_per_frame is not None
            ts_input = (
                timestep.reshape(batch, -1, v_patches_per_frame)[:, :, 0] * LTXV_TIMESTEP_MULTIPLIER
            )
        else:
            ts_input = scaled
        v_rows, v_embedded = self.adaln_single(ts_input.flatten(), video.dtype)
        v_timestep = _CompressedTimestep(
            v_rows.view(batch, -1, v_rows.shape[-1]), v_patches_per_frame, per_frame=per_frame_path
        )
        v_embedded_timestep = _CompressedTimestep(
            v_embedded.view(batch, -1, v_embedded.shape[-1]),
            v_patches_per_frame,
            per_frame=per_frame_path,
        )
        v_prompt_timestep = _prompt_timestep(self.prompt_adaln_single, scaled, batch, video.dtype)

        a_timestep_input = audio_timestep
        if ref_length > 0:
            if a_timestep_input.dim() <= 1:
                a_timestep_input = a_timestep_input.view(-1, 1).expand(batch, audio_length)
            ref_zeros = torch.zeros(
                batch,
                ref_length,
                *a_timestep_input.shape[2:],
                device=a_timestep_input.device,
                dtype=a_timestep_input.dtype,
            )
            a_timestep_input = torch.cat((ref_zeros, a_timestep_input), dim=1)
        a_scaled = a_timestep_input * LTXV_TIMESTEP_MULTIPLIER
        a_flat = a_scaled.flatten()
        v_flat = scaled.flatten()
        gate_factor = config.av_ca_timestep_scale_multiplier / LTXV_TIMESTEP_MULTIPLIER

        a_cross_rows, _ = self.av_ca_audio_scale_shift_adaln_single(a_flat, video.dtype)
        v_cross_rows, _ = self.av_ca_video_scale_shift_adaln_single(v_flat, video.dtype)
        a2v_gate_rows, _ = self.av_ca_a2v_gate_adaln_single(
            a_scaled.max().expand_as(v_flat) * gate_factor, video.dtype
        )
        v2a_gate_rows, _ = self.av_ca_v2a_gate_adaln_single(
            scaled.max().expand_as(a_flat) * gate_factor, video.dtype
        )
        a_cross_scale_shift = a_cross_rows.view(batch, -1, a_cross_rows.shape[-1])
        v_cross_scale_shift = _CompressedTimestep(
            v_cross_rows.view(batch, -1, v_cross_rows.shape[-1]), v_patches_per_frame
        )
        v_cross_gate = _CompressedTimestep(
            a2v_gate_rows.view(batch, -1, a2v_gate_rows.shape[-1]), v_patches_per_frame
        )
        a_cross_gate = v2a_gate_rows.view(batch, -1, v2a_gate_rows.shape[-1])

        a_rows, a_embedded = self.audio_adaln_single(a_flat, video.dtype)
        a_timestep_rows = a_rows.view(batch, -1, a_rows.shape[-1])
        a_embedded_timestep = a_embedded.view(batch, -1, a_embedded.shape[-1])
        a_prompt_timestep = _prompt_timestep(
            self.audio_prompt_adaln_single, a_scaled, batch, video.dtype
        )

        prepared_context = context if context_preprocessed else self.preprocess_text_embeds(context)
        v_context, a_context = torch.split(
            prepared_context,
            [config.cross_attention_dim, config.audio_cross_attention_dim],
            dim=-1,
        )
        mask = _additive_attention_mask(attention_mask, video.dtype)

        v_coords = _video_coordinates(
            batch, frames, height, width, config.causal_temporal_positioning, video.device
        )
        if self.keyframes_abs_pos_embedding is not None:
            first_frame = v_coords[:, 0, :, 0] == 0
            if generated_keyframes is not None:
                if type(generated_keyframes) is not LTXGeneratedKeyframes:
                    raise TypeError("generated_keyframes must be exact LTXGeneratedKeyframes")
                tokens_per_frame = height * width
                if generated_keyframes.tokens_per_frame != tokens_per_frame:
                    raise ValueError(
                        "LTX generated keyframes were recorded at"
                        f" {generated_keyframes.tokens_per_frame} tokens per latent frame but"
                        f" this latent has {tokens_per_frame}"
                    )
                first_token = generated_keyframes.first_latent_frame * tokens_per_frame
                num_tokens = generated_keyframes.num_keyframes * tokens_per_frame
                slots = torch.zeros(
                    frames * tokens_per_frame, dtype=torch.bool, device=first_frame.device
                )
                slots[first_token : first_token + num_tokens] = True
                first_frame = first_frame | slots
            embedding = self.keyframes_abs_pos_embedding.to(
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = tokens + first_frame.unsqueeze(-1).to(tokens.dtype) * embedding
        v_fractional = v_coords.to(torch.float32)
        v_fractional[:, 0] = v_fractional[:, 0] * (1.0 / frame_rate)
        v_rope = _split_rope(
            v_fractional,
            config.hidden_size,
            config.num_attention_heads,
            LTXV_MAX_POS,
            config.use_middle_indices_grid,
            video.dtype,
        )
        a_rope = _split_rope(
            a_coords,
            config.audio_hidden_size,
            config.audio_num_attention_heads,
            LTXAV_AUDIO_MAX_POS,
            config.use_middle_indices_grid,
            video.dtype,
        )
        cross_max_pos = (max(LTXV_MAX_POS[0], LTXAV_AUDIO_MAX_POS[0]),)
        v_cross_fractional = v_coords.to(torch.float32)
        v_cross_fractional[:, 0] = v_cross_fractional[:, 0] * (1.0 / frame_rate)
        v_cross_rope = _split_rope(
            v_cross_fractional[:, 0:1],
            config.audio_cross_attention_dim,
            config.audio_num_attention_heads,
            cross_max_pos,
            True,
            video.dtype,
        )
        a_cross_rope = _split_rope(
            a_coords[:, 0:1],
            config.audio_cross_attention_dim,
            config.audio_num_attention_heads,
            cross_max_pos,
            True,
            video.dtype,
        )

        vx, ax = tokens, audio_tokens
        for index, block in enumerate(self.transformer_blocks):
            vx, ax = block(
                vx,
                ax,
                v_context,
                a_context,
                mask,
                v_timestep,
                a_timestep_rows,
                v_rope,
                a_rope,
                v_cross_rope,
                a_cross_rope,
                v_cross_scale_shift,
                a_cross_scale_shift,
                v_cross_gate,
                a_cross_gate,
                v_prompt_timestep,
                a_prompt_timestep,
                self_attention_passthrough=index in stg_self_attn_blocks,
                a2v_cross_attention=a2v_cross_attention,
                v2a_cross_attention=v2a_cross_attention,
            )

        if ref_length > 0:
            ax = ax[:, ref_length:]
            if a_embedded_timestep.shape[1] > 1:
                a_embedded_timestep = a_embedded_timestep[:, ref_length:]

        v_embedded_rows = v_embedded_timestep.expand()
        table = self.scale_shift_table[None, None].to(device=vx.device, dtype=vx.dtype)
        rows = table + v_embedded_rows[:, :, None]
        shift, scale = rows[:, :, 0], rows[:, :, 1]
        vx = self.norm_out(vx) * (1 + scale) + shift
        vx = self.proj_out(vx)
        video_out = vx.transpose(1, 2).reshape(batch, -1, frames, height, width)

        a_table = self.audio_scale_shift_table[None, None].to(device=ax.device, dtype=ax.dtype)
        a_rows_out = a_table + a_embedded_timestep[:, :, None]
        a_shift, a_scale = a_rows_out[:, :, 0], a_rows_out[:, :, 1]
        ax = self.audio_norm_out(ax) * (1 + a_scale) + a_shift
        ax = self.audio_proj_out(ax)
        audio_out = ax.view(
            batch, audio_length, LTXAV_AUDIO_CHANNELS, LTXAV_AUDIO_FREQUENCY_BINS
        ).permute(0, 2, 1, 3)
        return video_out, audio_out


__all__ = [
    "LTXAVAttention",
    "LTXAVModel",
    "LTXAVTransformerBlock",
    "pack_av_latents",
    "unpack_av_latents",
]
