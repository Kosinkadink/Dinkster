"""Direct-import MiniMax H3 packed audio-video diffusion transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.metadata import version as _distribution_version
from typing import Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    MINIMAX_H3_SIGMAS,
    MINIMAX_H3_VIDEO_TEMPORAL_MAPPING,
    ComponentPlan,
    LatentStream,
    MiniMaxH3Config,
    MiniMaxH3GuideTokenGeometry,
    MiniMaxH3Sigmas,
    MiniMaxH3VideoLatentGeometry,
    MultiStreamLatent,
    TimelineGuide,
    validate_minimax_h3_guide_timeline,
)
from dinkster_inference.minimax_h3_dit import MiniMaxH3TimeEmbeddingKind

from .attention import (
    BUILTIN_SDPA_PROVIDER,
    COMFY_KITCHEN_INT8_PROVIDER,
    SAGE2_PROVIDER,
    SOL_ATTENTION_PROVIDER,
    AttentionKernel,
    AttentionSelection,
    AttentionTensorLease,
    QkvConsumingAttentionKernel,
    attention_provider_identity,
    bind_packed_attention_kernel,
    exclude_packed_attention_modifiers,
    sage2_distribution_version,
)
from .minimax_h3_attention import (
    MiniMaxH3AttentionKernelFactory,
    MiniMaxH3PackedSegmentKind,
    MiniMaxH3PackedSequenceFacts,
    MiniMaxH3SequenceSharding,
)
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import (
    INITLESS,
    Operations,
    ResidencyRouted,
    materialized_linear_parameters,
    materialized_rms_norm_weight,
)
from .quant_linear import Int8Linear, linear_input_act

_FRAME_PER_TOKEN = MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_frames_per_latent
_FRAME_RESCALE = MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(1)
_VISUAL_CONDITION_TIMESTEP = 0.999
_AUDIO_CONDITION_TIMESTEP = 1.0
_MLP_TIME_EMBED_DIM = 2688
_CURVE_TABLE_ROWS = 1025
_CURVE_TIME_EMBED_DIM = 8

_ModulationRow = int | torch.Tensor
_ModulationSegment = tuple[int, int, _ModulationRow]


def _h3_latent(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))


def _h3_stream_sigmas(
    video_sigma: float,
    sigmas: MiniMaxH3Sigmas,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = (
        (torch.tensor(video_sigma, dtype=torch.float32, device=device) * 1000.0) / 1000.0
    ).clamp(min=1e-6)
    base = sigma / (sigmas.video.shift + sigma * (1.0 - sigmas.video.shift))
    audio = sigmas.audio_shift * base / (1.0 + (sigmas.audio_shift - 1.0) * base)
    return sigma, audio


@dataclass(frozen=True, slots=True)
class MiniMaxH3AttentionProviderEvidence:
    """Exact immutable evidence for the selected H3 attention provider."""

    provider: str
    torch_version: str
    provider_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.provider) is not str:
            raise TypeError("provider must be a string")
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if type(self.torch_version) is not str:
            raise TypeError("torch_version must be a string")
        running_version = str(torch.__version__)
        if not self.torch_version or self.torch_version != running_version:
            raise ValueError("torch_version must equal the non-empty running torch version")
        if self.provider == BUILTIN_SDPA_PROVIDER:
            if self.provider_version is not None:
                raise ValueError("the built-in SDPA provider carries no provider version")
        elif self.provider in (COMFY_KITCHEN_INT8_PROVIDER, SOL_ATTENTION_PROVIDER):
            installed = _distribution_version("dinkster-kitchen")
            if self.provider_version != installed:
                raise ValueError(
                    "provider_version must equal the installed "
                    f"dinkster-kitchen version {installed!r}"
                )
        elif self.provider == SAGE2_PROVIDER:
            installed = sage2_distribution_version()
            if installed is None:
                raise ValueError(
                    "SageAttention 2 provider evidence requires an installed SageAttention"
                    " distribution"
                )
            if self.provider_version != installed:
                raise ValueError(
                    f"provider_version must equal the installed SageAttention version {installed!r}"
                )
        elif self.provider_version is not None and (
            type(self.provider_version) is not str or not self.provider_version
        ):
            raise ValueError("provider_version must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class MiniMaxH3AttentionGeometry:
    """Construction geometry for production and reduced CPU proofs."""

    hidden_width: int
    heads: int
    head_dim: int
    rotary_dim: int
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        dimensions = (self.hidden_width, self.heads, self.head_dim, self.rotary_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("attention dimensions must be positive integers")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 6 != 0:
            raise ValueError("rotary_dim must fit the head and divide into three paired axes")
        if type(self.norm_eps) is not float or self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be a positive float")

    @property
    def inner_width(self) -> int:
        return self.heads * self.head_dim


MINIMAX_H3_ATTENTION_GEOMETRY = MiniMaxH3AttentionGeometry(
    hidden_width=MINIMAX_H3_CONFIG.hidden_width,
    heads=MINIMAX_H3_CONFIG.attention_heads,
    head_dim=MINIMAX_H3_CONFIG.attention_head_dim,
    rotary_dim=96,
)


def _apply_split_half_rope(
    value: torch.Tensor,
    table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    prefix = value[..., :rotary_dim]
    pairs = prefix.reshape(*prefix.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
    pairs = pairs.to(table.dtype)
    rotated = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
    rotated = rotated.movedim(-1, -2).reshape_as(prefix).type_as(value)
    return torch.cat((rotated, value[..., rotary_dim:]), dim=-1)


def _fused_h3_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    table: torch.Tensor,
    query_norm: torch.nn.RMSNorm,
    key_norm: torch.nn.RMSNorm,
    epsilon: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        return (
            _apply_split_half_rope(query_norm(query), table, rotary_dim),
            _apply_split_half_rope(key_norm(key), table, rotary_dim),
        )
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    with (
        materialized_rms_norm_weight(query_norm) as query_weight,
        materialized_rms_norm_weight(key_norm) as key_weight,
    ):
        return dinkster_kitchen.rms_rope_split_half_(
            query,
            key,
            table,
            query_weight.detach(),
            key_weight.detach(),
            epsilon=epsilon,
            rot_dim=rotary_dim,
        )


class MiniMaxH3Attention(torch.nn.Module):
    """One H3 full-attention projection with an injected opaque kernel."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        geometry: MiniMaxH3AttentionGeometry,
        attention_kernel: AttentionKernel,
        provider_evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        if type(geometry) is not MiniMaxH3AttentionGeometry:
            raise TypeError("geometry must be exact MiniMaxH3AttentionGeometry")
        if not isinstance(cast("object", attention_kernel), AttentionKernel):
            raise TypeError("attention_kernel must implement AttentionKernel")
        if type(provider_evidence) is not MiniMaxH3AttentionProviderEvidence:
            raise TypeError("provider_evidence must be exact frozen MiniMaxH3 evidence")
        self.geometry = geometry
        self.provider_evidence = provider_evidence
        self.qkv_proj = operations.linear(
            geometry.hidden_width,
            3 * geometry.inner_width,
            bias=False,
        )
        self.q_norm = operations.rms_norm(geometry.head_dim, eps=geometry.norm_eps)
        self.k_norm = operations.rms_norm(geometry.head_dim, eps=geometry.norm_eps)
        self.out_proj = operations.linear(
            geometry.inner_width,
            geometry.hidden_width,
            bias=False,
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    @property
    def attention_kernel(self) -> AttentionKernel:
        return self._attention_kernel

    def forward(
        self,
        hidden: torch.Tensor,
        rope_table: torch.Tensor | None = None,
        *,
        attention_kernel: AttentionKernel | None = None,
    ) -> torch.Tensor:
        geometry = self.geometry
        if not isinstance(cast("object", hidden), torch.Tensor) or hidden.ndim != 3:
            raise ValueError("hidden must be a rank-3 tensor [1, S, hidden]")
        if hidden.shape[0] != 1 or hidden.shape[2] != geometry.hidden_width:
            raise ValueError("hidden must have exact batch-1 H3 geometry")
        if not hidden.is_floating_point():
            raise ValueError("hidden must use a floating-point dtype")
        if rope_table is not None:
            expected_table = (1, hidden.shape[1], 1, geometry.rotary_dim // 2, 2, 2)
            if tuple(rope_table.shape) != expected_table:
                raise ValueError(f"rope_table must have shape {expected_table}")
            if not rope_table.is_floating_point() or rope_table.device != hidden.device:
                raise ValueError("rope_table must be floating-point on the hidden device")

        sequence = hidden.shape[1]
        query, key, value = self.qkv_proj(hidden).chunk(3, dim=-1)
        head_shape = (1, sequence, geometry.heads, geometry.head_dim)
        query = query.view(head_shape)
        key = key.view(head_shape)
        value = value.view(head_shape)
        # In the fused inference path the in-place norm+rope keeps q, k, and v
        # as views of the single fused projection buffer; every other path
        # rebuilds q and k as new tensors while v stays a projection view.
        qkv_share_projection = rope_table is not None and not torch.is_grad_enabled()
        if rope_table is not None:
            query, key = _fused_h3_norm_rope(
                query,
                key,
                rope_table,
                self.q_norm,
                self.k_norm,
                geometry.norm_eps,
                geometry.rotary_dim,
            )
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)
        kernel = self._attention_kernel if attention_kernel is None else attention_kernel
        if isinstance(kernel, QkvConsumingAttentionKernel):
            # A consuming kernel releases q/k/v before allocating its output.
            # When q/k/v share the projection buffer, consuming all three
            # frees it; otherwise v alone would pin the full 3x-width buffer,
            # so hand over an unaliased copy (the audited ComfyUI V clone).
            if not qkv_share_projection:
                value = value.clone()
            query_lease = AttentionTensorLease(query.transpose(1, 2))
            key_lease = AttentionTensorLease(key.transpose(1, 2))
            value_lease = AttentionTensorLease(value.transpose(1, 2))
            del query, key, value
            output = kernel.consume(
                query_lease,
                key_lease,
                value_lease,
                mask=None,
                causal=False,
                scale=None,
                enable_gqa=False,
            )
        else:
            output = kernel(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                mask=None,
                causal=False,
                scale=None,
                enable_gqa=False,
            )
        output = output.transpose(1, 2).reshape(1, sequence, geometry.inner_width)
        return self.out_proj(output)


def minimax_h3_attention_provider(
    selection: AttentionSelection,
) -> tuple[AttentionKernel, MiniMaxH3AttentionProviderEvidence]:
    """Turn one generic attention selection into an H3 kernel/evidence pair."""
    if type(selection) is not AttentionSelection:
        raise TypeError("attention selection must be an exact AttentionSelection")
    provider, provider_version = attention_provider_identity(selection.status)
    evidence = MiniMaxH3AttentionProviderEvidence(
        provider, str(torch.__version__), provider_version
    )
    return selection.kernel, evidence


def assemble_minimax_h3_attention(
    *, operations: Operations, attention_selection: AttentionSelection
) -> MiniMaxH3Attention:
    """Build the exact production primitive from a centrally selected provider."""
    kernel, evidence = minimax_h3_attention_provider(attention_selection)
    return MiniMaxH3Attention(
        MINIMAX_H3_ATTENTION_GEOMETRY,
        kernel,
        evidence,
        operations=operations,
    )


@dataclass(frozen=True, slots=True)
class MiniMaxH3KeyframeLatent:
    """One realized first- or last-frame video condition."""

    resolved_frame_index: int
    video: torch.Tensor


class MiniMaxH3ReferenceKind(StrEnum):
    """The exact realized reference block kinds understood by the DiT."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class MiniMaxH3ReferenceLatents:
    """One ordered realized reference block."""

    kind: MiniMaxH3ReferenceKind
    video: torch.Tensor | None = None
    audio: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MiniMaxH3DiTConditioning:
    """Realized tensor inputs needed to assemble one packed H3 sequence."""

    text_token_tags: torch.Tensor | None = None
    keyframes: tuple[MiniMaxH3KeyframeLatent, ...] = ()
    references: tuple[MiniMaxH3ReferenceLatents, ...] = ()
    frame_count: int | None = None
    visual_noise_timestep: float = _VISUAL_CONDITION_TIMESTEP
    audio_noise_timestep: float = _AUDIO_CONDITION_TIMESTEP
    seed: int = 0
    guides: tuple[TimelineGuide[torch.Tensor], ...] = ()


def _patchify_video(latent: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    batch, channels, full_t, full_h, full_w = latent.shape
    patch_t, patch_h, patch_w = patch
    temporal, height, width = full_t // patch_t, full_h // patch_h, full_w // patch_w
    latent = latent.reshape(batch, channels, temporal, patch_t, height, patch_h, width, patch_w)
    latent = torch.einsum("nctrhpwq->nthwcrpq", latent)
    return latent.reshape(batch * temporal * height * width, -1)


def _unpatchify_video(
    rows: torch.Tensor,
    temporal: int,
    height: int,
    width: int,
    channels: int,
    patch: tuple[int, int, int],
) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch
    value = rows.reshape(-1, temporal, height, width, channels, patch_t, patch_h, patch_w)
    value = torch.einsum("nthwcrpq->nctrhpwq", value)
    return value.reshape(
        -1,
        channels,
        temporal * patch_t,
        height * patch_h,
        width * patch_w,
    )


def _pack_audio(latent: torch.Tensor) -> torch.Tensor:
    _, channels, stereo, temporal = latent.shape
    return latent[0].permute(1, 2, 0).reshape(stereo * temporal, channels)


def _unpack_audio(rows: torch.Tensor) -> torch.Tensor:
    temporal = rows.shape[0] // 2
    return rows.reshape(2, temporal, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _pad_video(latent: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    pad_t = (-latent.shape[2]) % patch[0]
    pad_h = (-latent.shape[3]) % patch[1]
    pad_w = (-latent.shape[4]) % patch[2]
    return F.pad(latent, (0, pad_w, 0, pad_h, 0, pad_t), mode="circular")


def _axis_from_sqrt_area(dimension: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dimension / sqrt_area
    count = dimension // patch
    return (torch.arange(count, dtype=torch.float64) * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    area = math.sqrt(height * width)
    height_axis = _axis_from_sqrt_area(height, 2, area)
    width_axis = _axis_from_sqrt_area(width, 2, area)
    hh, ww = torch.meshgrid(height_axis, width_axis, indexing="ij")
    return torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1), width_axis


def _video_t_spans(count: int) -> tuple[float, ...]:
    return tuple(_FRAME_RESCALE * _FRAME_PER_TOKEN[index % 5] for index in range(count))


def _video_t_grid(count: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(_video_t_spans(count), dtype=torch.float64)
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def _audio_grid(cursor: float, temporal: int, low: float, high: float) -> torch.Tensor:
    grid = torch.zeros(temporal * 2, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(temporal, dtype=torch.float64)).repeat(2)
    grid[:temporal, 2] = low
    grid[temporal:, 2] = high
    return grid


def _video_grid(temporal: int, frame: torch.Tensor, cursor: float) -> torch.Tensor:
    grid = torch.empty(temporal, frame.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = _video_t_grid(temporal, cursor)[:, None]
    grid[:, :, 1:] = frame[None]
    return grid.reshape(-1, 3)


def _reference_t_span(reference: MiniMaxH3ReferenceLatents) -> float:
    if reference.kind is MiniMaxH3ReferenceKind.IMAGE:
        return 1.0
    audio_span = float(0 if reference.audio is None else reference.audio.shape[-1])
    if reference.video is None:
        return audio_span
    return max(audio_span, sum(_video_t_spans(reference.video.shape[2])))


class _PackedLayout:
    def __init__(
        self,
        text_length: int,
        target_video: torch.Tensor,
        target_audio: torch.Tensor,
        conditioning: MiniMaxH3DiTConditioning,
    ) -> None:
        temporal, height, width = target_video.shape[2:]
        audio_temporal = target_audio.shape[-1]
        frame, width_grid = _frame_grid(height, width)
        frame_rows = frame.shape[0]
        segments: list[tuple[str, int]] = [("text", text_length)]
        text_grid = torch.zeros(text_length, 3, dtype=torch.float64)
        text_grid[:, 0] = torch.arange(text_length, dtype=torch.float64)
        positions = [text_grid]
        video_updates: list[torch.Tensor] = []
        audio_updates: list[torch.Tensor] = []
        target_audio_width = (float(width_grid[0]), float(width_grid[-1]))
        target_cursor = float(text_length)
        for reference in conditioning.references:
            target_cursor += _reference_t_span(reference)

        for keyframe in conditioning.keyframes:
            condition_t = target_cursor + _FRAME_RESCALE * keyframe.resolved_frame_index
            grid = torch.empty(frame_rows, 3, dtype=torch.float64)
            grid[:, 0] = condition_t
            grid[:, 1:] = frame
            segments.append(("condition", frame_rows))
            positions.append(grid)
            video_updates.append(torch.zeros(frame_rows, dtype=torch.bool))
        for guide in conditioning.guides:
            condition_t = target_cursor + MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(
                guide.frame_index
            )
            if "video" in guide.latent.roles:
                guide_video = guide.latent.by_role("video")
                count = guide_video.shape[2] * frame_rows
                segments.append(("condition", count))
                positions.append(_video_grid(guide_video.shape[2], frame, condition_t))
                video_updates.append(torch.zeros(count, dtype=torch.bool))
            if "audio" in guide.latent.roles:
                guide_audio = guide.latent.by_role("audio")
                count = guide_audio.shape[-1] * 2
                segments.append(("condition_audio", count))
                positions.append(
                    _audio_grid(condition_t, guide_audio.shape[-1], *target_audio_width)
                )
                audio_updates.append(torch.zeros(count, dtype=torch.bool))

        cursor = float(text_length)
        for reference in conditioning.references:
            if reference.audio is not None:
                reference_audio_t = reference.audio.shape[-1]
                if reference.video is None:
                    audio_width = target_audio_width
                else:
                    ref_frame, ref_width_grid = _frame_grid(
                        reference.video.shape[3], reference.video.shape[4]
                    )
                    audio_width = (float(ref_width_grid[0]), float(ref_width_grid[-1]))
                segments.append(("reference_audio", reference_audio_t * 2))
                positions.append(_audio_grid(cursor, reference_audio_t, *audio_width))
                audio_updates.append(torch.zeros(reference_audio_t * 2, dtype=torch.bool))
            if reference.video is not None:
                reference_t = reference.video.shape[2]
                ref_frame, _ = _frame_grid(reference.video.shape[3], reference.video.shape[4])
                count = reference_t * ref_frame.shape[0]
                segments.append(("reference_video", count))
                positions.append(_video_grid(reference_t, ref_frame, cursor))
                video_updates.append(torch.zeros(count, dtype=torch.bool))
            cursor += _reference_t_span(reference)

        segments.append(("audio", audio_temporal * 2))
        positions.append(_audio_grid(cursor, audio_temporal, *target_audio_width))
        audio_updates.append(torch.ones(audio_temporal * 2, dtype=torch.bool))
        segments.append(("video", temporal * frame_rows))
        positions.append(_video_grid(temporal, frame, cursor))
        video_updates.append(torch.ones(temporal * frame_rows, dtype=torch.bool))

        absolute: list[tuple[int, int, str]] = []
        offset = 0
        for kind, count in segments:
            absolute.append((offset, offset + count, kind))
            offset += count
        self.segments = tuple(absolute)
        self.sequence_length = offset
        self.position_ids = torch.cat(positions)
        self.video_update = torch.cat(video_updates)
        self.audio_update = torch.cat(audio_updates)


@dataclass(frozen=True, slots=True)
class MiniMaxH3ControlBlockContext:
    """Per-forward facts a control patch needs alongside every base block."""

    layout: _PackedLayout
    time_embedding: torch.Tensor
    segments: tuple[_ModulationSegment, ...]
    rope_table: torch.Tensor
    attention_kernel: AttentionKernel | None


@runtime_checkable
class MiniMaxH3ControlPatch(Protocol):
    """Block-level control application around the base H3 block loop."""

    def before_base_block(self, hidden: torch.Tensor, block_index: int) -> None: ...

    def after_base_block(
        self,
        hidden: torch.Tensor,
        block_index: int,
        context: MiniMaxH3ControlBlockContext,
    ) -> torch.Tensor: ...


class _MiniMaxH3MLP(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int, *, operations: Operations) -> None:
        super().__init__()
        self.fc1 = operations.linear(hidden, ffn * 2, bias=False)
        self.fc2 = operations.linear(ffn, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return linear_input_act(self.fc2, self.fc1(hidden), "swiglu")


class _MiniMaxH3AdaLN(torch.nn.Module):
    def __init__(
        self,
        time_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        *,
        apply_silu: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.linear(time_dim, expand * hidden * modalities)

    def forward(self, time: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.apply_silu:
            time = F.silu(time)
        value = self.linear(time).view(time.shape[0] * self.modalities, self.expand * self.hidden)
        return value.chunk(self.expand, dim=-1)


class _MiniMaxH3TimeEmbedder(torch.nn.Module):
    def __init__(self, hidden_width: int, *, operations: Operations) -> None:
        super().__init__()
        self.proj_in = operations.linear(256, hidden_width)
        self.proj_out = operations.linear(hidden_width, _MLP_TIME_EMBED_DIM)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=time.device) / half
        )
        angles = time.to(torch.float32)[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedding)))


def _modulate(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: tuple[_ModulationSegment, ...],
) -> torch.Tensor:
    # Autograd rejects in-place writes into slice views and needs the
    # pre-mutation values for backward; keep the in-place fast path for
    # inference only. Both paths run the same mul/add kernels.
    if torch.is_grad_enabled():
        result = hidden.clone()
        for start, stop, row in segments:
            result[:, start:stop] = hidden[:, start:stop] * (
                1.0 + scale[row].to(hidden.dtype)
            ) + shift[row].to(hidden.dtype)
        return result
    for start, stop, row in segments:
        hidden[:, start:stop].mul_(1.0 + scale[row].to(hidden.dtype)).add_(
            shift[row].to(hidden.dtype)
        )
    return hidden


def _gated_residual(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    update: torch.Tensor,
    segments: tuple[_ModulationSegment, ...],
) -> torch.Tensor:
    if torch.is_grad_enabled():
        result = hidden.clone()
        for start, stop, row in segments:
            result[:, start:stop] = torch.addcmul(
                hidden[:, start:stop], update[:, start:stop], gate[row].to(hidden.dtype)
            )
        return result
    for start, stop, row in segments:
        hidden[:, start:stop].addcmul_(update[:, start:stop], gate[row].to(hidden.dtype))
    return hidden


def _translate_modulation_segments(
    segments: tuple[_ModulationSegment, ...], start: int, stop: int
) -> tuple[_ModulationSegment, ...]:
    translated: list[_ModulationSegment] = []
    for segment_start, segment_stop, row in segments:
        intersection_start = max(segment_start, start)
        intersection_stop = min(segment_stop, stop)
        if intersection_start >= intersection_stop:
            continue
        local_row = row
        if type(row) is torch.Tensor:
            offset = intersection_start - segment_start
            local_row = row[offset : offset + intersection_stop - intersection_start]
        translated.append((intersection_start - start, intersection_stop - start, local_row))
    return tuple(translated)


def _video_mask_row_values(
    mask: torch.Tensor,
    latent_shape: tuple[int, int, int],
    patch: tuple[int, int, int],
) -> torch.Tensor | None:
    temporal, height, width = latent_shape
    _, patch_h, patch_w = patch
    source = mask[0].to(torch.float32).amax(dim=0)
    source = F.pad(
        source,
        (0, width - source.shape[-1], 0, height - source.shape[-2]),
        mode="replicate",
    )
    values = (
        source.reshape(
            temporal,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        .amax(dim=(2, 4))
        .reshape(-1)
    )
    return None if bool((values >= 1.0 - 1e-3).all()) else values


def _audio_mask_row_values(mask: torch.Tensor) -> torch.Tensor | None:
    values = mask[0].to(torch.float32).amax(dim=0).reshape(-1)
    return None if bool((values >= 1.0 - 1e-3).all()) else values


def _row_time_indices(
    rows: torch.Tensor,
    time_row: dict[float, int],
) -> torch.Tensor:
    levels = rows.unique()
    base = torch.tensor(
        [time_row[float(value)] for value in levels.tolist()],
        dtype=torch.long,
        device=rows.device,
    )
    return base[torch.searchsorted(levels, rows)]


class _MiniMaxH3RefinerBlock(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Config,
        attention_kernel: AttentionKernel,
        evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations,
        rotary_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = operations.rms_norm(config.hidden_width, eps=1e-5)
        self.norm2 = operations.rms_norm(config.hidden_width, eps=1e-5)
        geometry = MiniMaxH3AttentionGeometry(
            config.hidden_width,
            config.attention_heads,
            config.attention_head_dim,
            rotary_dim,
        )
        self.attn = MiniMaxH3Attention(geometry, attention_kernel, evidence, operations=operations)
        self.mlp = _MiniMaxH3MLP(config.hidden_width, config.ffn_width, operations=operations)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            hidden = self.attn(self.norm1(hidden)) + hidden
            return self.mlp(self.norm2(hidden)) + hidden
        hidden = self.attn(self.norm1(hidden)).add_(hidden)
        return self.mlp(self.norm2(hidden)).add_(hidden)


class _MiniMaxH3TokenRefiner(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Config,
        attention_kernel: AttentionKernel,
        evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations,
        rotary_dim: int,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            _MiniMaxH3RefinerBlock(
                config,
                attention_kernel,
                evidence,
                operations=operations,
                rotary_dim=rotary_dim,
            )
            for _ in range(2)
        )
        self.final_norm = operations.rms_norm(config.hidden_width, eps=1e-5)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        queue = make_prefetch_queue(self.blocks)
        try:
            for block in self.blocks:
                prefetch_queue_pop(queue, block)
                hidden = block(hidden)
            prefetch_queue_pop(queue, None)
            return self.final_norm(hidden)
        finally:
            close_prefetch_queue(queue)


class MiniMaxH3BlockShapes(Protocol):
    """Block width facts shared by base and control checkpoint shapes."""

    @property
    def hidden_width(self) -> int: ...

    @property
    def attention_heads(self) -> int: ...

    @property
    def attention_head_dim(self) -> int: ...

    @property
    def ffn_width(self) -> int: ...


class _MiniMaxH3Block(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxH3BlockShapes,
        attention_kernel: AttentionKernel,
        evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations,
        fp32_operations: Operations,
        rotary_dim: int,
        time_dim: int,
        apply_silu: bool,
    ) -> None:
        super().__init__()
        self.norm1 = operations.rms_norm(config.hidden_width, eps=1e-5)
        self.norm2 = operations.rms_norm(config.hidden_width, eps=1e-5)
        geometry = MiniMaxH3AttentionGeometry(
            config.hidden_width,
            config.attention_heads,
            config.attention_head_dim,
            rotary_dim,
        )
        self.attn = MiniMaxH3Attention(geometry, attention_kernel, evidence, operations=operations)
        self.mlp = _MiniMaxH3MLP(config.hidden_width, config.ffn_width, operations=operations)
        self.adaln_proj = _MiniMaxH3AdaLN(
            time_dim,
            config.hidden_width,
            6,
            3,
            apply_silu=apply_silu,
            operations=fp32_operations,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        segments: tuple[tuple[int, int, int], ...],
        rope_table: torch.Tensor,
        *,
        attention_kernel: AttentionKernel | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(time)
        normalized = _modulate(self.norm1(hidden), shift_attn, scale_attn, segments)
        if attention_kernel is None:
            hidden = _gated_residual(hidden, gate_attn, self.attn(normalized, rope_table), segments)
        else:
            hidden = _gated_residual(
                hidden,
                gate_attn,
                self.attn(normalized, rope_table, attention_kernel=attention_kernel),
                segments,
            )
        normalized = _modulate(self.norm2(hidden), shift_mlp, scale_mlp, segments)
        return _gated_residual(hidden, gate_mlp, self.mlp(normalized), segments)


class _MiniMaxH3FinalLayer(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        time_dim: int,
        apply_silu: bool,
        norm_operations: Operations,
        adaln_operations: Operations,
        fp32_operations: Operations,
    ) -> None:
        super().__init__()
        patch_width = config.video_latent_channels * math.prod(config.patch)
        self.norm = norm_operations.rms_norm(config.hidden_width, eps=1e-5)
        self.adaln_proj = _MiniMaxH3AdaLN(
            time_dim,
            config.hidden_width,
            2,
            1,
            apply_silu=apply_silu,
            operations=adaln_operations,
        )
        self.video_out = fp32_operations.linear(config.hidden_width, patch_width)
        self.audio_out = fp32_operations.linear(config.hidden_width, config.audio_latent_channels)

    def forward(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        video_segment: _ModulationSegment,
        audio_segment: _ModulationSegment,
        video_sigma: float,
        sampler_sigmas: tuple[float, ...] | None,
        schedule_shifts: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(time)
        video_start, video_stop, video_row = video_segment
        audio_start, audio_stop, audio_row = audio_segment
        video = (
            self.norm(hidden[video_start:video_stop]) * (1.0 + scale[video_row]) + shift[video_row]
        ).float()
        audio = (
            self.norm(hidden[audio_start:audio_stop]) * (1.0 + scale[audio_row]) + shift[audio_row]
        ).float()
        with (
            materialized_linear_parameters(self.video_out) as video_parameters,
            materialized_linear_parameters(self.audio_out) as audio_parameters,
        ):
            video_weight, video_bias = video_parameters
            audio_weight, audio_bias = audio_parameters
            video_heads = video_weight.shape[0] // self.video_out.out_features
            audio_heads = audio_weight.shape[0] // self.audio_out.out_features
            if video_heads != audio_heads:
                raise ValueError("MiniMax H3 PDD video and audio head-bank sizes differ")
            if video_heads == 1:
                return (
                    F.linear(video, video_weight, video_bias),
                    F.linear(audio, audio_weight, audio_bias),
                )
            if not sampler_sigmas:
                raise ValueError("MiniMax H3 PDD heads require the sampler sigma schedule")
            index = min(
                range(len(sampler_sigmas)),
                key=lambda item: abs(sampler_sigmas[item] - video_sigma),
            )
            sigma_next = sampler_sigmas[min(index + 1, len(sampler_sigmas) - 1)]
            start, stop = (
                round(
                    (1.0 - sigma / (schedule_shifts[0] + sigma * (1.0 - schedule_shifts[0])))
                    * video_heads
                )
                for sigma in (video_sigma, sigma_next)
            )
            start = min(start, video_heads - 1)
            stop = max(stop, start + 1)
            return (
                _minimax_h3_pdd_head(
                    video,
                    video_weight,
                    video_bias,
                    self.video_out.out_features,
                    video_heads,
                    start,
                    stop,
                    schedule_shifts[0],
                ),
                _minimax_h3_pdd_head(
                    audio,
                    audio_weight,
                    audio_bias,
                    self.audio_out.out_features,
                    audio_heads,
                    start,
                    stop,
                    schedule_shifts[1],
                ),
            )


def _minimax_h3_pdd_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_width: int,
    heads: int,
    start: int,
    stop: int,
    schedule_shift: float,
) -> torch.Tensor:
    grid = torch.linspace(1.0, 0.0, heads + 1, dtype=torch.float64, device=hidden.device)
    shifted = 1.0 - schedule_shift * grid / (1.0 + (schedule_shift - 1.0) * grid)
    coefficients = shifted.diff()[start:stop]
    coefficients = (coefficients / coefficients.sum()).to(hidden)
    rows = weight.reshape(heads, output_width, weight.shape[1])
    selected_weight = rows[0]
    first_offset = max(start, 1)
    if first_offset < stop:
        selected_weight = selected_weight + torch.einsum(
            "n,noi->oi", coefficients[first_offset - start :], rows[first_offset:stop]
        )
    selected_bias = None
    if bias is not None:
        bias_rows = bias.reshape(heads, output_width)
        selected_bias = bias_rows[0]
        if first_offset < stop:
            selected_bias = selected_bias + torch.einsum(
                "n,no->o",
                coefficients[first_offset - start :],
                bias_rows[first_offset:stop],
            )
    return F.linear(hidden, selected_weight, selected_bias)


class _MiniMaxH3RoPE(ResidencyRouted, torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(width))

    def table(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        stored = cast(torch.Tensor, self.inv_freq)
        positions = position_ids.to(device=device, dtype=torch.float32)

        def build(inverse: torch.Tensor) -> torch.Tensor:
            per_axis = positions.unsqueeze(-1) * inverse.view(1, 1, -1)
            half = torch.cat(per_axis.unbind(dim=1), dim=-1)
            angles = torch.cat((half, half), dim=-1)
            rotary_dim = stored.numel() * 6
            cosine, sine = (
                torch.cos(angles[:, : rotary_dim // 2]),
                torch.sin(angles[:, : rotary_dim // 2]),
            )
            return (
                torch.stack((cosine, -sine, sine, cosine), dim=-1)
                .reshape(1, angles.shape[0], 1, rotary_dim // 2, 2, 2)
                .to(dtype)
            )

        binding = self._offloaded_residency()
        if binding is None:
            return build(stored.to(device=device))
        with binding.lease() as lease:
            return build(lease.get("inv_freq", dtype=stored.dtype).to(device=device))


class MiniMaxH3DiT(ResidencyRouted, torch.nn.Module):
    """Unregistered batch-1 H3 DiT with native packed AV input and output."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: MiniMaxH3Config,
        attention_kernel: AttentionKernel,
        provider_evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations = INITLESS,
        fp32_operations: Operations | None = None,
        text_operations: Operations | None = None,
        time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
    ) -> None:
        super().__init__()
        fp32_operations = operations if fp32_operations is None else fp32_operations
        text_operations = operations if text_operations is None else text_operations
        if time_embedding_kind not in ("curve", "mlp"):
            raise ValueError("time_embedding_kind must be curve or mlp")
        self.config = config
        self.time_embedding_kind = time_embedding_kind
        rotary_dim = min(96, config.attention_head_dim // 6 * 6)
        if rotary_dim < 6:
            raise ValueError("MiniMax H3 attention head dimension must support three RoPE axes")
        patch_width = config.video_latent_channels * math.prod(config.patch)
        self.video_patch_proj = fp32_operations.linear(patch_width, config.hidden_width)
        self.audio_patch_proj = fp32_operations.linear(
            config.audio_latent_channels, config.hidden_width
        )
        self.condition_proj = text_operations.linear(config.text_width, config.hidden_width)
        if time_embedding_kind == "curve":
            self.register_buffer(
                "adaln_t_table", torch.empty(_CURVE_TABLE_ROWS, _CURVE_TIME_EMBED_DIM)
            )
        else:
            self.time_embedder = _MiniMaxH3TimeEmbedder(
                config.hidden_width, operations=fp32_operations
            )
        self.rope = _MiniMaxH3RoPE(rotary_dim // 6)
        self.token_refiner = _MiniMaxH3TokenRefiner(
            config,
            exclude_packed_attention_modifiers(attention_kernel),
            provider_evidence,
            operations=text_operations,
            rotary_dim=rotary_dim,
        )
        adaln_operations = fp32_operations if time_embedding_kind == "curve" else operations
        adaln_input_width = (
            _CURVE_TIME_EMBED_DIM if time_embedding_kind == "curve" else _MLP_TIME_EMBED_DIM
        )
        self.blocks = torch.nn.ModuleList(
            _MiniMaxH3Block(
                config,
                attention_kernel,
                provider_evidence,
                operations=operations,
                fp32_operations=adaln_operations,
                rotary_dim=rotary_dim,
                time_dim=adaln_input_width,
                apply_silu=time_embedding_kind == "mlp",
            )
            for _ in range(config.depth)
        )
        self.final_layer = _MiniMaxH3FinalLayer(
            config,
            time_dim=adaln_input_width,
            apply_silu=time_embedding_kind == "mlp",
            norm_operations=operations,
            adaln_operations=adaln_operations,
            fp32_operations=fp32_operations,
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self._rotary_dim = rotary_dim

    def _validate_inputs(
        self,
        value: MultiStreamLatent[torch.Tensor],
        video_sigma: float,
        context: torch.Tensor,
        conditioning: MiniMaxH3DiTConditioning,
        control: object | None,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None,
    ) -> None:
        if type(value) is not MultiStreamLatent:
            raise TypeError("value must be an exact MultiStreamLatent")
        if value.roles != ("video", "audio"):
            raise ValueError("MiniMax H3 requires exact ordered roles video, audio")
        video, audio = value.by_role("video"), value.by_role("audio")
        if video.ndim != 5 or video.shape[0:2] != (1, self.config.video_latent_channels):
            raise ValueError("MiniMax H3 video latent must be [1,24,t,h,w]")
        if audio.ndim != 4 or audio.shape[0:3] != (
            1,
            self.config.audio_latent_channels,
            self.config.audio_content_channels,
        ):
            raise ValueError("MiniMax H3 audio latent must be [1,32,2,t]")
        if control is not None and not isinstance(control, MiniMaxH3ControlPatch):
            raise ValueError("MiniMax H3 does not support generic control")
        if type(video_sigma) is not float or not 0.0 < video_sigma <= 1.0:
            raise ValueError("video_sigma must be a float within (0, 1]")
        if not video.is_floating_point() or not audio.is_floating_point():
            raise ValueError("MiniMax H3 latents must be floating-point")
        if video.device != audio.device or video.dtype != audio.dtype:
            raise ValueError("MiniMax H3 video and audio must share device and dtype")
        if denoise_mask is not None:
            if type(denoise_mask) is not MultiStreamLatent:
                raise TypeError("denoise_mask must be an exact MultiStreamLatent")
            if denoise_mask.roles != value.roles:
                raise ValueError(
                    "MiniMax H3 denoise mask requires exact ordered roles video, audio"
                )
            for stream, target in zip(denoise_mask.streams, value.streams, strict=True):
                mask = stream.payload
                if (
                    type(mask) is not torch.Tensor
                    or tuple(mask.shape) != tuple(target.payload.shape)
                    or not mask.is_floating_point()
                    or mask.device != target.payload.device
                ):
                    raise ValueError(
                        f"MiniMax H3 {stream.role} denoise mask must match "
                        "its latent shape and device"
                    )
                if not bool(torch.isfinite(mask).all()):
                    raise ValueError("MiniMax H3 denoise mask values must be finite")
                if float(mask.amin()) < 0.0 or float(mask.amax()) > 1.0:
                    raise ValueError("MiniMax H3 denoise mask values must be within [0, 1]")
        if (
            context.ndim != 3
            or context.shape[0] != 1
            or context.shape[1] < 1
            or context.shape[2] not in (self.config.text_width, self.config.hidden_width)
            or not context.is_floating_point()
            or context.device != video.device
        ):
            raise ValueError("context must be floating [1, tokens, text_width|hidden_width]")
        if type(conditioning) is not MiniMaxH3DiTConditioning:
            raise TypeError("conditioning must be exact MiniMaxH3DiTConditioning")
        tags = conditioning.text_token_tags
        if tags is not None and (
            tags.shape != context.shape[:2]
            or tags.device != video.device
            or tags.dtype == torch.bool
            or tags.is_floating_point()
            or not bool(torch.all((tags == 0) | (tags == 1)))
        ):
            raise ValueError("text token tags must be integer vision/text tags matching context")
        if type(conditioning.keyframes) is not tuple:
            raise TypeError("keyframes must be a tuple")
        if type(conditioning.guides) is not tuple:
            raise TypeError("guides must be a tuple")
        if type(conditioning.references) is not tuple:
            raise TypeError("references must be a tuple")
        if conditioning.keyframes and conditioning.references:
            raise ValueError("keyframes and references are mutually exclusive H3 tasks")
        if type(conditioning.seed) is not int:
            raise TypeError("condition seed must be an integer")
        for name, timestep in (
            ("visual", conditioning.visual_noise_timestep),
            ("audio", conditioning.audio_noise_timestep),
        ):
            if type(timestep) is not float or not 0.0 <= timestep <= 1.0:
                raise ValueError(f"{name} condition timestep must be within [0, 1]")
        if len(conditioning.keyframes) > 2:
            raise ValueError("MiniMax H3 accepts at most two keyframes")
        if conditioning.frame_count is not None and (
            type(conditioning.frame_count) is not int or conditioning.frame_count < 1
        ):
            raise ValueError("frame_count must be a positive integer")
        indices: list[int] = []
        for keyframe in conditioning.keyframes:
            if type(keyframe) is not MiniMaxH3KeyframeLatent:
                raise TypeError("keyframes must be exact MiniMaxH3KeyframeLatent values")
            if type(keyframe.resolved_frame_index) is not int:
                raise TypeError("keyframe index must be an integer")
            indices.append(keyframe.resolved_frame_index)
            if keyframe.video.shape != (
                1,
                self.config.video_latent_channels,
                1,
                video.shape[3],
                video.shape[4],
            ):
                raise ValueError("keyframe latent must be one target-sized video frame")
            self._validate_condition_tensor(keyframe.video, video, "keyframe")
        if len(indices) != len(set(indices)):
            raise ValueError("keyframe indices must be unique")
        allowed_indices = {0}
        if conditioning.frame_count is not None:
            allowed_indices.add(conditioning.frame_count - 1)
        if any(index not in allowed_indices for index in indices):
            raise ValueError("keyframes must resolve to the first or declared last frame")
        guide_geometries: list[MiniMaxH3GuideTokenGeometry] = []
        for guide in conditioning.guides:
            if type(guide) is not TimelineGuide:
                raise TypeError("guides must be exact TimelineGuide values")
            if conditioning.frame_count is None:
                raise ValueError("timeline guides require the target frame_count")
            guide.validate_for_target(conditioning.frame_count)
            if guide.latent.roles not in (("video",), ("audio",), ("video", "audio")):
                raise ValueError(
                    "timeline guide roles must be video and/or audio in canonical order"
                )
            if "video" in guide.latent.roles:
                guide_video = guide.latent.by_role("video")
                if (
                    type(guide_video) is not torch.Tensor
                    or guide_video.ndim != 5
                    or guide_video.shape[:2]
                    != (
                        1,
                        self.config.video_latent_channels,
                    )
                    or guide_video.shape[2] < 1
                    or guide_video.shape[3:] != video.shape[3:]
                ):
                    raise ValueError("guide video must have target-sized H3 latent geometry")
                if (
                    MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.content_extent(guide_video.shape[2])
                    != guide.frame_count
                ):
                    raise ValueError("guide video temporal geometry must match its frame count")
                self._validate_condition_tensor(guide_video, video, "guide video")
            else:
                guide_video = None
            if "audio" in guide.latent.roles:
                guide_audio = guide.latent.by_role("audio")
                if (
                    type(guide_audio) is not torch.Tensor
                    or guide_audio.ndim != 4
                    or guide_audio.shape[:3]
                    != (
                        1,
                        self.config.audio_latent_channels,
                        self.config.audio_content_channels,
                    )
                    or guide_audio.shape[3] < 1
                ):
                    raise ValueError("guide audio must have H3 stereo latent geometry")
                remaining_audio = int(
                    audio.shape[-1]
                    - MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(guide.frame_index)
                )
                if guide_audio.shape[-1] > remaining_audio:
                    raise ValueError("guide audio exceeds the target timeline")
                self._validate_condition_tensor(guide_audio, audio, "guide audio")
            else:
                guide_audio = None
            guide_geometries.append(
                MiniMaxH3GuideTokenGeometry(
                    guide.frame_index,
                    guide.frame_count,
                    None
                    if guide_video is None
                    else MiniMaxH3VideoLatentGeometry(
                        guide_video.shape[2], guide_video.shape[3], guide_video.shape[4]
                    ),
                    None if guide_audio is None else guide_audio.shape[-1],
                )
            )
        validate_minimax_h3_guide_timeline(tuple(guide_geometries))
        for reference in conditioning.references:
            if type(reference) is not MiniMaxH3ReferenceLatents:
                raise TypeError("references must be exact MiniMaxH3ReferenceLatents values")
            if type(reference.kind) is not MiniMaxH3ReferenceKind:
                raise TypeError("reference kind must be exact MiniMaxH3ReferenceKind")
            if reference.kind is MiniMaxH3ReferenceKind.IMAGE and (
                reference.video is None
                or reference.audio is not None
                or reference.video.ndim != 5
                or reference.video.shape[2] != 1
            ):
                raise ValueError("image references require exactly one video latent frame")
            if reference.kind is MiniMaxH3ReferenceKind.AUDIO and (
                reference.audio is None or reference.video is not None
            ):
                raise ValueError("audio references require only an audio latent")
            if reference.kind is MiniMaxH3ReferenceKind.VIDEO and reference.video is None:
                raise ValueError("video references require a video latent")
            if reference.video is not None:
                if (
                    reference.video.ndim != 5
                    or reference.video.shape[:2]
                    != (
                        1,
                        self.config.video_latent_channels,
                    )
                    or min(reference.video.shape[2:]) < 1
                ):
                    raise ValueError("reference video must have H3 video latent geometry")
                self._validate_condition_tensor(reference.video, video, "reference video")
            if reference.audio is not None:
                if (
                    reference.audio.ndim != 4
                    or reference.audio.shape[:3]
                    != (
                        1,
                        self.config.audio_latent_channels,
                        self.config.audio_content_channels,
                    )
                    or reference.audio.shape[3] < 1
                ):
                    raise ValueError("reference audio must have H3 stereo latent geometry")
                self._validate_condition_tensor(reference.audio, audio, "reference audio")

    @staticmethod
    def _validate_condition_tensor(
        condition: torch.Tensor, target: torch.Tensor, name: str
    ) -> None:
        if not condition.is_floating_point() or condition.device != target.device:
            raise ValueError(f"{name} must be floating on the target device")

    def _rope_table(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.rope.table(position_ids, device, dtype)

    @staticmethod
    def _curve_embedding(table: torch.Tensor, values: tuple[float, ...]) -> torch.Tensor:
        times = torch.tensor(values, dtype=torch.float32, device=table.device)
        position = times.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))

    def _curve_time_embedding(
        self, values: tuple[float, ...], device: torch.device
    ) -> torch.Tensor:
        stored = cast(torch.Tensor, self.adaln_t_table)
        binding = self._offloaded_residency()
        if binding is None:
            return self._curve_embedding(stored.to(device=device), values)
        with binding.lease() as lease:
            table = lease.get("adaln_t_table", dtype=stored.dtype).to(device=device)
            return self._curve_embedding(table, values)

    def preprocess_text_embeddings(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[2] == self.config.hidden_width:
            return context
        return self.token_refiner(self.condition_proj(context))

    def _condition_rows(
        self,
        conditioning: MiniMaxH3DiTConditioning,
        patch: tuple[int, int, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        video_rows: list[torch.Tensor] = []
        audio_rows: list[torch.Tensor] = []
        for video in (
            *(keyframe.video for keyframe in conditioning.keyframes),
            *(
                guide.latent.by_role("video")
                for guide in conditioning.guides
                if "video" in guide.latent.roles
            ),
            *(
                reference.video
                for reference in conditioning.references
                if reference.video is not None
            ),
        ):
            rows = _patchify_video(_pad_video(video.float(), patch), patch)
            if conditioning.visual_noise_timestep < 1.0:
                generator = torch.Generator("cpu").manual_seed(conditioning.seed)
                noise = torch.randn(rows.shape, generator=generator, dtype=torch.float32)
                amount = conditioning.visual_noise_timestep
                rows = amount * rows + (1.0 - amount) * noise.to(rows.device)
            video_rows.append(rows)
        for audio in (
            *(
                guide.latent.by_role("audio")
                for guide in conditioning.guides
                if "audio" in guide.latent.roles
            ),
            *(
                reference.audio
                for reference in conditioning.references
                if reference.audio is not None
            ),
        ):
            rows = _pack_audio(audio.float())
            if conditioning.audio_noise_timestep < 1.0:
                generator = torch.Generator("cpu").manual_seed(conditioning.seed + 1)
                noise = torch.randn(rows.shape, generator=generator, dtype=torch.float32)
                amount = conditioning.audio_noise_timestep
                rows = amount * rows + (1.0 - amount) * noise.to(rows.device)
            audio_rows.append(rows)
        return (
            torch.cat(video_rows) if video_rows else None,
            torch.cat(audio_rows) if audio_rows else None,
        )

    def forward(
        self,
        value: MultiStreamLatent[torch.Tensor],
        video_sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning | None = None,
        sigmas: MiniMaxH3Sigmas = MINIMAX_H3_SIGMAS,
        sampler_sigmas: tuple[float, ...] | None = None,
        control: object | None = None,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
        attention_kernel_factory: MiniMaxH3AttentionKernelFactory | None = None,
        sequence_sharding: MiniMaxH3SequenceSharding | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        if sequence_sharding is not None:
            if type(sequence_sharding) is not MiniMaxH3SequenceSharding:
                raise TypeError("sequence_sharding must be an exact MiniMaxH3SequenceSharding")
            if attention_kernel_factory is None:
                raise ValueError("sequence sharding requires an attention kernel factory")
            if control is not None:
                raise ValueError("MiniMax H3 control does not support sequence sharding")
        selected_conditioning = MiniMaxH3DiTConditioning() if conditioning is None else conditioning
        self._validate_inputs(
            value, video_sigma, context, selected_conditioning, control, denoise_mask
        )
        if type(sigmas) is not MiniMaxH3Sigmas:
            raise TypeError("sigmas must be an exact MiniMaxH3Sigmas")
        video_source, audio_source = value.by_role("video"), value.by_role("audio")
        video_sigma_tensor, audio_sigma_tensor = _h3_stream_sigmas(
            video_sigma, sigmas, video_source.device
        )
        audio_carry = (audio_sigma_tensor / video_sigma_tensor).to(audio_source.dtype)
        network_input = _h3_latent(video_source, audio_source * audio_carry)
        if attention_kernel_factory is None:
            output = self._forward_network(
                network_input,
                video_sigma,
                context,
                selected_conditioning,
                sigmas,
                sampler_sigmas,
                control=cast("MiniMaxH3ControlPatch | None", control),
                denoise_mask=denoise_mask,
            )
        elif sequence_sharding is None:
            output = self._forward_network(
                network_input,
                video_sigma,
                context,
                selected_conditioning,
                sigmas,
                sampler_sigmas,
                control=cast("MiniMaxH3ControlPatch | None", control),
                denoise_mask=denoise_mask,
                attention_kernel_factory=attention_kernel_factory,
            )
        else:
            output = self._forward_network(
                network_input,
                video_sigma,
                context,
                selected_conditioning,
                sigmas,
                sampler_sigmas,
                control=cast("MiniMaxH3ControlPatch | None", control),
                denoise_mask=denoise_mask,
                attention_kernel_factory=attention_kernel_factory,
                sequence_sharding=sequence_sharding,
            )
        first = 1.0 - sigmas.audio_scale
        second = (1.0 + (sigmas.audio_scale - 1.0) * audio_sigma_tensor).to(
            output.by_role("audio").dtype
        )
        audio_output = first * network_input.by_role("audio") + second * output.by_role("audio")
        return _h3_latent(output.by_role("video"), audio_output.to(audio_source.dtype))

    def _forward_network(
        self,
        value: MultiStreamLatent[torch.Tensor],
        video_sigma: float,
        context: torch.Tensor,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: MiniMaxH3Sigmas,
        sampler_sigmas: tuple[float, ...] | None,
        *,
        control: MiniMaxH3ControlPatch | None = None,
        denoise_mask: MultiStreamLatent[torch.Tensor] | None = None,
        attention_kernel_factory: MiniMaxH3AttentionKernelFactory | None = None,
        sequence_sharding: MiniMaxH3SequenceSharding | None = None,
    ) -> MultiStreamLatent[torch.Tensor]:
        video_source, audio_source = value.by_role("video"), value.by_role("audio")
        original_shape = video_source.shape[2:]
        video = _pad_video(video_source, self.config.patch)
        references = tuple(
            MiniMaxH3ReferenceLatents(
                reference.kind,
                None if reference.video is None else _pad_video(reference.video, self.config.patch),
                reference.audio,
            )
            for reference in conditioning.references
        )
        selected_conditioning = replace(conditioning, references=references)
        layout = _PackedLayout(context.shape[1], video, audio_source, selected_conditioning)
        facts = MiniMaxH3PackedSequenceFacts(
            layout.sequence_length,
            cast(
                "tuple[tuple[int, int, MiniMaxH3PackedSegmentKind], ...]",
                layout.segments,
            ),
        )
        attention_kernel = None
        if attention_kernel_factory is None:
            bound_kernel = bind_packed_attention_kernel(self._attention_kernel, facts)
            if bound_kernel is not self._attention_kernel:
                attention_kernel = bound_kernel
        else:
            attention_kernel = attention_kernel_factory(facts)
            if not isinstance(cast("object", attention_kernel), AttentionKernel):
                raise TypeError("attention kernel factory must return an AttentionKernel")
            attention_kernel = bind_packed_attention_kernel(attention_kernel, facts)
            if sequence_sharding is not None:
                if sequence_sharding.facts != facts:
                    raise ValueError("sequence sharding facts do not match the packed invocation")
                sequence_sharding.validate_kernel(attention_kernel, self.config.attention_heads)
        video_sigma_tensor, audio_sigma_tensor = _h3_stream_sigmas(
            video_sigma, sigmas, video_source.device
        )
        video_time = float(1.0 - video_sigma_tensor)
        audio_time = float(1.0 - audio_sigma_tensor)
        segment_time = {
            "text": video_time,
            "video": video_time,
            "audio": audio_time,
            "condition": max(video_time, conditioning.visual_noise_timestep),
            "condition_audio": max(audio_time, conditioning.audio_noise_timestep),
            "reference_video": max(video_time, conditioning.visual_noise_timestep),
            "reference_audio": max(audio_time, conditioning.audio_noise_timestep),
        }
        video_rows_time: torch.Tensor | None = None
        audio_rows_time: torch.Tensor | None = None
        if denoise_mask is not None:
            video_values = _video_mask_row_values(
                denoise_mask.by_role("video"),
                cast("tuple[int, int, int]", tuple(video.shape[2:])),
                self.config.patch,
            )
            if video_values is not None:
                video_rows_time = (1.0 - video_values * video_sigma_tensor).clamp(
                    max=max(video_time, _VISUAL_CONDITION_TIMESTEP)
                )
                if video_rows_time.unique().numel() == 1:
                    segment_time["video"] = float(video_rows_time[0])
                    video_rows_time = None
            audio_values = _audio_mask_row_values(denoise_mask.by_role("audio"))
            if audio_values is not None:
                audio_rows_time = (1.0 - audio_values * audio_sigma_tensor).clamp(
                    max=max(audio_time, _AUDIO_CONDITION_TIMESTEP)
                )
                if audio_rows_time.unique().numel() == 1:
                    segment_time["audio"] = float(audio_rows_time[0])
                    audio_rows_time = None
        unique_times = tuple(
            sorted(
                {video_time, audio_time}
                | {segment_time[kind] for _, _, kind in layout.segments}
                | (set(video_rows_time.unique().tolist()) if video_rows_time is not None else set())
                | (set(audio_rows_time.unique().tolist()) if audio_rows_time is not None else set())
            )
        )
        time_row = {time: index for index, time in enumerate(unique_times)}
        video_row_indices = (
            None if video_rows_time is None else _row_time_indices(video_rows_time, time_row)
        )
        audio_row_indices = (
            None if audio_rows_time is None else _row_time_indices(audio_rows_time, time_row)
        )
        modality = {
            "text": 1,
            "video": 0,
            "audio": 2,
            "condition": 0,
            "condition_audio": 2,
            "reference_video": 0,
            "reference_audio": 2,
        }
        modulation_segments: list[_ModulationSegment] = []
        tags = conditioning.text_token_tags
        for start, stop, kind in layout.segments:
            row_base = time_row[segment_time[kind]] * 3
            if kind == "text" and tags is not None:
                values = tags[0].tolist()
                run_start = 0
                for index in range(1, stop - start + 1):
                    if index == stop - start or values[index] != values[run_start]:
                        modulation_segments.append(
                            (start + run_start, start + index, row_base + int(values[run_start]))
                        )
                        run_start = index
            elif kind == "video" and video_row_indices is not None:
                modulation_segments.append((start, stop, video_row_indices * 3 + modality[kind]))
            elif kind == "audio" and audio_row_indices is not None:
                modulation_segments.append((start, stop, audio_row_indices * 3 + modality[kind]))
            else:
                modulation_segments.append((start, stop, row_base + modality[kind]))
        segments_tuple = tuple(modulation_segments)

        target_video_rows = _patchify_video(video.float(), self.config.patch)
        target_audio_rows = _pack_audio(audio_source.float())
        condition_video, condition_audio = self._condition_rows(
            selected_conditioning, self.config.patch
        )
        all_video_rows = target_video_rows
        if condition_video is not None:
            all_video_rows = torch.empty(
                layout.video_update.shape[0],
                target_video_rows.shape[1],
                dtype=torch.float32,
                device=video.device,
            )
            update = layout.video_update.to(video.device)
            all_video_rows[~update] = condition_video
            all_video_rows[update] = target_video_rows
        all_audio_rows = target_audio_rows
        if condition_audio is not None:
            all_audio_rows = torch.empty(
                layout.audio_update.shape[0],
                target_audio_rows.shape[1],
                dtype=torch.float32,
                device=video.device,
            )
            update = layout.audio_update.to(video.device)
            all_audio_rows[~update] = condition_audio
            all_audio_rows[update] = target_audio_rows
        video_embeddings = self.video_patch_proj(all_video_rows).to(video.dtype)
        audio_embeddings = self.audio_patch_proj(all_audio_rows).to(video.dtype)
        text_embeddings = self.preprocess_text_embeddings(context)[0]

        hidden = torch.empty(
            1,
            layout.sequence_length,
            self.config.hidden_width,
            dtype=video.dtype,
            device=video.device,
        )
        video_offset = 0
        audio_offset = 0
        for start, stop, kind in layout.segments:
            count = stop - start
            if kind == "text":
                hidden[0, start:stop] = text_embeddings
            elif kind in ("condition", "reference_video", "video"):
                hidden[0, start:stop] = video_embeddings[video_offset : video_offset + count]
                video_offset += count
            else:
                hidden[0, start:stop] = audio_embeddings[audio_offset : audio_offset + count]
                audio_offset += count

        if self.time_embedding_kind == "curve":
            time_embedding = self._curve_time_embedding(unique_times, video.device)
        else:
            time_values = torch.tensor(unique_times, dtype=torch.float32, device=video.device)
            time_embedding = self.time_embedder(time_values).to(video.dtype)
        rope_table = self._rope_table(layout.position_ids, video.device, video.dtype)
        if sequence_sharding is not None:
            shard = sequence_sharding.shard
            hidden = hidden[:, shard.start : shard.stop]
            rope_table = rope_table[:, shard.start : shard.stop]
            if shard.padded_rows:
                hidden = F.pad(hidden, (0, 0, 0, shard.padded_rows))
                rope_table = torch.cat(
                    (
                        rope_table,
                        torch.zeros(
                            1,
                            shard.padded_rows,
                            *rope_table.shape[2:],
                            dtype=rope_table.dtype,
                            device=rope_table.device,
                        ),
                    ),
                    dim=1,
                )
            local_segments = _translate_modulation_segments(segments_tuple, shard.start, shard.stop)
            queue = make_prefetch_queue(self.blocks)
            try:
                for block in self.blocks:
                    prefetch_queue_pop(queue, block)
                    hidden = block(
                        hidden,
                        time_embedding,
                        local_segments,
                        rope_table,
                        attention_kernel=attention_kernel,
                    )
                prefetch_queue_pop(queue, None)
            finally:
                close_prefetch_queue(queue)
            hidden = sequence_sharding.gather(hidden, sequence_sharding.partition, shard)
            if type(hidden) is not torch.Tensor:
                raise TypeError("sequence gather must return an exact torch.Tensor")
            expected_hidden = (1, layout.sequence_length, self.config.hidden_width)
            if tuple(hidden.shape) != expected_hidden:
                raise ValueError(f"sequence gather must return shape {expected_hidden}")
            if hidden.dtype != video.dtype or hidden.device != video.device:
                raise ValueError("sequence gather must preserve the hidden dtype and device")
        else:
            queue = make_prefetch_queue(self.blocks)
            try:
                control_context = MiniMaxH3ControlBlockContext(
                    layout,
                    time_embedding,
                    segments_tuple,
                    rope_table,
                    attention_kernel,
                )
                for block_index, block in enumerate(self.blocks):
                    prefetch_queue_pop(queue, block)
                    if control is not None:
                        control.before_base_block(hidden, block_index)
                    if attention_kernel is None:
                        hidden = block(hidden, time_embedding, segments_tuple, rope_table)
                    else:
                        hidden = block(
                            hidden,
                            time_embedding,
                            segments_tuple,
                            rope_table,
                            attention_kernel=attention_kernel,
                        )
                    if control is not None:
                        hidden = control.after_base_block(hidden, block_index, control_context)
                prefetch_queue_pop(queue, None)
            finally:
                close_prefetch_queue(queue)

        video_segment: _ModulationSegment = next(
            (
                start,
                stop,
                time_row[segment_time["video"]] if video_row_indices is None else video_row_indices,
            )
            for start, stop, kind in layout.segments
            if kind == "video"
        )
        audio_segment: _ModulationSegment = next(
            (
                start,
                stop,
                time_row[segment_time["audio"]] if audio_row_indices is None else audio_row_indices,
            )
            for start, stop, kind in layout.segments
            if kind == "audio"
        )
        video_rows, audio_rows = self.final_layer(
            hidden[0],
            time_embedding,
            video_segment,
            audio_segment,
            video_sigma,
            sampler_sigmas,
            (sigmas.video.shift, sigmas.audio_shift),
        )
        video_output = _unpatchify_video(
            video_rows,
            video.shape[2],
            video.shape[3] // self.config.patch[1],
            video.shape[4] // self.config.patch[2],
            self.config.video_latent_channels,
            self.config.patch,
        )
        temporal, height, width = original_shape
        video_output = video_output[:, :, :temporal, :height, :width]
        audio_output = _unpack_audio(audio_rows)
        return _h3_latent(
            -video_output.to(video_source.dtype), -audio_output.to(audio_source.dtype)
        )


def assemble_minimax_h3_dit(
    *,
    operations: Operations = INITLESS,
    fp32_operations: Operations | None = None,
    text_operations: Operations | None = None,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
    attention_selection: AttentionSelection,
) -> MiniMaxH3DiT:
    """Construct the exact unregistered production H3 DiT source."""
    kernel, evidence = minimax_h3_attention_provider(attention_selection)
    return MiniMaxH3DiT(
        MINIMAX_H3_CONFIG,
        kernel,
        evidence,
        operations=operations,
        fp32_operations=fp32_operations,
        text_operations=text_operations,
        time_embedding_kind=time_embedding_kind,
    )


def _minimax_h3_execution_provider(source: ComponentPlan[object] | MiniMaxH3DiT) -> str:
    """Name the H3 DiT execution provider from its plan or live module tree."""

    if type(source) is ComponentPlan:
        quant = tuple(source.quant.values())
        if not quant:
            return "bf16-linear"
        if all(
            item.format == "int8_tensorwise"
            and item.parameters.get("convrot") is True
            and type(item.parameters.get("convrot_groupsize")) is int
            for item in quant
        ):
            return "int8-convrot-kitchen"
        raise ValueError("H3 plan has an unsupported execution provider")
    if type(source) is MiniMaxH3DiT:
        block = cast("_MiniMaxH3Block", source.blocks[0])
        projection = cast("object", block.attn.qkv_proj)
        if type(projection) is Int8Linear and projection.convrot:
            return "int8-convrot-kitchen"
        if isinstance(projection, torch.nn.Linear):
            return "bf16-linear"
        raise ValueError("H3 model has an unsupported execution provider")
    raise TypeError("H3 integration source must be a component plan or live DiT")


def minimax_h3_guidance_integration_facts(
    source: ComponentPlan[object] | MiniMaxH3DiT,
) -> tuple[str, ...]:
    """Declare the H3 full-model guidance route from its live provider configuration."""

    return ("topology=guidance", f"execution_provider={_minimax_h3_execution_provider(source)}")


def minimax_h3_sequence_integration_facts(
    source: MiniMaxH3DiT,
    *,
    sequence_ulysses: int,
    sequence_ring: int,
    sequence_guidance: int,
) -> tuple[str, ...]:
    """Declare the H3 sequence-parallel route and its exact rank geometry.

    World size alone does not determine a sequence topology (world four may
    be Ulysses degree four or a CFG-by-Ulysses hybrid), so the receipt
    pre-image binds every degree explicitly.
    """

    degrees = (
        ("sequence_ulysses", sequence_ulysses),
        ("sequence_ring", sequence_ring),
        ("sequence_guidance", sequence_guidance),
    )
    for name, value in degrees:
        if type(value) is not int or value < 1:
            raise ValueError(f"H3 {name} degree must be an exact int of at least one")
    if type(source) is not MiniMaxH3DiT:
        raise TypeError("H3 sequence integration facts require a live MiniMaxH3DiT")
    attention = cast("MiniMaxH3Attention", source.blocks[0].attn)
    evidence = attention.provider_evidence
    return (
        "topology=sequence",
        f"execution_provider={_minimax_h3_execution_provider(source)}",
        f"attention_provider={evidence.provider}",
        *(f"{name}={value}" for name, value in degrees),
    )


__all__ = [
    "MINIMAX_H3_ATTENTION_GEOMETRY",
    "MiniMaxH3Attention",
    "MiniMaxH3AttentionGeometry",
    "MiniMaxH3AttentionProviderEvidence",
    "MiniMaxH3DiT",
    "MiniMaxH3DiTConditioning",
    "MiniMaxH3KeyframeLatent",
    "MiniMaxH3ReferenceKind",
    "MiniMaxH3ReferenceLatents",
    "assemble_minimax_h3_attention",
    "assemble_minimax_h3_dit",
    "minimax_h3_attention_provider",
    "minimax_h3_guidance_integration_facts",
    "minimax_h3_sequence_integration_facts",
]
