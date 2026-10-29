"""Native execution for the maintained Wan 2.1 InfiniteTalk/MultiTalk patch."""

from __future__ import annotations

import ctypes
import math
import sys
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import WAN21_MULTITALK, Wan21MultiTalkConfig

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, bound_compute_dtype

_DEFAULT_ATTENTION = select_attention("flux").kernel


class Wan21MultiTalkBindingError(ValueError):
    """A MultiTalk resource or prepared input no longer matches its identity."""


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    paired = value.reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class MultiTalkRotaryEmbedding1D(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("MultiTalk rotary head width must be positive and even")
        self.head_dim = head_dim
        self.base = 10000

    def forward(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        frequencies = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, device=positions.device).float() / self.head_dim)
        )
        phases = torch.einsum("..., f -> ... f", positions.float(), frequencies)
        phases = phases.repeat_interleave(2, dim=-1)
        cosine = phases.cos()[None, None]
        sine = phases.sin()[None, None]
        value_float = value.float()
        return (value_float * cosine + _rotate_half(value_float) * sine).to(value.dtype)


def _calculate_x_ref_attention_map(
    visual_q: torch.Tensor,
    ref_k: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    split_num: int = 8,
) -> torch.Tensor:
    visual_q = visual_q.transpose(1, 2) * (1.0 / visual_q.shape[-1] ** 0.5)
    batch, heads, sequence, _ = visual_q.shape
    maps: list[torch.Tensor] = []
    chunk_size = min(max(sequence // split_num, 1), sequence)
    keys = ref_k.permute(0, 2, 3, 1)
    for target_mask in target_masks:
        mask = target_mask.reshape(1, 1, 1, -1)
        attention_map = torch.zeros(
            (batch, heads, sequence),
            device=visual_q.device,
            dtype=visual_q.dtype,
        )
        for start in range(0, sequence, chunk_size):
            stop = min(start + chunk_size, sequence)
            attention = visual_q[:, :, start:stop] @ keys
            attention_max = attention.max(dim=-1, keepdim=True).values
            attention = (attention - attention_max).exp()
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
            attention_map[:, :, start:stop] = (attention * mask).sum(-1) / (mask.sum() + 1e-8)
        maps.append(attention_map.mean(dim=1))
    return torch.cat(maps, dim=0)


def multitalk_attention_map(
    visual_q: torch.Tensor,
    ref_k: torch.Tensor,
    grid_shape: tuple[int, int, int],
    target_masks: torch.Tensor,
    *,
    split_num: int = 2,
) -> torch.Tensor:
    """Localize each of two target masks with the Wan self-attention q/k state."""
    if (
        type(visual_q) is not torch.Tensor
        or type(ref_k) is not torch.Tensor
        or type(target_masks) is not torch.Tensor
    ):
        raise TypeError("MultiTalk attention-map inputs must be exact torch tensors")
    if visual_q.ndim != 4 or ref_k.ndim != 4 or target_masks.ndim != 2:
        raise ValueError("MultiTalk attention-map inputs have invalid ranks")
    if (
        not visual_q.is_floating_point()
        or not ref_k.is_floating_point()
        or not (target_masks.is_floating_point() or target_masks.dtype == torch.bool)
        or visual_q.dtype != ref_k.dtype
        or visual_q.device != ref_k.device
    ):
        raise ValueError("MultiTalk attention-map q/k must share floating dtype and device")
    if visual_q.shape[0] != 1 or ref_k.shape[0] != 1 or target_masks.shape[0] != 2:
        raise ValueError("MultiTalk localization requires one video and exactly two targets")
    if visual_q.shape[2:] != ref_k.shape[2:]:
        raise ValueError("MultiTalk localization q/k heads and widths must match")
    if split_num <= 0 or visual_q.shape[2] % split_num:
        raise ValueError("MultiTalk localization heads must divide evenly across splits")
    if len(grid_shape) != 3 or any(type(size) is not int or size < 1 for size in grid_shape):
        raise ValueError("MultiTalk target grid must contain three positive integer dimensions")
    _, height, width = grid_shape
    spatial_tokens = height * width
    if spatial_tokens < 1 or ref_k.shape[1] < spatial_tokens:
        raise ValueError("MultiTalk reference keys do not cover the spatial target grid")
    if target_masks.shape[1] != spatial_tokens:
        raise ValueError("MultiTalk target masks must match the spatial target grid")
    ref_k = ref_k[:, :spatial_tokens]
    head_chunk = visual_q.shape[2] // split_num
    result = torch.zeros(
        (target_masks.shape[0], visual_q.shape[1]),
        device=visual_q.device,
        dtype=visual_q.dtype,
    )
    for index in range(split_num):
        start = index * head_chunk
        stop = (index + 1) * head_chunk
        result += _calculate_x_ref_attention_map(
            visual_q[:, :, start:stop],
            ref_k[:, :, start:stop],
            target_masks.to(device=visual_q.device),
        )
    return result / split_num


def _normalize_and_scale(
    values: torch.Tensor,
    source_range: tuple[torch.Tensor, torch.Tensor],
    target_range: tuple[int, int],
) -> torch.Tensor:
    source_min, source_max = source_range
    target_min, target_max = target_range
    normalized = (values - source_min) / (source_max - source_min + 1e-8)
    return normalized * (target_max - target_min) + target_min


def multitalk_audio_windows(
    encoded_audio: Sequence[torch.Tensor],
    audio_start: int,
    audio_end: int,
) -> torch.Tensor:
    """Gather each raw frame's exact five-frame clamped audio neighborhood."""
    streams = tuple(encoded_audio)
    if len(streams) not in (1, 2):
        raise ValueError("MultiTalk requires one or two encoded audio streams")
    if (
        type(audio_start) is not int
        or type(audio_end) is not int
        or audio_start < 0
        or audio_end <= audio_start
    ):
        raise ValueError("MultiTalk audio bounds must be nonnegative and increasing")
    first = streams[0]
    if (
        type(first) is not torch.Tensor
        or first.layout is not torch.strided
        or not first.is_floating_point()
        or first.ndim != 3
        or any(size < 1 for size in first.shape)
    ):
        raise ValueError("MultiTalk encoded audio must be floating [frames,blocks,width]")
    windows: list[torch.Tensor] = []
    offsets = torch.arange(-2, 3, device=first.device)
    centers = torch.arange(audio_start, audio_end, device=first.device)[:, None] + offsets[None]
    for stream in streams:
        if (
            type(stream) is not torch.Tensor
            or stream.layout is not torch.strided
            or not stream.is_floating_point()
            or stream.ndim != 3
            or stream.shape[1:] != first.shape[1:]
            or stream.dtype != first.dtype
            or stream.device != first.device
            or stream.shape[0] < 1
        ):
            raise ValueError("MultiTalk encoded audio streams must share floating geometry")
        if audio_end > stream.shape[0]:
            pad_length = audio_end - stream.shape[0]
            padding = stream[:1].repeat(pad_length, 1, 1)
            stream = torch.cat((stream, padding), dim=0)
        indices = centers.clamp(min=0, max=stream.shape[0] - 1)
        windows.append(stream[indices][None])
    return torch.cat(windows, dim=0)


def _split_audio_windows(windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    speakers, raw_frames, window, blocks, width = windows.shape
    if window != 5 or (raw_frames - 1) % 4:
        raise ValueError("MultiTalk requires one initial frame followed by groups of four")
    first = windows[:, :1]
    latter_count = (raw_frames - 1) // 4
    latter = windows[:, 1:].reshape(speakers, latter_count, 4, window, blocks, width)
    first_edge = latter[:, :, :1, :3].reshape(speakers, latter_count, 3, blocks, width)
    middle = latter[:, :, 1:-1, 2:3].reshape(speakers, latter_count, 2, blocks, width)
    last_edge = latter[:, :, -1:, 2:].reshape(speakers, latter_count, 3, blocks, width)
    return first, torch.cat((first_edge, middle, last_edge), dim=2)


class MultiTalkAudioProjection(torch.nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        seq_len_vf: int,
        blocks: int,
        channels: int,
        intermediate_dim: int,
        out_dim: int,
        context_tokens: int,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.seq_len_vf = seq_len_vf
        self.blocks = blocks
        self.channels = channels
        self.context_tokens = context_tokens
        self.out_dim = out_dim
        self.proj1 = operations.linear(seq_len * blocks * channels, intermediate_dim)
        self.proj1_vf = operations.linear(seq_len_vf * blocks * channels, intermediate_dim)
        self.proj2 = operations.linear(intermediate_dim, intermediate_dim)
        self.proj3 = operations.linear(intermediate_dim, context_tokens * out_dim)
        self.norm = operations.layer_norm(out_dim)

    def forward(self, first: torch.Tensor, latter: torch.Tensor) -> torch.Tensor:
        if (
            first.ndim != 5
            or latter.ndim != 5
            or first.shape[0] != latter.shape[0]
            or first.shape[2:] != (self.seq_len, self.blocks, self.channels)
            or latter.shape[2:] != (self.seq_len_vf, self.blocks, self.channels)
        ):
            raise ValueError("MultiTalk audio projection received invalid window geometry")
        speakers = first.shape[0]
        video_length = first.shape[1] + latter.shape[1]
        first_hidden = F.relu(self.proj1(first.reshape(-1, self.proj1.in_features)))
        latter_hidden = F.relu(self.proj1_vf(latter.reshape(-1, self.proj1_vf.in_features)))
        combined = torch.cat(
            (
                first_hidden.reshape(speakers, first.shape[1], self.proj2.in_features),
                latter_hidden.reshape(speakers, latter.shape[1], self.proj2.in_features),
            ),
            dim=1,
        ).reshape(-1, self.proj2.in_features)
        combined = F.relu(self.proj2(combined))
        context = self.proj3(combined).reshape(-1, self.context_tokens, self.out_dim)
        return self.norm(context).reshape(speakers, video_length, self.context_tokens, self.out_dim)


def project_audio_features(
    audio_projection: MultiTalkAudioProjection,
    encoded_audio: Sequence[torch.Tensor],
    audio_start: int,
    audio_end: int,
) -> torch.Tensor:
    windows = multitalk_audio_windows(encoded_audio, audio_start, audio_end)
    first, latter = _split_audio_windows(windows)
    context = audio_projection(first, latter)
    return torch.cat(context.split(1, dim=0), dim=2)


class MultiTalkCrossAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        *,
        class_range: int,
        class_interval: int,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("MultiTalk attention width must divide evenly across heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.class_range = class_range
        self.class_interval = class_interval
        self.q_linear = operations.linear(dim, dim)
        self.proj = operations.linear(dim, dim)
        self.kv_linear = operations.linear(encoder_hidden_states_dim, 2 * dim)
        self.rope_1d = MultiTalkRotaryEmbedding1D(self.head_dim)
        self.attention_kernel = attention_kernel

    def _single_speaker(
        self,
        hidden: torch.Tensor,
        audio_context: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        time, height, width = grid_shape
        expected_tokens = time * height * width
        extra = None
        if hidden.shape[1] != expected_tokens:
            extra = hidden[:, -(height * width) :]
            hidden = hidden[:, : -(height * width)]
            time -= 1
        batch = hidden.shape[0]
        spatial = height * width
        if time < 1 or hidden.shape[1] != time * spatial:
            raise ValueError("MultiTalk hidden rows do not match the video token grid")
        hidden = hidden.reshape(batch * time, spatial, self.dim)
        query = self.q_linear(hidden).reshape(batch * time, spatial, self.num_heads, self.head_dim)
        key_value = self.kv_linear(audio_context)
        if key_value.shape[0] != batch * time:
            raise ValueError("MultiTalk audio frames must match the video batch and time")
        key, value = key_value.reshape(
            batch * time,
            audio_context.shape[1],
            2,
            self.num_heads,
            self.head_dim,
        ).unbind(2)
        attended = self.attention_kernel(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        output = self.proj(attended.reshape(batch * time, spatial, self.dim))
        output = output.reshape(batch, time * spatial, self.dim)
        if extra is not None:
            output = torch.cat((output, torch.zeros_like(extra)), dim=1)
        return output

    def _two_speakers(
        self,
        hidden: torch.Tensor,
        audio_context: torch.Tensor,
        grid_shape: tuple[int, int, int],
        attention_map: torch.Tensor,
    ) -> torch.Tensor:
        time, height, width = grid_shape
        extra = None
        if hidden.shape[0] * time != audio_context.shape[0]:
            extra = hidden[:, -(height * width) :]
            hidden = hidden[:, : -(height * width)]
            time -= 1
        batch = hidden.shape[0]
        spatial = height * width
        if time < 1 or hidden.shape[1] != time * spatial:
            raise ValueError("MultiTalk hidden rows do not match the video token grid")
        if attention_map.shape != (2, time * spatial):
            raise ValueError("MultiTalk speaker map must cover both speakers and all video rows")
        hidden = hidden.reshape(batch * time, spatial, self.dim)
        batch_time, sequence, channels = hidden.shape
        query = self.q_linear(hidden).reshape(batch_time, sequence, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)

        speaker_one_range = (0, self.class_interval)
        speaker_two_range = (self.class_range - self.class_interval, self.class_range)
        background_bucket = self.class_range // 2
        speaker_one = _normalize_and_scale(
            attention_map[0],
            (attention_map[0].min(), attention_map[0].max()),
            speaker_one_range,
        )
        speaker_two = _normalize_and_scale(
            attention_map[1],
            (attention_map[1].min(), attention_map[1].max()),
            speaker_two_range,
        )
        background = torch.full(
            (attention_map.shape[1],),
            background_bucket,
            dtype=speaker_one.dtype,
            device=speaker_one.device,
        )
        dominant_speaker = attention_map.argmax(dim=0)
        normalized = torch.stack((speaker_one, speaker_two, background), dim=1)
        positions = normalized[
            torch.arange(attention_map.shape[1], device=attention_map.device),
            dominant_speaker,
        ]

        query = query.reshape(batch, time, self.num_heads, spatial, self.head_dim)
        query = query.permute(0, 2, 1, 3, 4).reshape(
            batch, self.num_heads, time * spatial, self.head_dim
        )
        query = self.rope_1d(query, positions)
        query = query.reshape(batch, self.num_heads, time, spatial, self.head_dim)
        query = query.permute(0, 2, 1, 3, 4).reshape(
            batch_time, self.num_heads, spatial, self.head_dim
        )

        audio_tokens = audio_context.shape[1]
        key_value = self.kv_linear(audio_context).reshape(
            batch_time,
            audio_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        key_positions = torch.zeros(
            audio_tokens,
            dtype=key.dtype,
            device=key.device,
        )
        midpoint = audio_tokens // 2
        key_positions[:midpoint] = sum(speaker_one_range) / 2
        key_positions[midpoint:] = sum(speaker_two_range) / 2
        key_positions = key_positions.repeat(time)
        key = key.reshape(batch, time, self.num_heads, audio_tokens, self.head_dim)
        key = key.permute(0, 2, 1, 3, 4).reshape(
            batch, self.num_heads, time * audio_tokens, self.head_dim
        )
        key = self.rope_1d(key, key_positions)
        key = key.reshape(batch, self.num_heads, time, audio_tokens, self.head_dim)
        key = key.permute(0, 2, 1, 3, 4).reshape(
            batch_time, self.num_heads, audio_tokens, self.head_dim
        )
        attended = self.attention_kernel(query, key, value).transpose(1, 2)
        output = self.proj(attended.reshape(batch_time, sequence, channels))
        output = output.reshape(batch, time * spatial, channels)
        if extra is not None:
            output = torch.cat((output, torch.zeros_like(extra)), dim=1)
        return output

    def forward(
        self,
        hidden: torch.Tensor,
        audio_context: torch.Tensor,
        grid_shape: tuple[int, int, int],
        attention_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_context.ndim != 4 or audio_context.shape[0] != 1:
            raise ValueError("MultiTalk audio context must be [1,time,tokens,width]")
        context = (
            audio_context.expand(hidden.shape[0], -1, -1, -1)
            .reshape(
                hidden.shape[0] * audio_context.shape[1],
                audio_context.shape[2],
                audio_context.shape[3],
            )
            .to(hidden.device)
        )
        if attention_map is None or attention_map.shape[0] <= 1:
            return self._single_speaker(hidden, context, grid_shape)
        if attention_map.shape[0] != 2:
            raise ValueError("MultiTalk supports exactly one or two speakers")
        return self._two_speakers(hidden, context, grid_shape, attention_map.to(hidden.device))


class Wan21MultiTalkAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: Wan21MultiTalkConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.audio_cross_attn = MultiTalkCrossAttention(
            config.patch_width,
            config.audio_context_width,
            config.attention_heads,
            class_range=config.speaker_class_range,
            class_interval=config.speaker_class_interval,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm_x = operations.layer_norm(config.patch_width, elementwise_affine=True)


class Wan21MultiTalk(torch.nn.Module):
    """The exact 40-block standalone InfiniteTalk/MultiTalk patch."""

    def __init__(
        self,
        config: Wan21MultiTalkConfig = WAN21_MULTITALK,
        *,
        operations: Operations = INITLESS,
        audio_operations: Operations | None = None,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config is not WAN21_MULTITALK:
            raise ValueError("Wan 2.1 MultiTalk requires the exact maintained profile")
        self.config = config
        audio_operations = operations if audio_operations is None else audio_operations
        self.audio_proj = MultiTalkAudioProjection(
            seq_len=config.audio_window,
            seq_len_vf=config.latter_audio_window,
            blocks=config.audio_encoder_blocks,
            channels=config.audio_input_width,
            intermediate_dim=config.audio_hidden_width,
            out_dim=config.audio_context_width,
            context_tokens=config.audio_context_tokens,
            operations=audio_operations,
        )
        self.blocks = torch.nn.ModuleList(
            Wan21MultiTalkAttentionBlock(
                config,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.layers)
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def project_audio(
        self,
        encoded_audio: Sequence[torch.Tensor],
        audio_start: int,
        audio_end: int,
    ) -> torch.Tensor:
        streams = tuple(encoded_audio)
        audio_dtype = bound_compute_dtype(self.audio_proj.proj1)
        for stream in streams:
            if (
                type(stream) is not torch.Tensor
                or stream.ndim != 3
                or (audio_dtype is not None and stream.dtype != audio_dtype)
                or stream.shape[1:]
                != (
                    self.config.audio_encoder_blocks,
                    self.config.audio_input_width,
                )
            ):
                raise ValueError(
                    "MultiTalk audio does not match maintained encoder geometry and dtype"
                )
        return project_audio_features(self.audio_proj, streams, audio_start, audio_end)

    def attention_map(
        self,
        visual_q: torch.Tensor,
        ref_k: torch.Tensor,
        grid_shape: tuple[int, int, int],
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        return multitalk_attention_map(visual_q, ref_k, grid_shape, target_masks)

    def forward_block(
        self,
        block_index: int,
        hidden: torch.Tensor,
        audio_context: torch.Tensor,
        grid_shape: tuple[int, int, int],
        *,
        attention_map: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if type(block_index) is not int or not 0 <= block_index < self.config.layers:
            raise IndexError("MultiTalk block index is out of range")
        if type(strength) is not float or not math.isfinite(strength):
            raise TypeError("MultiTalk block strength must be a finite float")
        if not -10.0 <= strength <= 10.0:
            raise ValueError("MultiTalk block strength must be within [-10, 10]")
        block = cast(Wan21MultiTalkAttentionBlock, self.blocks[block_index])
        audio = block.audio_cross_attn(
            block.norm_x(hidden),
            audio_context.to(dtype=hidden.dtype),
            grid_shape,
            attention_map,
        )
        return hidden + audio * strength


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


_RESOURCE_SEALS: weakref.WeakKeyDictionary[Wan21MultiTalk, _ResourceSeal] = (
    weakref.WeakKeyDictionary()
)


def _resource_tensors(model: Wan21MultiTalk) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", tensor) for name, tensor in model.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in model.named_buffers()),
    )


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("Wan 2.1 MultiTalk tensors must track mutation versions") from error


def wan21_multitalk_resource_digest(asset_digest: str, compute_dtype: torch.dtype) -> str:
    """Identity for one assembled MultiTalk artifact."""
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("Wan 2.1 MultiTalk source digest must be canonical blake3 identity")
    if not compute_dtype.is_floating_point:
        raise TypeError("Wan 2.1 MultiTalk compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.wan21-multitalk-resource.v1\n")
    hasher.update(f"asset={asset_digest}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"audio_compute_dtype=float16\n")
    return hasher.hexdigest()


def _bind_wan21_multitalk_resource(  # pyright: ignore[reportUnusedFunction]
    model: Wan21MultiTalk,
    digest: str,
) -> None:
    if type(model) is not Wan21MultiTalk:
        raise TypeError("Wan 2.1 MultiTalk resource must be the exact maintained model")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Wan 2.1 MultiTalk resource digest must be lowercase BLAKE3")
    if model in _RESOURCE_SEALS:
        raise ValueError("Wan 2.1 MultiTalk resource is already bound")
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


def validate_wan21_multitalk_resource(model: Wan21MultiTalk, digest: str) -> None:
    """Require a MultiTalk model to retain assembly-proven state."""
    if type(model) is not Wan21MultiTalk:
        raise TypeError("Wan 2.1 MultiTalk resource must be the exact maintained model")
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
        raise Wan21MultiTalkBindingError(
            "wan21-multitalk-resource-binding-mismatch: model provenance is absent or changed"
        )


def wan21_multitalk_tensor_digest(value: torch.Tensor) -> str:
    """Content identity for one canonical MultiTalk execution tensor."""
    if type(value) is not torch.Tensor or value.layout is not torch.strided:
        raise TypeError("Wan 2.1 MultiTalk input must be an exact strided tensor")
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
    hasher.update(b"dinkster.wan21-multitalk-tensor.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in tensor.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(tensor.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class Wan21MultiTalkExecution:
    """One projected, identity-bound MultiTalk application."""

    model: Wan21MultiTalk
    audio_context: torch.Tensor
    target_masks: torch.Tensor | None
    strength: float
    model_digest: str
    audio_digest: str
    target_masks_digest: str | None

    def __post_init__(self) -> None:
        if type(self.model) is not Wan21MultiTalk:
            raise TypeError("Wan 2.1 MultiTalk model must be the exact maintained patch")
        config = self.model.config
        if (
            type(self.audio_context) is not torch.Tensor
            or self.audio_context.layout is not torch.strided
            or not self.audio_context.is_floating_point()
            or self.audio_context.ndim != 4
            or self.audio_context.shape[0] != 1
            or self.audio_context.shape[1] < 1
            or self.audio_context.shape[3] != config.audio_context_width
            or self.audio_context.shape[2]
            not in (config.audio_context_tokens, 2 * config.audio_context_tokens)
        ):
            raise ValueError("Wan 2.1 MultiTalk audio context must be floating [1,time,32|64,768]")
        speakers = self.audio_context.shape[2] // config.audio_context_tokens
        if speakers == 1:
            if self.target_masks is not None or self.target_masks_digest is not None:
                raise ValueError("single-speaker MultiTalk execution must not carry target masks")
        elif (
            type(self.target_masks) is not torch.Tensor
            or self.target_masks.layout is not torch.strided
            or self.target_masks.ndim != 2
            or self.target_masks.shape[0] != 2
            or self.target_masks.shape[1] < 1
            or not (self.target_masks.is_floating_point() or self.target_masks.dtype == torch.bool)
            or self.target_masks_digest is None
        ):
            raise ValueError("two-speaker MultiTalk execution requires two spatial target masks")
        if type(self.strength) is not float or not math.isfinite(self.strength):
            raise TypeError("Wan 2.1 MultiTalk strength must be a finite float")
        if not -10.0 <= self.strength <= 10.0:
            raise ValueError("Wan 2.1 MultiTalk strength must be within [-10, 10]")
        validate_wan21_multitalk_resource(self.model, self.model_digest)
        if wan21_multitalk_tensor_digest(self.audio_context) != self.audio_digest:
            raise Wan21MultiTalkBindingError("Wan 2.1 MultiTalk audio context identity changed")
        if self.target_masks is not None and (
            wan21_multitalk_tensor_digest(self.target_masks) != self.target_masks_digest
        ):
            raise Wan21MultiTalkBindingError("Wan 2.1 MultiTalk target-mask identity changed")


def snapshot_wan21_multitalk_execution(
    execution: Wan21MultiTalkExecution,
) -> Wan21MultiTalkExecution:
    """Validate then own projected audio and target masks for one sampling execution."""
    if type(execution) is not Wan21MultiTalkExecution:
        raise TypeError("Wan 2.1 MultiTalk input must be exact Wan21MultiTalkExecution")
    return Wan21MultiTalkExecution(
        execution.model,
        execution.audio_context.detach().clone(),
        None if execution.target_masks is None else execution.target_masks.detach().clone(),
        execution.strength,
        execution.model_digest,
        execution.audio_digest,
        execution.target_masks_digest,
    )


__all__ = [
    "Wan21MultiTalk",
    "Wan21MultiTalkBindingError",
    "Wan21MultiTalkExecution",
    "snapshot_wan21_multitalk_execution",
    "validate_wan21_multitalk_resource",
    "wan21_multitalk_resource_digest",
    "wan21_multitalk_tensor_digest",
]
