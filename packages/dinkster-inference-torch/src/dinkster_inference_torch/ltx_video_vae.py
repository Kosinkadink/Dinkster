"""Native LTX causal video VAE.

Faithful standalone port of ``comfy/ldm/lightricks/vae/``
(``causal_video_autoencoder.py``, ``causal_conv3d.py``, ``pixel_norm.py``)
at ComfyUI commit b78cec87 for the LTXV-family configs Dinkster supports
(the reference's Pruna-specific decoder channel overrides are not
ported). State-dict names match the reference.

The reference streams long videos through every stage: each causal conv
keeps a temporal tail so the next chunk sees the same receptive field a
single pass would, the encoder splits the input into
``[first frame] + fixed-size chunks``, and the decoder recursively
re-chunks between up blocks and writes finished frames straight into one
preallocated output buffer. The reference stores that per-call state on
the modules keyed by thread; here it lives in a ``stream`` dictionary
created per encode/decode call and threaded explicitly, so nothing
outlives the call.

The reference sizes chunks from the device's total memory;
:func:`ltxv_vae_max_chunk_bytes` ports that policy as a pure function and
the modules take the resulting byte budget as a parameter (default: the
reference's largest budget).

Noise injection and the timestep-conditioned decode draw from torch's
global RNG in reference order; ``decode`` optionally takes a generator
for its latent noise mix. The reference's timestep machinery views one
embedded timestep row as the whole batch, so timestep-conditioned decode
supports batch size one, exactly like the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.nn.functional as F
from dinkster_inference import LTXVideoVAEConfig

from .ltx_model import _CombinedTimestepEmbedding  # pyright: ignore[reportPrivateUsage]
from .operations import INITLESS, Operations, ResidencyRouted

__all__ = [
    "LTXVideoDecoder",
    "LTXVideoEncoder",
    "LTXVideoVAE",
    "ltxv_vae_max_chunk_bytes",
]

_MIN_VRAM_FOR_CHUNK_SCALING = 6 * 1024**3
_MAX_VRAM_FOR_CHUNK_SCALING = 24 * 1024**3
_MIN_CHUNK_BYTES = 32 * 1024**2
_MAX_CHUNK_BYTES = 128 * 1024**2


def ltxv_vae_max_chunk_bytes(total_memory_bytes: float) -> int:
    """Interpolate the reference budget from total device memory."""
    if total_memory_bytes <= _MIN_VRAM_FOR_CHUNK_SCALING:
        return _MIN_CHUNK_BYTES
    if total_memory_bytes >= _MAX_VRAM_FOR_CHUNK_SCALING:
        return _MAX_CHUNK_BYTES
    interp = (total_memory_bytes - _MIN_VRAM_FOR_CHUNK_SCALING) / (
        _MAX_VRAM_FOR_CHUNK_SCALING - _MIN_VRAM_FOR_CHUNK_SCALING
    )
    return int(_MIN_CHUNK_BYTES + interp * (_MAX_CHUNK_BYTES - _MIN_CHUNK_BYTES))


def _encoder_chunk_frames(frame_bytes: int, max_chunk_bytes: int) -> int:
    """The reference encoder's chunk length rounding: at least two frames,
    then snapped down to four or a multiple of eight."""
    chunk_t = max(2, max_chunk_bytes // frame_bytes)
    if chunk_t < 4:
        return 2
    if chunk_t < 8:
        return 4
    return (chunk_t // 8) * 8


_Stream = dict[torch.nn.Module, object]

_StateT = TypeVar("_StateT")


def _state(stream: _Stream, module: torch.nn.Module, factory: type[_StateT]) -> _StateT:
    found = stream.get(module)
    if found is None:
        found = factory()
        stream[module] = found
    return cast(_StateT, found)


@dataclass
class _ConvState:
    cached: torch.Tensor | None = None
    ended: bool = False


@dataclass
class _DownsampleState:
    cached: torch.Tensor | None = None
    pad_first: bool = True
    cached_x: torch.Tensor | None = None
    cached_input: torch.Tensor | None = None


@dataclass
class _UpsampleState:
    cached: torch.Tensor | None = None
    drop_first_conv: bool = True
    drop_first_res: bool = True


@dataclass
class _ResidualState:
    cached: torch.Tensor | None = None


def _mark_ended(module: torch.nn.Module, stream: _Stream) -> None:
    """The reference's ``mark_conv3d_ended``: flag every causal conv under
    ``module`` that its next chunk is the last."""
    for child in module.modules():
        if isinstance(child, _CausalConv3d):
            _state(stream, child, _ConvState).ended = True


def _cat_if_needed(pieces: list[torch.Tensor | None], dim: int = 2) -> torch.Tensor | None:
    present = [piece for piece in pieces if piece is not None and piece.shape[dim] > 0]
    if len(present) > 1:
        return torch.cat(present, dim)
    if len(present) == 1:
        return present[0]
    return None


def _split2(tensor: torch.Tensor, split_point: int, dim: int = 2) -> tuple[torch.Tensor, ...]:
    return torch.split(tensor, [split_point, tensor.shape[dim] - split_point], dim=dim)


def _add_exchange_cache(
    dest: torch.Tensor | None,
    cache_in: torch.Tensor | None,
    new_input: torch.Tensor,
    dim: int = 2,
) -> torch.Tensor | None:
    """The reference's ``add_exchange_cache``: fold the cached residual
    tail into the leading frames of ``dest`` in place, add the aligned
    body of ``new_input``, and return the residual tail to carry."""
    if dest is not None:
        if cache_in is not None:
            lead = min(dest.shape[dim], cache_in.shape[dim])
            lead_dest, dest = _split2(dest, lead, dim=dim)
            lead_source, cache_in = _split2(cache_in, lead, dim=dim)
            lead_dest.add_(lead_source)
        body, new_input = _split2(new_input, dest.shape[dim], dim)
        dest.add_(body)
    return _cat_if_needed([cache_in, new_input], dim=dim)


def _pixel_norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-8)


def _space_to_depth(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    batch, channels, frames, height, width = x.shape
    p1, p2, p3 = stride
    x = x.view(batch, channels, frames // p1, p1, height // p2, p2, width // p3, p3)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(batch, channels * p1 * p2 * p3, frames // p1, height // p2, width // p3)


def _depth_to_space(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    batch, packed, frames, height, width = x.shape
    p1, p2, p3 = stride
    channels = packed // (p1 * p2 * p3)
    x = x.view(batch, channels, p1, p2, p3, frames, height, width)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(batch, channels, frames * p1, height * p2, width * p3)


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """The reference's ``patchify`` with temporal patch size one; the
    packed channel order is (c, r, q) with the width factor outermost."""
    if patch_size == 1:
        return x
    batch, channels, frames, height, width = x.shape
    x = x.view(
        batch, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch, channels * patch_size**2, frames, height // patch_size, width // patch_size
    )


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    batch, packed, frames, height, width = x.shape
    channels = packed // patch_size**2
    x = x.view(batch, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(batch, channels, frames, height * patch_size, width * patch_size)


class _CausalConv3d(torch.nn.Module):
    """3x3x3 conv with streaming temporal padding. The temporal receptive
    field comes from the carried cache (first-frame repeats before the
    first chunk, plus last-frame repeats at the end when non-causal). Zero
    spatial padding stays on Conv3d to preserve its backend dispatch; reflect
    padding is applied before the convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        spatial_padding_mode: str,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_stride = stride[0]
        self.time_kernel_size = 3
        self.spatial_padding_mode = spatial_padding_mode
        padding = (0, 1, 1) if spatial_padding_mode == "zeros" else 0
        self.conv = operations.conv3d(
            in_channels,
            out_channels,
            (3, 3, 3),
            stride=stride,
            padding=padding,
        )

    def _empty_output(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-frame output with the conv's channel count and spatial dims
        (the 1-pixel spatial halo and 3x3 kernel shrink each spatial dim to
        (size - 1) // stride + 1)."""
        stride = self.conv.stride
        height = (x.shape[3] - 1) // stride[1] + 1
        width = (x.shape[4] - 1) // stride[2] + 1
        return x.new_empty((x.shape[0], self.out_channels, 0, height, width))

    def forward(self, x: torch.Tensor, *, causal: bool, stream: _Stream) -> torch.Tensor:
        state = _state(stream, self, _ConvState)
        cached = state.cached
        if cached is None:
            padding_length = self.time_kernel_size - 1
            if not causal:
                padding_length //= 2
            if x.shape[2] == 0:
                return self._empty_output(x)
            cached = x[:, :, :1].repeat((1, 1, padding_length, 1, 1))
        pieces = [cached, x]
        if state.ended and not causal:
            pieces.append(x[:, :, -1:].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
        input_length = sum(piece.shape[2] for piece in pieces)
        cache_length = (self.time_kernel_size - self.time_stride) + (
            (input_length - self.time_kernel_size) % self.time_stride
        )

        needs_caching = not state.ended
        if needs_caching and cache_length == 0:
            state.cached = x[:, :, :0]
            needs_caching = False
        if needs_caching and x.shape[2] >= cache_length:
            needs_caching = False
            state.cached = x[:, :, -cache_length:]

        x = torch.cat(pieces, dim=2)
        del pieces
        del cached

        if needs_caching:
            state.cached = x[:, :, -cache_length:]
        elif state.ended:
            state.cached = None

        if x.shape[2] < self.time_kernel_size:
            return self._empty_output(x)
        if self.spatial_padding_mode == "reflect":
            x = F.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
        return self.conv(x)


class _ChannelLastLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.norm = operations.layer_norm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class _ResnetBlock3d(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        inject_noise: bool,
        timestep_conditioning: bool,
        spatial_padding_mode: str,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.inject_noise = inject_noise
        self.timestep_conditioning = timestep_conditioning
        self.conv1 = _CausalConv3d(
            in_channels,
            out_channels,
            spatial_padding_mode=spatial_padding_mode,
            operations=operations,
        )
        self.conv2 = _CausalConv3d(
            out_channels,
            out_channels,
            spatial_padding_mode=spatial_padding_mode,
            operations=operations,
        )
        if inject_noise:
            self.per_channel_scale1 = torch.nn.Parameter(torch.empty(in_channels, 1, 1))
            self.per_channel_scale2 = torch.nn.Parameter(torch.empty(in_channels, 1, 1))
        if in_channels != out_channels:
            self.conv_shortcut: torch.nn.Module = operations.conv3d(
                in_channels, out_channels, (1, 1, 1)
            )
            self.norm3: torch.nn.Module = _ChannelLastLayerNorm(in_channels, operations=operations)
        else:
            self.conv_shortcut = torch.nn.Identity()
            self.norm3 = torch.nn.Identity()
        if timestep_conditioning:
            self.scale_shift_table = torch.nn.Parameter(torch.empty(4, in_channels))

    def _feed_spatial_noise(
        self, hidden: torch.Tensor, per_channel_scale: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn(hidden.shape[-2:], device=hidden.device, dtype=hidden.dtype)[None]
        return hidden + (noise * per_channel_scale)[None, :, None]

    def forward(
        self,
        input_tensor: torch.Tensor,
        *,
        causal: bool,
        stream: _Stream,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = input_tensor.shape[0]
        hidden = _pixel_norm(input_tensor)
        shift2 = scale2 = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("timestep-conditioned res blocks need a timestep")
            with self.materialized_state(
                "scale_shift_table", device=hidden.device, dtype=hidden.dtype
            ) as table:
                ada = table[None, ..., None, None, None] + timestep.reshape(batch, 4, -1, 1, 1, 1)
            shift1, scale1, shift2, scale2 = ada.unbind(dim=1)
            hidden = hidden * (1 + scale1) + shift1
        hidden = F.silu(hidden)
        hidden = self.conv1(hidden, causal=causal, stream=stream)
        if self.inject_noise:
            with self.materialized_state(
                "per_channel_scale1", device=hidden.device, dtype=hidden.dtype
            ) as scale:
                hidden = self._feed_spatial_noise(hidden, scale)
        hidden = _pixel_norm(hidden)
        if shift2 is not None and scale2 is not None:
            hidden = hidden * (1 + scale2) + shift2
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden, causal=causal, stream=stream)
        if self.inject_noise:
            with self.materialized_state(
                "per_channel_scale2", device=hidden.device, dtype=hidden.dtype
            ) as scale:
                hidden = self._feed_spatial_noise(hidden, scale)
        residual = self.conv_shortcut(self.norm3(input_tensor))
        state = _state(stream, self, _ResidualState)
        state.cached = _add_exchange_cache(hidden, state.cached, residual, dim=2)
        return hidden


class _MidBlock3d(torch.nn.Module):
    """The reference's ``UNetMidBlock3D``: equal-width res blocks with one
    shared timestep embedder."""

    def __init__(
        self,
        channels: int,
        num_layers: int,
        *,
        inject_noise: bool,
        timestep_conditioning: bool,
        spatial_padding_mode: str,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = _CombinedTimestepEmbedding(channels * 4, operations=operations)
        self.res_blocks = torch.nn.ModuleList(
            _ResnetBlock3d(
                channels,
                channels,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
                operations=operations,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        causal: bool,
        stream: _Stream,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("timestep-conditioned mid blocks need a timestep")
            embedded = self.time_embedder(timestep.flatten(), hidden.dtype)
            embedded = embedded.view(hidden.shape[0], embedded.shape[-1], 1, 1, 1)
        for block in self.res_blocks:
            hidden = block(hidden, causal=causal, stream=stream, timestep=embedded)
        return hidden


class _SpaceToDepthDownsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        *,
        spatial_padding_mode: str,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = _CausalConv3d(
            in_channels,
            out_channels // math.prod(stride),
            spatial_padding_mode=spatial_padding_mode,
            operations=operations,
        )

    def forward(self, x: torch.Tensor, *, causal: bool, stream: _Stream) -> torch.Tensor | None:
        state = _state(stream, self, _DownsampleState)
        if state.cached_input is not None:
            joined = _cat_if_needed([state.cached_input, x], dim=2)
            if joined is None:
                return None
            x = joined
            state.cached_input = None

        if self.stride[0] == 2 and state.pad_first:
            x = torch.cat([x[:, :, :1], x], dim=2)
            state.pad_first = False

        if x.shape[2] < self.stride[0]:
            state.cached_input = x
            return None

        skip = _space_to_depth(x, self.stride)
        batch, packed = skip.shape[0], skip.shape[1]
        skip = skip.view(batch, packed // self.group_size, self.group_size, *skip.shape[2:]).mean(
            dim=2
        )

        y: torch.Tensor | None = self.conv(x, causal=causal, stream=stream)
        assert y is not None
        if self.stride[0] == 2 and y.shape[2] == 1:
            if state.cached_x is not None:
                y = _cat_if_needed([state.cached_x, y], dim=2)
                state.cached_x = None
            else:
                state.cached_x = y
                y = None

        if y is not None:
            y = _space_to_depth(y, self.stride)

        state.cached = _add_exchange_cache(y, state.cached, skip, dim=2)
        return y


class _DepthToSpaceUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        *,
        residual: bool,
        reduction: int,
        spatial_padding_mode: str,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.residual = residual
        self.reduction = reduction
        self.out_channels = math.prod(stride) * in_channels // reduction
        self.conv = _CausalConv3d(
            in_channels,
            self.out_channels,
            spatial_padding_mode=spatial_padding_mode,
            operations=operations,
        )

    def forward(self, x: torch.Tensor, *, causal: bool, stream: _Stream) -> torch.Tensor | None:
        state = _state(stream, self, _UpsampleState)
        y: torch.Tensor | None = _depth_to_space(
            self.conv(x, causal=causal, stream=stream), self.stride
        )
        assert y is not None
        if self.stride[0] == 2 and y.shape[2] > 0 and state.drop_first_conv:
            y = y[:, :, 1:]
            state.drop_first_conv = False
        if self.residual:
            skip = _depth_to_space(x, self.stride)
            num_repeat = math.prod(self.stride) // self.reduction
            skip = skip.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2 and skip.shape[2] > 0 and state.drop_first_res:
                skip = skip[:, :, 1:]
                state.drop_first_res = False
            if y.shape[2] == 0:
                y = None
            state.cached = _add_exchange_cache(y, state.cached, skip, dim=2)
        return y


class LTXVideoEncoder(torch.nn.Module):
    """Causal streaming encoder: the input is split into
    ``[first frame] + chunks`` and each chunk is patchified and pushed
    through the block stack with the temporal caches carrying the seams.
    The uniform log-variance channel is expanded to match the mean width,
    exactly like the reference."""

    def __init__(self, config: LTXVideoVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        padding_mode = config.encoder_spatial_padding_mode
        self.conv_in = _CausalConv3d(
            3 * config.patch_size**2,
            config.base_channels,
            spatial_padding_mode=padding_mode,
            operations=operations,
        )
        blocks: list[torch.nn.Module] = []
        for block, (into, out) in zip(
            config.encoder_blocks, config.encoder_channels(), strict=True
        ):
            if block.kind == "res_x":
                blocks.append(
                    _MidBlock3d(
                        into,
                        block.layers,
                        inject_noise=False,
                        timestep_conditioning=False,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
            elif block.kind == "res_x_y":
                blocks.append(
                    _ResnetBlock3d(
                        into,
                        out,
                        inject_noise=False,
                        timestep_conditioning=False,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
            elif block.kind.endswith("_res"):
                blocks.append(
                    _SpaceToDepthDownsample(
                        into,
                        out,
                        block.stride,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
            else:
                blocks.append(
                    _CausalConv3d(
                        into,
                        out,
                        stride=block.stride,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
        self.down_blocks = torch.nn.ModuleList(blocks)
        self.conv_out = _CausalConv3d(
            config.encoder_channels()[-1][1],
            config.latent_channels + 1,
            spatial_padding_mode=padding_mode,
            operations=operations,
        )

    def _forward_chunk(self, sample: torch.Tensor, stream: _Stream) -> torch.Tensor | None:
        out = self.conv_in(sample, causal=True, stream=stream)
        for block in self.down_blocks:
            out = block(out, causal=True, stream=stream)
            if out is None or out.shape[2] == 0:
                return None
        out = _pixel_norm(out)
        out = F.silu(out)
        out = self.conv_out(out, causal=True, stream=stream)
        if out.shape[2] == 0:
            return None
        last_channel = out[:, -1:].repeat(1, out.shape[1] - 2, 1, 1, 1)
        return torch.cat([out, last_channel], dim=1)

    def forward(self, sample: torch.Tensor, *, max_chunk_bytes: int | None = None) -> torch.Tensor:
        stream: _Stream = {}
        # The reference doubles the budget for the encoder.
        budget = 2 * (_MAX_CHUNK_BYTES if max_chunk_bytes is None else max_chunk_bytes)
        frame_bytes = sample[:, :, :1].numel() * sample.element_size()
        frame_bytes = int(frame_bytes * (self.conv_in.out_channels / self.conv_in.in_channels))

        chunks = [sample[:, :, :1]]
        if sample.shape[2] > 1:
            chunk_t = _encoder_chunk_frames(frame_bytes, budget)
            chunks += list(torch.split(sample[:, :, 1:], chunk_t, dim=2))

        outputs: list[torch.Tensor | None] = []
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                _mark_ended(self, stream)
            output = self._forward_chunk(_patchify(chunk, self.config.patch_size), stream)
            if output is not None:
                outputs.append(output)
        merged = _cat_if_needed(outputs, dim=2)
        if merged is None:
            raise ValueError("the encoder consumed every frame without producing a latent")
        return merged


class LTXVideoDecoder(ResidencyRouted, torch.nn.Module):
    """Causal streaming decoder: one pass keeps memory bounded by
    re-chunking the growing sample between up blocks and writing finished
    frames into a preallocated output buffer."""

    def __init__(self, config: LTXVideoVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        self.causal = config.causal_decoder
        self.timestep_conditioning = config.timestep_conditioning
        padding_mode = config.decoder_spatial_padding_mode
        self.conv_in = _CausalConv3d(
            config.latent_channels,
            config.decoder_input_channels,
            spatial_padding_mode=padding_mode,
            operations=operations,
        )
        blocks: list[torch.nn.Module] = []
        for block, (into, out) in zip(
            tuple(reversed(config.decoder_blocks)), config.decoder_channels(), strict=True
        ):
            if block.kind == "res_x":
                blocks.append(
                    _MidBlock3d(
                        into,
                        block.layers,
                        inject_noise=block.inject_noise,
                        timestep_conditioning=config.timestep_conditioning,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
            elif block.kind == "res_x_y":
                blocks.append(
                    _ResnetBlock3d(
                        into,
                        out,
                        inject_noise=block.inject_noise,
                        timestep_conditioning=False,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
            else:
                blocks.append(
                    _DepthToSpaceUpsample(
                        into,
                        block.stride,
                        residual=block.residual,
                        reduction=block.multiplier,
                        spatial_padding_mode=padding_mode,
                        operations=operations,
                    )
                )
        self.up_blocks = torch.nn.ModuleList(blocks)
        final_channels = config.decoder_channels()[-1][1]
        self.conv_out = _CausalConv3d(
            final_channels,
            3 * config.patch_size**2,
            spatial_padding_mode=padding_mode,
            operations=operations,
        )

        time_scale, height_scale, width_scale, time_offset = 1, 1, 1, 0
        for block in reversed(config.decoder_blocks):
            if block.kind.startswith("compress_"):
                stride = block.stride
                time_scale *= stride[0]
                height_scale *= stride[1]
                width_scale *= stride[2]
                if stride[0] > 1:
                    time_offset = time_offset * stride[0] + 1
        self._output_scale = (
            (time_scale, height_scale * config.patch_size, width_scale * config.patch_size),
            time_offset,
        )

        if config.timestep_conditioning:
            self.timestep_scale_multiplier = torch.nn.Parameter(torch.empty(()))
            self.last_time_embedder = _CombinedTimestepEmbedding(
                final_channels * 2, operations=operations
            )
            self.last_scale_shift_table = torch.nn.Parameter(torch.empty(2, final_channels))

    def decode_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
        (time_scale, height_scale, width_scale), time_offset = self._output_scale
        return (
            input_shape[0],
            3,
            input_shape[2] * time_scale - time_offset,
            input_shape[3] * height_scale,
            input_shape[4] * width_scale,
        )

    def _run_up(
        self,
        index: int,
        sample_holder: list[torch.Tensor | None],
        ended: bool,
        shift_scale: tuple[torch.Tensor, ...] | None,
        scaled_timestep: torch.Tensor | None,
        output_buffer: torch.Tensor,
        output_offset: list[int],
        max_chunk_bytes: int,
        stream: _Stream,
    ) -> None:
        sample = sample_holder[0]
        assert sample is not None
        sample_holder[0] = None
        if index >= len(self.up_blocks):
            sample = _pixel_norm(sample)
            if shift_scale is not None:
                shift, scale = shift_scale
                sample = sample * (1 + scale) + shift
            sample = F.silu(sample)
            if ended:
                _mark_ended(self.conv_out, stream)
            sample = self.conv_out(sample, causal=self.causal, stream=stream)
            if sample.shape[2] > 0:
                sample = _unpatchify(sample, self.config.patch_size)
                frames = sample.shape[2]
                output_buffer[:, :, output_offset[0] : output_offset[0] + frames].copy_(sample)
                output_offset[0] += frames
            return

        block = self.up_blocks[index]
        if ended:
            _mark_ended(block, stream)
        if self.timestep_conditioning and isinstance(block, _MidBlock3d):
            out = block(sample, causal=self.causal, stream=stream, timestep=scaled_timestep)
        else:
            out = block(sample, causal=self.causal, stream=stream)
        del sample
        if out is None or out.shape[2] == 0:
            return

        total_bytes = out.numel() * out.element_size()
        num_chunks = (total_bytes + max_chunk_bytes - 1) // max_chunk_bytes
        if num_chunks == 1:
            # Hand the only reference over so the callee can free it.
            holder: list[torch.Tensor | None] = [out]
            del out
            self._run_up(
                index + 1,
                holder,
                ended,
                shift_scale,
                scaled_timestep,
                output_buffer,
                output_offset,
                max_chunk_bytes,
                stream,
            )
            return
        pieces = torch.chunk(out, chunks=num_chunks, dim=2)
        for piece_index, piece in enumerate(pieces):
            self._run_up(
                index + 1,
                [piece],
                ended and piece_index == len(pieces) - 1,
                shift_scale,
                scaled_timestep,
                output_buffer,
                output_offset,
                max_chunk_bytes,
                stream,
            )

    def forward(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
        max_chunk_bytes: int | None = None,
    ) -> torch.Tensor:
        stream: _Stream = {}
        batch = sample.shape[0]
        _mark_ended(self.conv_in, stream)
        sample = self.conv_in(sample, causal=self.causal, stream=stream)

        shift_scale: tuple[torch.Tensor, ...] | None = None
        scaled_timestep: torch.Tensor | None = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("a timestep-conditioned decoder needs a timestep")
            with self.materialized_state(
                "timestep_scale_multiplier", device=sample.device, dtype=sample.dtype
            ) as multiplier:
                scaled_timestep = timestep * multiplier
            embedded = self.last_time_embedder(scaled_timestep.flatten(), sample.dtype)
            embedded = embedded.view(batch, embedded.shape[-1], 1, 1, 1)
            with self.materialized_state(
                "last_scale_shift_table", device=sample.device, dtype=sample.dtype
            ) as table:
                ada = table[None, ..., None, None, None] + embedded.reshape(batch, 2, -1, 1, 1, 1)
            shift_scale = ada.unbind(dim=1)

        if output_buffer is None:
            output_buffer = torch.empty(
                self.decode_output_shape(sample.shape), dtype=sample.dtype, device=sample.device
            )
        budget = _MAX_CHUNK_BYTES if max_chunk_bytes is None else max_chunk_bytes
        self._run_up(
            0,
            [sample],
            True,
            shift_scale,
            scaled_timestep,
            output_buffer,
            [0],
            budget,
            stream,
        )
        return output_buffer


class LTXPerChannelStatistics(ResidencyRouted, torch.nn.Module):
    """The reference's ``processor``: dataset latent statistics stored
    under hyphenated buffer names."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(channels))
        self.register_buffer("mean-of-means", torch.empty(channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        with (
            self.materialized_state("std-of-means", device=x.device, dtype=x.dtype) as std,
            self.materialized_state("mean-of-means", device=x.device, dtype=x.dtype) as mean,
        ):
            return (x - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        with (
            self.materialized_state("std-of-means", device=x.device, dtype=x.dtype) as std,
            self.materialized_state("mean-of-means", device=x.device, dtype=x.dtype) as mean,
        ):
            return x * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)


class LTXVideoVAE(torch.nn.Module):
    """The reference's ``VideoVAE``: streaming encoder and decoder around
    normalized latents. ``encode`` returns the normalized latent means;
    ``decode`` mixes decode noise into the latent when the decoder is
    timestep conditioned (pass ``generator`` to pin that draw) and returns
    pixels in the reference's [-1, 1] convention."""

    def __init__(self, config: LTXVideoVAEConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.config = config
        self.encoder = LTXVideoEncoder(config, operations=operations)
        self.decoder = LTXVideoDecoder(config, operations=operations)
        self.per_channel_statistics = LTXPerChannelStatistics(config.latent_channels)

    def encode(self, x: torch.Tensor, *, max_chunk_bytes: int | None = None) -> torch.Tensor:
        ratio = self.config.temporal_ratio
        x = x[:, :, : max(1, 1 + ((x.shape[2] - 1) // ratio) * ratio)]
        means, _ = torch.chunk(self.encoder(x, max_chunk_bytes=max_chunk_bytes), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
        return self.decoder.decode_output_shape(input_shape)

    def decode(
        self,
        x: torch.Tensor,
        *,
        output_buffer: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        max_chunk_bytes: int | None = None,
    ) -> torch.Tensor:
        if self.config.timestep_conditioning:
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            scale = self.config.decode_noise_scale
            x = noise * scale + (1.0 - scale) * x
        return self.decoder(
            self.per_channel_statistics.un_normalize(x),
            timestep=self.config.decode_timestep,
            output_buffer=output_buffer,
            max_chunk_bytes=max_chunk_bytes,
        )
