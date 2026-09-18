"""LTX-2.5 neighborhood-attention diffusion video VAE."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from dinkster_inference import LTXAV_22B_V25_VAE_CONFIG, LTXDiffusionVideoVAEConfig

from .ltx_video_vae import LTXPerChannelStatistics, LTXVideoEncoder
from .operations import INITLESS, Operations, ResidencyRouted, materialized_rms_norm_weight

MLP_TOKEN_CHUNK = 65536


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == patch_size_t == 1:
        return x
    b, c, f, h, w = x.shape
    x = x.reshape(
        b,
        c,
        f // patch_size_t,
        patch_size_t,
        h // patch_size_hw,
        patch_size_hw,
        w // patch_size_hw,
        patch_size_hw,
    )
    return x.permute(0, 1, 3, 7, 5, 2, 4, 6).reshape(
        b,
        c * patch_size_t * patch_size_hw**2,
        f // patch_size_t,
        h // patch_size_hw,
        w // patch_size_hw,
    )


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == patch_size_t == 1:
        return x
    b, packed, f, h, w = x.shape
    c = packed // (patch_size_t * patch_size_hw**2)
    x = x.reshape(b, c, patch_size_t, patch_size_hw, patch_size_hw, f, h, w)
    return x.permute(0, 1, 5, 2, 6, 4, 7, 3).reshape(
        b, c, f * patch_size_t, h * patch_size_hw, w * patch_size_hw
    )


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _rope_inv_freqs(dim: int, device: torch.device, base: float = 10000.0) -> torch.Tensor:
    output_device = device
    if device.type in ("mps", "xpu"):
        device = torch.device("cpu")
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    return (1.0 / torch.pow(torch.tensor(base, dtype=torch.float64, device=device), exponents)).to(
        dtype=torch.float32, device=output_device
    )


def _rope_tables(
    t: int, h: int, w: int, split: tuple[int, int, int], device: torch.device
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    tables = []
    for length, dim in zip((t, h, w), split, strict=True):
        positions = torch.arange(length, dtype=torch.float32, device=device)[:, None]
        angles = positions * _rope_inv_freqs(dim, device)[None]
        tables.append((angles.cos(), angles.sin()))
    return tuple(tables)


def _rope_table_slice(
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    t0: int,
    t1: int,
    h: int,
    w: int,
) -> torch.Tensor:
    parts = []
    for (cosine, sine), selection in zip(
        tables, (slice(t0, t1), slice(None), slice(None)), strict=True
    ):
        cosine, sine = cosine[selection], sine[selection]
        parts.append(
            torch.stack((cosine, -sine, sine, cosine), -1).reshape(
                cosine.shape[0], 1, 1, cosine.shape[1], 2, 2
            )
        )
    t = t1 - t0
    return torch.cat(
        (
            parts[0].expand(t, h, w, -1, -1, -1),
            parts[1].transpose(0, 1).expand(t, h, w, -1, -1, -1),
            parts[2].movedim(0, 2).expand(t, h, w, -1, -1, -1),
        ),
        3,
    ).reshape(1, t * h * w, 1, -1, 2, 2)


def _rope_fallback(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    pairs = x.reshape(shape[0], -1, shape[-2], shape[-1] // 2, 2).float()
    out = table[..., 0] * pairs[..., 0, None] + table[..., 1] * pairs[..., 1, None]
    return out.reshape(shape).to(x.dtype)


class NeighborhoodAttention3D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, dim // head_dim, head_dim
        self.kernel_size, self.scale = kernel_size, head_dim**-0.5
        self.qkv = operations.linear(dim, dim * 3)
        self.proj = operations.linear(dim, dim)
        self.q_norm = operations.rms_norm(head_dim, eps=1e-6)
        self.k_norm = operations.rms_norm(head_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        pre: Callable[[torch.Tensor], torch.Tensor] | None = None,
        add_to: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]
        import comfy_kitchen.backends.eager.na as eager_na  # pyright: ignore[reportMissingTypeStubs]

        b, t, h, w, _ = x.shape
        shape = (b, t, h, w, self.num_heads, self.head_dim)
        gradient_mode = torch.is_grad_enabled()
        q_chunks: list[torch.Tensor] = []
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        q: torch.Tensor | None = None
        k: torch.Tensor | None = None
        v: torch.Tensor | None = None
        if not gradient_mode:
            q = torch.empty(shape, dtype=x.dtype, device=x.device)
            k = torch.empty(shape, dtype=x.dtype, device=x.device)
            v = torch.empty(shape, dtype=x.dtype, device=x.device)
        tables = _rope_tables(t, h, w, default_rope_dim_split(self.head_dim), x.device)
        chunk = max(1, 2**25 // max(h * w * self.dim, 1))
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            source = x[:, t0:t1]
            qc, kc, vc = self.qkv(source if pre is None else pre(source)).chunk(3, -1)
            chunk_shape = (b, t1 - t0, h, w, self.num_heads, self.head_dim)
            table = _rope_table_slice(tables, t0, t1, h, w)
            if gradient_mode:
                q_chunks.append(
                    _rope_fallback(self.q_norm(qc.reshape(chunk_shape)), table) * self.scale
                )
                k_chunks.append(_rope_fallback(self.k_norm(kc.reshape(chunk_shape)), table))
                v_chunks.append(vc.reshape(chunk_shape))
                continue
            assert q is not None and k is not None and v is not None
            q[:, t0:t1] = qc.reshape(chunk_shape)
            k[:, t0:t1] = kc.reshape(chunk_shape)
            v[:, t0:t1] = vc.reshape(chunk_shape)
            with (
                materialized_rms_norm_weight(self.q_norm) as qw,
                materialized_rms_norm_weight(self.k_norm) as kw,
            ):
                comfy_kitchen.rms_rope_(
                    q[:, t0:t1].reshape(b, -1, self.num_heads, self.head_dim),
                    k[:, t0:t1].reshape(b, -1, self.num_heads, self.head_dim),
                    table,
                    (qw.detach() * self.scale).to(q.dtype),
                    kw.detach().to(k.dtype),
                )
        if gradient_mode:
            q, k, v = torch.cat(q_chunks, 1), torch.cat(k_chunks, 1), torch.cat(v_chunks, 1)
            attended = eager_na.na3d(q, k, v, list(self.kernel_size), None, 1.0)
        else:
            assert q is not None and k is not None and v is not None
            attended = comfy_kitchen.na3d(q, k, v, list(self.kernel_size), None, 1.0)
        attended = attended.reshape(b, t, h, w, self.dim)
        del q, k, v
        if gradient_mode:
            projected = torch.cat(
                tuple(self.proj(attended[:, t0 : t0 + chunk]) for t0 in range(0, t, chunk)),
                dim=1,
            )
            return projected if add_to is None else add_to + projected
        result = torch.empty_like(attended) if add_to is None else add_to
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            projected = self.proj(attended[:, t0:t1])
            if add_to is None:
                result[:, t0:t1] = projected
            else:
                result[:, t0:t1].add_(projected)
        return result


class SwiGLU(torch.nn.Module):
    def __init__(self, dim: int, hidden: int, operations: Operations) -> None:
        super().__init__()
        self.w_up = operations.linear(dim, hidden, bias=False)
        self.w_gate = operations.linear(dim, hidden, bias=False)
        self.w_down = operations.linear(hidden, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pre: Callable[[torch.Tensor], torch.Tensor] | None = None,
        add_to: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, t, h, w, _ = x.shape
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        pieces = [] if torch.is_grad_enabled() else None
        result = torch.empty_like(x) if pieces is None and add_to is None else add_to
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            source = x[:, t0:t1]
            source = source if pre is None else pre(source)
            value = self.w_down(F.silu(self.w_gate(source)) * self.w_up(source))
            if pieces is not None:
                pieces.append(value)
            elif add_to is None:
                assert result is not None
                result[:, t0:t1] = value
            else:
                add_to[:, t0:t1].add_(value)
        if pieces is not None:
            value = torch.cat(pieces, dim=1)
            return value if add_to is None else add_to + value
        assert result is not None
        return result


class NABlock(torch.nn.Module):
    def __init__(
        self, dim: int, kernel: tuple[int, int, int], head_dim: int, operations: Operations
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = (
            operations.rms_norm(dim, eps=1e-6),
            operations.rms_norm(dim, eps=1e-6),
        )
        self.attn = NeighborhoodAttention3D(dim, kernel, head_dim, operations)
        self.mlp = SwiGLU(dim, (int(dim * 4.0) + 15) // 16 * 16, operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, self.norm1, add_to=x)
        return self.mlp(x, self.norm2, add_to=x)


class AdaLNZero(torch.nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_dim: int, operations: Operations) -> None:
        super().__init__()
        self.proj = operations.linear(t_dim, self.NUM_CHUNKS * dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            v[:, None, None, None] for v in self.proj(F.silu(x)).chunk(self.NUM_CHUNKS, -1)
        )


class DiffusionNABlock(ResidencyRouted, torch.nn.Module):
    def __init__(
        self, dim: int, kernel: tuple[int, int, int], head_dim: int, operations: Operations
    ) -> None:
        super().__init__()
        self.context_proj = operations.linear(dim, dim)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1, self.norm2 = (
            operations.rms_norm(dim, eps=1e-6),
            operations.rms_norm(dim, eps=1e-6),
        )
        self.attn = NeighborhoodAttention3D(dim, kernel, head_dim, operations)
        self.mlp = SwiGLU(dim, (dim * 4 + 15) // 16 * 16, operations)

    @contextmanager
    def _table(self) -> Generator[torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            yield self.scale_shift_table
        else:
            with binding.lease() as lease:
                yield lease.get("scale_shift_table", dtype=self.scale_shift_table.dtype)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, modulation: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        with self._table() as table:
            values = [modulation[i] + table[i].reshape(1, 1, 1, 1, -1) for i in range(7)]
        scale_a, shift_a, _, scale_m, shift_m, _, _ = values
        chunk = max(1, MLP_TOKEN_CHUNK // max(x.shape[2] * x.shape[3], 1))
        if torch.is_grad_enabled():
            x = x + torch.cat(
                tuple(
                    self.context_proj(context[:, t0 : t0 + chunk])
                    for t0 in range(0, x.shape[1], chunk)
                ),
                dim=1,
            )
        else:
            for t0 in range(0, x.shape[1], chunk):
                x[:, t0 : t0 + chunk].add_(self.context_proj(context[:, t0 : t0 + chunk]))

        def norm_a(value: torch.Tensor) -> torch.Tensor:
            return self.norm1(value) * (1 + scale_a) + shift_a

        def norm_m(value: torch.Tensor) -> torch.Tensor:
            return self.norm2(value) * (1 + scale_m) + shift_m

        x = self.attn(x, norm_a, add_to=x)
        return self.mlp(x, norm_m, add_to=x)


class LinearPixelShuffleUpsample(torch.nn.Module):
    def __init__(
        self, channels: int, stride: tuple[int, int, int], reduction: int, operations: Operations
    ) -> None:
        super().__init__()
        self.stride = stride
        expanded = math.prod(stride) * channels // reduction
        self.out_channels = expanded // math.prod(stride)
        self.proj = operations.linear(channels, expanded)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        out = torch.empty(
            (b, t * p1, h * p2, w * p3, self.out_channels),
            dtype=x.dtype,
            device=x.device,
        )
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        pieces = [] if torch.is_grad_enabled() else None
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            projected = self.proj(x[:, t0:t1]).reshape(
                b, t1 - t0, h, w, self.out_channels, p1, p2, p3
            )
            projected = projected.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
                b, (t1 - t0) * p1, h * p2, w * p3, self.out_channels
            )
            if pieces is None:
                out[:, t0 * p1 : t1 * p1] = projected
            else:
                pieces.append(projected)
        if pieces is not None:
            out = torch.cat(pieces, dim=1)
        return out[:, 1:] if p1 == 2 and drop_leading_frame else out


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.flatten().float()[:, None] * frequencies[None]
    return torch.cat((args.cos(), args.sin()), -1)


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, dim: int, operations: Operations) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            operations.linear(256, dim), torch.nn.SiLU(), operations.linear(dim, dim)
        )

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(_timestep_embedding(t, 256).to(dtype))


class NADiffusionDecoder(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = (
            (3, 7, 7),
            (3, 7, 7),
            (3, 5, 5),
            (3, 5, 5),
            (11, 11, 11),
        ),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
            ((1, 2, 2), 2),
            ((2, 1, 1), 2),
            ((2, 2, 2), 1),
            ((2, 2, 2), 2),
        ),
        t_emb_dim: int = 384,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.patch_size, self.out_channels = patch_size, out_channels
        self.temporal_upscale = math.prod(v[0][0] for v in upsamples)
        self.trailing_pad_latent_frames = stage_kernels[0][0] // 2 * 2
        self.conv_in = operations.linear(in_channels, stage_channels[0])
        self.det_stages = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        NABlock(stage_channels[i], stage_kernels[i], head_dim, operations)
                        for _ in range(stage_depths[i])
                    ]
                )
                for i in range(4)
            ]
        )
        self.upsamples = torch.nn.ModuleList(
            [
                LinearPixelShuffleUpsample(stage_channels[i], stride, reduction, operations)
                for i, (stride, reduction) in enumerate(upsamples)
            ]
        )
        self.t_embedder = TimestepEmbedder(t_emb_dim, operations)
        c5, pixel_channels = stage_channels[-1], out_channels * patch_size**2
        self.conv_in_x_t = operations.linear(pixel_channels, c5)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim, operations)
        self.diff_blocks = torch.nn.ModuleList(
            [
                DiffusionNABlock(c5, stage_kernels[-1], head_dim, operations)
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = operations.rms_norm(c5, eps=1e-6)
        self.conv_out = operations.linear(c5, pixel_channels)
        self.register_buffer("default_inference_timesteps", torch.ones(1), persistent=False)

    def forward_pre_diffusion(
        self,
        z: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        n = self.trailing_pad_latent_frames if pad_trailing else 0
        if n:
            z = torch.cat((z, z[:, :, -1:].expand(-1, -1, n, -1, -1)), 2)
        x = self.conv_in(z.permute(0, 2, 3, 4, 1))
        for blocks, upsample in zip(self.det_stages, self.upsamples, strict=True):
            for block in blocks.children():
                x = block(x)
            x = upsample(x, drop_leading_frame=drop_leading_frame)
        return x[:, : -(n * self.temporal_upscale)] if n else x

    def forward_diff_step(
        self, context: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_in_x_t(patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1))
        modulation = self.shared_adaln(self.t_embedder(1000.0 * t, x.dtype))
        for block in self.diff_blocks:
            x = block(x, context, modulation)
        return unpatchify(self.conv_out(self.norm_out(x)).permute(0, 4, 1, 2, 3), self.patch_size)

    def forward(
        self,
        z: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        context = self.forward_pre_diffusion(
            z,
            drop_leading_frame=drop_leading_frame,
            pad_trailing=pad_trailing,
        )
        b, t, h, w, _ = context.shape
        x_t = torch.randn(
            (b, self.out_channels, t, h * self.patch_size, w * self.patch_size),
            device=z.device,
            dtype=z.dtype,
            generator=generator,
        )
        timesteps = self.get_buffer("default_inference_timesteps").to(device=z.device)
        return self.forward_diff_step(context, x_t, timesteps.expand(b))


class LTXDiffusionVideoVAE(torch.nn.Module):
    """Concrete LTX-2.5 codec with the standard encode/decode facade."""

    def __init__(
        self,
        config: LTXDiffusionVideoVAEConfig = LTXAV_22B_V25_VAE_CONFIG,
        *,
        operations: Operations = INITLESS,
        decoder: NADiffusionDecoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = LTXVideoEncoder(config.encoder_config, operations=operations)
        self.decoder = (
            NADiffusionDecoder(
                in_channels=config.latent_channels,
                out_channels=config.output_channels,
                patch_size=config.patch_size,
                head_dim=config.head_dim,
                stage_channels=config.stage_channels,
                stage_depths=config.stage_depths,
                stage_kernels=config.stage_kernels,
                upsamples=config.upsamples,
                t_emb_dim=config.timestep_dim,
                operations=operations,
            )
            if decoder is None
            else decoder
        )
        self.per_channel_statistics = LTXPerChannelStatistics(config.latent_channels)

    def encode(self, x: torch.Tensor, *, max_chunk_bytes: int | None = None) -> torch.Tensor:
        ratio = math.prod(
            block.stride[0]
            for block in self.config.encoder_blocks
            if block.kind.startswith("compress_")
        )
        x = x[:, :, : max(1, 1 + (x.shape[2] - 1) // ratio * ratio)]
        means, _ = self.encoder(x, max_chunk_bytes=max_chunk_bytes).chunk(2, 1)
        return self.per_channel_statistics.normalize(means)

    def decode_output_shape(self, shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
        b, _, t, h, w = shape
        temporal = math.prod(stride[0] for stride, _ in self.config.upsamples)
        spatial = (
            math.prod(stride[1] for stride, _ in self.config.upsamples) * self.config.patch_size
        )
        return b, self.config.output_channels, 1 + (t - 1) * temporal, h * spatial, w * spatial

    def decode(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator(device=x.device).manual_seed(0)
        return self.decoder(
            self.per_channel_statistics.un_normalize(x),
            generator=generator,
            drop_leading_frame=drop_leading_frame,
            pad_trailing=pad_trailing,
        )
