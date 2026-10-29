"""Native torch Z-Image diffusion transformers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F
from dinkster_inference.z_image import Z_IMAGE_CONFIG, Z_IMAGE_PIXEL_CONFIG

from .attention import AttentionKernel, select_attention
from .flux import EmbedND, apply_rope
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, ResidencyRouted, materialized_rms_norm_weight

if TYPE_CHECKING:
    from .z_image_control import ZImageControl

_DEFAULT_ATTENTION = select_attention("flux").kernel


class _Config(Protocol):
    @property
    def family_id(self) -> str: ...
    @property
    def hidden_width(self) -> int: ...
    @property
    def caption_width(self) -> int: ...
    @property
    def main_blocks(self) -> int: ...
    @property
    def noise_refiner_blocks(self) -> int: ...
    @property
    def context_refiner_blocks(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def kv_heads(self) -> int: ...
    @property
    def attention_head_dim(self) -> int: ...
    @property
    def ffn_width(self) -> int: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def patch(self) -> tuple[int, int]: ...
    @property
    def rope_axes(self) -> tuple[int, int, int]: ...
    @property
    def rope_theta(self) -> float: ...
    @property
    def qk_norm_eps(self) -> float: ...
    @property
    def timestep_embedding_width(self) -> int: ...
    @property
    def modulation_width(self) -> int: ...
    @property
    def timestep_multiplier(self) -> float: ...
    @property
    def block_modulation_silu(self) -> bool: ...
    @property
    def pad_tokens_multiple(self) -> int: ...
    @property
    def learned_padding(self) -> bool: ...


class _PixelConfig(_Config, Protocol):
    @property
    def decoder_hidden_width(self) -> int: ...
    @property
    def decoder_blocks(self) -> int: ...
    @property
    def decoder_max_frequencies(self) -> int: ...


def _clamp_fp16(value: torch.Tensor) -> torch.Tensor:
    if value.dtype == torch.float16:
        return torch.nan_to_num(value, nan=0.0, posinf=65504, neginf=-65504)
    return value


def z_image_timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    arguments = t[:, None].float() * frequencies[None]
    return torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1).to(t.dtype)


class ZImageTimestepEmbedder(torch.nn.Module):
    def __init__(
        self,
        embedding_width: int,
        output_width: int,
        *,
        hidden_width: int = 1024,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.embedding_width = embedding_width
        self.mlp = torch.nn.Sequential(
            operations.linear(embedding_width, hidden_width),
            torch.nn.SiLU(),
            operations.linear(hidden_width, output_width),
        )

    def forward(self, timestep: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.mlp(z_image_timestep_embedding(timestep, self.embedding_width).to(dtype=dtype))


class ZImageAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_width
        self.heads = config.attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.attention_head_dim
        self.qk_norm_eps = config.qk_norm_eps
        self.fused_qk_rope = config.family_id == "dinkster.lumina2"
        self.qkv = operations.linear(
            hidden, (self.heads + 2 * self.kv_heads) * self.head_dim, bias=False
        )
        self.out = operations.linear(self.heads * self.head_dim, hidden, bias=False)
        qk_eps = None if self.fused_qk_rope else config.qk_norm_eps
        self.q_norm = operations.rms_norm(self.head_dim, eps=qk_eps)
        self.k_norm = operations.rms_norm(self.head_dim, eps=qk_eps)
        self.fused_qk_norm_eps = (
            self.q_norm.eps if self.q_norm.eps is not None else torch.finfo(torch.float32).eps
        )
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).split(
            (
                self.heads * self.head_dim,
                self.kv_heads * self.head_dim,
                self.kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.kv_heads, self.head_dim)
        v = v.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.fused_qk_rope and not torch.is_grad_enabled():
            import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

            with (
                materialized_rms_norm_weight(self.q_norm) as q_scale,
                materialized_rms_norm_weight(self.k_norm) as k_scale,
            ):
                kitchen_rope = rope.movedim(1, 2)
                if self.heads == self.kv_heads:
                    q, k = dinkster_kitchen.rms_rope(
                        q,
                        k,
                        kitchen_rope,
                        q_scale.detach(),
                        k_scale.detach(),
                        epsilon=self.fused_qk_norm_eps,
                    )
                else:
                    q = dinkster_kitchen.rms_rope1(
                        q,
                        kitchen_rope,
                        q_scale.detach(),
                        epsilon=self.fused_qk_norm_eps,
                    )
                    k = dinkster_kitchen.rms_rope1(
                        k,
                        kitchen_rope,
                        k_scale.detach(),
                        epsilon=self.fused_qk_norm_eps,
                    )
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            q, k = apply_rope(q, k, rope)
        if self.heads != self.kv_heads:
            groups = self.heads // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        output = self._attention_kernel(q, k, v)
        return self.out(output.transpose(1, 2).reshape(batch, length, -1))


class ZImageFeedForward(torch.nn.Module):
    def __init__(self, config: _Config, *, operations: Operations) -> None:
        super().__init__()
        self.w1 = operations.linear(config.hidden_width, config.ffn_width, bias=False)
        self.w2 = operations.linear(config.ffn_width, config.hidden_width, bias=False)
        self.w3 = operations.linear(config.hidden_width, config.ffn_width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(_clamp_fp16(F.silu(self.w1(x)) * self.w3(x)))


class ZImageBlock(torch.nn.Module):
    def __init__(
        self,
        config: _Config,
        *,
        modulated: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_width
        self.attention = ZImageAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.feed_forward = ZImageFeedForward(config, operations=operations)
        self.attention_norm1 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.attention_norm2 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.ffn_norm1 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.ffn_norm2 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        if not modulated:
            self.adaLN_modulation = None
        elif config.block_modulation_silu:
            self.adaLN_modulation = torch.nn.Sequential(
                torch.nn.SiLU(), operations.linear(config.modulation_width, 4 * hidden)
            )
        else:
            self.adaLN_modulation = torch.nn.Sequential(
                operations.linear(config.modulation_width, 4 * hidden)
            )

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor | None
    ) -> torch.Tensor:
        if self.adaLN_modulation is None:
            attended = _clamp_fp16(self.attention(self.attention_norm1(x), rope))
            x = x + self.attention_norm2(attended)
            return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        if modulation is None:
            raise ValueError("modulated Z-Image block requires a timestep embedding")
        scale_attn, gate_attn, scale_ffn, gate_ffn = self.adaLN_modulation(modulation).chunk(
            4, dim=1
        )
        attn_input = self.attention_norm1(x) * (1 + scale_attn.unsqueeze(1))
        x = x + gate_attn.unsqueeze(1).tanh() * self.attention_norm2(
            _clamp_fp16(self.attention(attn_input, rope))
        )
        ffn_input = self.ffn_norm1(x) * (1 + scale_ffn.unsqueeze(1))
        return x + gate_ffn.unsqueeze(1).tanh() * self.ffn_norm2(self.feed_forward(ffn_input))


class ZImageFinalLayer(torch.nn.Module):
    def __init__(self, config: _Config, *, operations: Operations) -> None:
        super().__init__()
        self.norm_final = operations.layer_norm(
            config.hidden_width, eps=1e-6, elementwise_affine=False
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), operations.linear(config.modulation_width, config.hidden_width)
        )
        self.linear = operations.linear(
            config.hidden_width, config.latent_channels * config.patch[0] * config.patch[1]
        )

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        scale = self.adaLN_modulation(modulation)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)))


class ZImage(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: _Config = Z_IMAGE_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        cfg: _Config = config
        self.config = cfg
        hidden = cfg.hidden_width
        patch_area = cfg.patch[0] * cfg.patch[1]
        self.x_embedder = operations.linear(cfg.latent_channels * patch_area, hidden)
        self.cap_embedder = torch.nn.Sequential(
            operations.rms_norm(cfg.caption_width, eps=cfg.qk_norm_eps),
            operations.linear(cfg.caption_width, hidden),
        )
        self.t_embedder = ZImageTimestepEmbedder(
            cfg.timestep_embedding_width,
            cfg.modulation_width,
            hidden_width=min(hidden, 1024),
            operations=operations,
        )
        self.context_refiner = torch.nn.ModuleList(
            ZImageBlock(
                cfg, modulated=False, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(cfg.context_refiner_blocks)
        )
        self.noise_refiner = torch.nn.ModuleList(
            ZImageBlock(
                cfg, modulated=True, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(cfg.noise_refiner_blocks)
        )
        self.layers = torch.nn.ModuleList(
            ZImageBlock(
                cfg, modulated=True, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(cfg.main_blocks)
        )
        self.final_layer = ZImageFinalLayer(cfg, operations=operations)
        if cfg.learned_padding:
            self.cap_pad_token = torch.nn.Parameter(torch.empty(1, hidden))
            self.x_pad_token = torch.nn.Parameter(torch.empty(1, hidden))
        self.rope_embedder = EmbedND(cfg.attention_head_dim, int(cfg.rope_theta), cfg.rope_axes)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        control: ZImageControl | None = None,
        control_latent: torch.Tensor | None = None,
        control_gains: tuple[float | torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if (control is None) != (control_latent is None):
            raise ValueError("Z-Image control and control latent must be supplied together")
        if control is not None and (control_gains is None or len(control_gains) != 6):
            raise ValueError("Z-Image control requires exactly six residual-site gains")
        if x.ndim != 4:
            raise ValueError(
                f"Z-Image latent must be [batch, channels, height, width], got {x.shape}"
            )
        batch, channels, height, width = x.shape
        if channels != cfg.latent_channels or min(batch, height, width) < 1:
            raise ValueError(
                f"Z-Image latent must have {cfg.latent_channels} channels and positive extents"
            )
        if timesteps.shape != (batch,) or timesteps.device != x.device:
            raise ValueError(f"timesteps must be [batch] = ({batch},) on the latent device")
        if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != cfg.caption_width:
            raise ValueError(
                f"context must be [batch, tokens, {cfg.caption_width}], got {context.shape}"
            )
        if context.shape[1] < 1 or context.device != x.device:
            raise ValueError("context must contain tokens on the latent device")
        patch_h, patch_w = cfg.patch
        pad_h, pad_w = (-height) % patch_h, (-width) % patch_w
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        padded_height, padded_width = x.shape[-2:]

        caption_length = context.shape[1]
        caption = self.cap_embedder(context)
        caption_pad = (-caption_length) % cfg.pad_tokens_multiple
        if caption_pad:
            binding = self._offloaded_residency()
            if binding is None:
                cap_pad_token = self.cap_pad_token.to(device=x.device, dtype=x.dtype)
                caption = torch.cat((caption, cap_pad_token.expand(batch, caption_pad, -1)), dim=1)
            else:
                with binding.lease() as lease:
                    cap_pad_token = lease.get("cap_pad_token", dtype=x.dtype)
                    caption = torch.cat(
                        (caption, cap_pad_token.expand(batch, caption_pad, -1)), dim=1
                    )
        padded_caption_length = caption.shape[1]

        image = x.view(
            batch, channels, padded_height // patch_h, patch_h, padded_width // patch_w, patch_w
        )
        image = image.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        image = self.x_embedder(image)
        image_length = image.shape[1]
        image_pad = (-image_length) % cfg.pad_tokens_multiple
        if image_pad:
            binding = self._offloaded_residency()
            if binding is None:
                x_pad_token = self.x_pad_token.to(device=x.device, dtype=x.dtype)
                image = torch.cat((image, x_pad_token.expand(batch, image_pad, -1)), dim=1)
            else:
                with binding.lease() as lease:
                    x_pad_token = lease.get("x_pad_token", dtype=x.dtype)
                    image = torch.cat((image, x_pad_token.expand(batch, image_pad, -1)), dim=1)

        caption_ids = torch.zeros(
            batch, padded_caption_length, 3, device=x.device, dtype=torch.float32
        )
        caption_ids[..., 0] = torch.arange(padded_caption_length, device=x.device) + 1
        image_ids = torch.zeros(batch, image.shape[1], 3, device=x.device, dtype=torch.float32)
        image_ids[:, :image_length, 0] = padded_caption_length + 1
        rows, columns = padded_height // patch_h, padded_width // patch_w
        image_ids[:, :image_length, 1] = (
            torch.arange(rows, device=x.device).view(-1, 1).expand(rows, columns).flatten()
        )
        image_ids[:, :image_length, 2] = (
            torch.arange(columns, device=x.device).view(1, -1).expand(rows, columns).flatten()
        )
        rope = self.rope_embedder(torch.cat((caption_ids, image_ids), dim=1))

        modulation = self.t_embedder(
            (1 - timesteps) * cfg.timestep_multiplier,
            dtype=x.dtype,
        )
        context_prefetch = make_prefetch_queue(self.context_refiner)
        try:
            for block in self.context_refiner:
                prefetch_queue_pop(context_prefetch, block)
                caption = block(caption, rope[:, :, :padded_caption_length], None)
            prefetch_queue_pop(context_prefetch, None)
        finally:
            close_prefetch_queue(context_prefetch)
        image_rope = rope[:, :, padded_caption_length:]
        noise_prefetch = make_prefetch_queue(self.noise_refiner)
        if control is not None:
            assert control_latent is not None
            control_context = control.embed(control_latent)
        else:
            control_context = None
        try:
            for index, block in enumerate(self.noise_refiner):
                prefetch_queue_pop(noise_prefetch, block)
                image = block(image, image_rope, modulation)
                if control is not None and index == 0:
                    assert control_context is not None
                    control_context = control.refine(
                        control_context,
                        image_rope[:, :, : control_context.shape[1]],
                        modulation,
                    )
            prefetch_queue_pop(noise_prefetch, None)
        finally:
            close_prefetch_queue(noise_prefetch)
        main_input = image
        combined = torch.cat((caption, image), dim=1)
        layer_prefetch = make_prefetch_queue(self.layers)
        try:
            control_index = 0
            for index, block in enumerate(self.layers):
                prefetch_queue_pop(layer_prefetch, block)
                combined = block(combined, rope, modulation)
                if control is not None and index in control.injection_blocks:
                    assert control_gains is not None
                    assert control_context is not None
                    image_start = padded_caption_length
                    image_end = image_start + control_context.shape[1]
                    residual, control_context = control.inject(
                        control_index,
                        control_context,
                        main_input[:, : control_context.shape[1]],
                        rope[:, :, image_start:image_end],
                        modulation,
                    )
                    assert residual is not None
                    combined[:, image_start : image_start + residual.shape[1]] += (
                        residual * control_gains[control_index]
                    )
                    control_index += 1
            prefetch_queue_pop(layer_prefetch, None)
        finally:
            close_prefetch_queue(layer_prefetch)
        output = self.final_layer(combined, modulation)
        output = output[:, padded_caption_length : padded_caption_length + image_length]
        output = output.view(batch, rows, columns, patch_h, patch_w, channels)
        output = output.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return -output[:, :, :height, :width]


class ZImageSplitAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_width
        self.heads = config.attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.attention_head_dim
        self.qk_norm_eps = config.qk_norm_eps
        self.to_q = operations.linear(hidden, self.heads * self.head_dim, bias=False)
        self.to_k = operations.linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.to_v = operations.linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.to_out = torch.nn.Sequential(
            operations.linear(self.heads * self.head_dim, hidden, bias=False)
        )
        self.norm_q = operations.rms_norm(self.head_dim, eps=config.qk_norm_eps)
        self.norm_k = operations.rms_norm(self.head_dim, eps=config.qk_norm_eps)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.to_q(x).view(batch, length, self.heads, self.head_dim)
        k = self.to_k(x).view(batch, length, self.kv_heads, self.head_dim)
        v = self.to_v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if torch.is_grad_enabled():
            q, k = apply_rope(
                self.norm_q(q).transpose(1, 2),
                self.norm_k(k).transpose(1, 2),
                rope,
            )
        else:
            import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

            with (
                materialized_rms_norm_weight(self.norm_q) as q_scale,
                materialized_rms_norm_weight(self.norm_k) as k_scale,
            ):
                q, k = dinkster_kitchen.rms_rope(
                    q,
                    k,
                    rope.movedim(1, 2),
                    q_scale.detach(),
                    k_scale.detach(),
                    epsilon=self.qk_norm_eps,
                )
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        if self.heads != self.kv_heads:
            groups = self.heads // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        output = self._attention_kernel(q, k, v)
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


class ZImagePixelBlock(ZImageBlock):
    def __init__(
        self,
        config: _Config,
        *,
        modulated: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        torch.nn.Module.__init__(self)
        hidden = config.hidden_width
        self.attention = ZImageSplitAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.feed_forward = ZImageFeedForward(config, operations=operations)
        self.attention_norm1 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.attention_norm2 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.ffn_norm1 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.ffn_norm2 = operations.rms_norm(hidden, eps=config.qk_norm_eps)
        self.adaLN_modulation = (
            torch.nn.Sequential(operations.linear(config.modulation_width, 4 * hidden))
            if modulated
            else None
        )


class ZImagePixelEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_width: int,
        max_frequencies: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.max_frequencies = max_frequencies
        self.embedder = torch.nn.Sequential(
            operations.linear(in_channels + max_frequencies**2, hidden_width)
        )

    def _positions(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        position_y, position_x = torch.meshgrid(position, position, indexing="ij")
        position_x = position_x.reshape(-1, 1, 1)
        position_y = position_y.reshape(-1, 1, 1)
        frequencies = torch.linspace(
            0,
            self.max_frequencies - 1,
            self.max_frequencies,
            device=device,
            dtype=dtype,
        )
        frequencies_x = frequencies[None, :, None]
        frequencies_y = frequencies[None, None, :]
        coefficients = (1 + frequencies_x * frequencies_y) ** -1
        dct_x = torch.cos(position_x * frequencies_x * torch.pi)
        dct_y = torch.cos(position_y * frequencies_y * torch.pi)
        return (dct_x * dct_y * coefficients).view(1, -1, self.max_frequencies**2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, positions, _ = inputs.shape
        patch_size = int(positions**0.5)
        weight_dtype = cast(torch.nn.Linear, self.embedder[0]).weight.dtype
        values = inputs.to(dtype=weight_dtype)
        dct = self._positions(patch_size, values.device, weight_dtype).expand(batch, -1, -1)
        return self.embedder(torch.cat((values, dct), dim=-1)).to(dtype=inputs.dtype)


class ZImagePixelResBlock(torch.nn.Module):
    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.in_ln = operations.layer_norm(channels, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            operations.linear(channels, channels),
            torch.nn.SiLU(),
            operations.linear(channels, channels),
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), operations.linear(channels, 3 * channels)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        hidden = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(hidden)


class ZImagePixelFinalLayer(torch.nn.Module):
    def __init__(self, hidden: int, output: int, *, operations: Operations) -> None:
        super().__init__()
        self.norm_final = operations.layer_norm(hidden, eps=1e-6, elementwise_affine=False)
        self.linear = operations.linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class ZImagePixelDecoder(torch.nn.Module):
    def __init__(self, config: _PixelConfig, *, operations: Operations) -> None:
        super().__init__()
        patch_values = config.latent_channels * config.patch[0] * config.patch[1]
        hidden = config.decoder_hidden_width
        self.cond_embed = operations.linear(config.hidden_width, hidden)
        self.input_embedder = ZImagePixelEmbedder(
            patch_values,
            hidden,
            config.decoder_max_frequencies,
            operations=operations,
        )
        self.res_blocks = torch.nn.ModuleList(
            ZImagePixelResBlock(hidden, operations=operations) for _ in range(config.decoder_blocks)
        )
        self.final_layer = ZImagePixelFinalLayer(hidden, patch_values, operations=operations)

    def forward(self, pixels: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        original_dtype = pixels.dtype
        weight_dtype = self.cond_embed.weight.dtype
        pixels = self.input_embedder(pixels).to(dtype=weight_dtype)
        condition = self.cond_embed(condition.to(dtype=weight_dtype)).unsqueeze(1)
        for block in self.res_blocks:
            pixels = block(pixels, condition)
        return self.final_layer(pixels).to(dtype=original_dtype)


class ZImagePixelSpace(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: _PixelConfig = Z_IMAGE_PIXEL_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_width
        patch_area = config.patch[0] * config.patch[1]
        self.x_embedder = operations.linear(config.latent_channels * patch_area, hidden)
        self.cap_embedder = torch.nn.Sequential(
            operations.rms_norm(config.caption_width, eps=config.qk_norm_eps),
            operations.linear(config.caption_width, hidden),
        )
        self.t_embedder = ZImageTimestepEmbedder(
            config.timestep_embedding_width,
            config.modulation_width,
            hidden_width=min(hidden, 1024),
            operations=operations,
        )
        self.context_refiner = torch.nn.ModuleList(
            ZImagePixelBlock(
                config, modulated=False, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(config.context_refiner_blocks)
        )
        self.noise_refiner = torch.nn.ModuleList(
            ZImagePixelBlock(
                config, modulated=True, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(config.noise_refiner_blocks)
        )
        self.layers = torch.nn.ModuleList(
            ZImagePixelBlock(
                config, modulated=True, operations=operations, attention_kernel=attention_kernel
            )
            for _ in range(config.main_blocks)
        )
        self.dec_net = ZImagePixelDecoder(config, operations=operations)
        if config.learned_padding:
            self.cap_pad_token = torch.nn.Parameter(torch.empty(1, hidden))
            self.x_pad_token = torch.nn.Parameter(torch.empty(1, hidden))
        self.rope_embedder = EmbedND(
            config.attention_head_dim, int(config.rope_theta), config.rope_axes
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        if x.ndim != 4:
            raise ValueError(f"Z-Image pixel input must be [batch,3,height,width], got {x.shape}")
        batch, channels, height, width = x.shape
        if channels != cfg.latent_channels or min(batch, height, width) < 1:
            raise ValueError("Z-Image pixel input must have 3 channels and positive extents")
        if timesteps.shape != (batch,) or timesteps.device != x.device:
            raise ValueError(f"timesteps must be [batch] = ({batch},) on the input device")
        if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != cfg.caption_width:
            raise ValueError(
                f"context must be [batch, tokens, {cfg.caption_width}], got {context.shape}"
            )
        if context.shape[1] < 1 or context.device != x.device:
            raise ValueError("context must contain tokens on the input device")
        patch_h, patch_w = cfg.patch
        padded = F.pad(x, (0, (-width) % patch_w, 0, (-height) % patch_h), mode="circular")
        padded_height, padded_width = padded.shape[-2:]

        caption_length = context.shape[1]
        caption = self.cap_embedder(context)
        caption_pad = (-caption_length) % cfg.pad_tokens_multiple
        if caption_pad:
            binding = self._offloaded_residency()
            if binding is None:
                cap_pad_token = self.cap_pad_token.to(device=x.device, dtype=x.dtype)
                caption = torch.cat((caption, cap_pad_token.expand(batch, caption_pad, -1)), dim=1)
            else:
                with binding.lease() as lease:
                    cap_pad_token = lease.get("cap_pad_token", dtype=x.dtype)
                    caption = torch.cat(
                        (caption, cap_pad_token.expand(batch, caption_pad, -1)), dim=1
                    )
        padded_caption_length = caption.shape[1]

        rows, columns = padded_height // patch_h, padded_width // patch_w
        pixel_values = padded.view(batch, channels, rows, patch_h, columns, patch_w)
        pixel_values = pixel_values.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        image_length = pixel_values.shape[1]
        image = self.x_embedder(pixel_values)
        image_pad = (-image_length) % cfg.pad_tokens_multiple
        if image_pad:
            binding = self._offloaded_residency()
            if binding is None:
                x_pad_token = self.x_pad_token.to(device=x.device, dtype=x.dtype)
                image = torch.cat((image, x_pad_token.expand(batch, image_pad, -1)), dim=1)
            else:
                with binding.lease() as lease:
                    x_pad_token = lease.get("x_pad_token", dtype=x.dtype)
                    image = torch.cat((image, x_pad_token.expand(batch, image_pad, -1)), dim=1)

        caption_ids = torch.zeros(
            batch, padded_caption_length, 3, device=x.device, dtype=torch.float32
        )
        caption_ids[..., 0] = torch.arange(padded_caption_length, device=x.device) + 1
        image_ids = torch.zeros(batch, image.shape[1], 3, device=x.device, dtype=torch.float32)
        image_ids[:, :image_length, 0] = padded_caption_length + 1
        image_ids[:, :image_length, 1] = (
            torch.arange(rows, device=x.device).view(-1, 1).expand(rows, columns).flatten()
        )
        image_ids[:, :image_length, 2] = (
            torch.arange(columns, device=x.device).view(1, -1).expand(rows, columns).flatten()
        )
        rope = self.rope_embedder(torch.cat((caption_ids, image_ids), dim=1))
        modulation = self.t_embedder(
            (1 - timesteps) * cfg.timestep_multiplier,
            dtype=x.dtype,
        )

        context_prefetch = make_prefetch_queue(self.context_refiner)
        try:
            for block in self.context_refiner:
                prefetch_queue_pop(context_prefetch, block)
                caption = block(caption, rope[:, :, :padded_caption_length], None)
            prefetch_queue_pop(context_prefetch, None)
        finally:
            close_prefetch_queue(context_prefetch)
        image_rope = rope[:, :, padded_caption_length:]
        noise_prefetch = make_prefetch_queue(self.noise_refiner)
        try:
            for block in self.noise_refiner:
                prefetch_queue_pop(noise_prefetch, block)
                image = block(image, image_rope, modulation)
            prefetch_queue_pop(noise_prefetch, None)
        finally:
            close_prefetch_queue(noise_prefetch)
        combined = torch.cat((caption, image), dim=1)
        layer_prefetch = make_prefetch_queue(self.layers)
        try:
            for block in self.layers:
                prefetch_queue_pop(layer_prefetch, block)
                combined = block(combined, rope, modulation)
            prefetch_queue_pop(layer_prefetch, None)
        finally:
            close_prefetch_queue(layer_prefetch)

        image_hidden = combined[:, padded_caption_length : padded_caption_length + image_length]
        decoded = self.dec_net(
            pixel_values.reshape(batch * image_length, 1, -1),
            image_hidden.reshape(batch * image_length, cfg.hidden_width),
        ).reshape(batch, image_length, -1)
        decoded = decoded.view(batch, rows, columns, patch_h, patch_w, channels)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        negative_x0 = -decoded[:, :, :height, :width]
        return (x - negative_x0) / timesteps.view(-1, 1, 1, 1)


class ZImagePixelCodec(torch.nn.Module):
    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return content

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class PixelSpaceCodec(torch.nn.Module):
    def __init__(self, *, compute_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return (content * 2.0 - 1.0).to(self.compute_dtype).to(torch.float32, copy=True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return (
            latent.to(self.compute_dtype)
            .to(torch.float32, copy=True)
            .add_(1.0)
            .div_(2.0)
            .clamp_(0.0, 1.0)
        )


__all__ = [
    "ZImage",
    "ZImageAttention",
    "ZImageBlock",
    "ZImageFeedForward",
    "ZImageFinalLayer",
    "PixelSpaceCodec",
    "ZImageTimestepEmbedder",
    "ZImagePixelSpace",
    "ZImagePixelCodec",
    "z_image_timestep_embedding",
]
