"""Native torch Krea 2 single-stream diffusion transformer.

The reference is comfy/ldm/krea2/model.py SingleStreamDiT @ b78cec87:
text tokens produced by a text-fusion adapter over stacked Qwen3-VL
layer taps and patchified image tokens are concatenated into one
sequence and run through shared transformer blocks with AdaLN-single
modulation (a shared per-step projection plus per-block bias), GQA
attention with per-head QK RMSNorm and a sigmoid output gate, SwiGLU
MLPs, and 3-axis RoPE. State-dict keys match the reference exactly.

Krea 2 spells its RMS scales zero-centered: the stored ``.scale``
parameter is added to 1.0 and the normalization runs in float32
regardless of the compute dtype, exactly as the reference RMSNorm.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from dinkster_inference import KREA2_CONFIG

from .attention import AttentionKernel, select_attention
from .flux import EmbedND, apply_rope, flux_timestep_embedding
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("flux").kernel


class _Config(Protocol):
    @property
    def features(self) -> int: ...
    @property
    def transformer_blocks(self) -> int: ...
    @property
    def attention_heads(self) -> int: ...
    @property
    def kv_heads(self) -> int: ...
    @property
    def mlp_width(self) -> int: ...
    @property
    def time_width(self) -> int: ...
    @property
    def text_width(self) -> int: ...
    @property
    def text_layers(self) -> int: ...
    @property
    def text_heads(self) -> int: ...
    @property
    def text_kv_heads(self) -> int: ...
    @property
    def text_mlp_width(self) -> int: ...
    @property
    def text_fusion_layerwise_blocks(self) -> int: ...
    @property
    def text_fusion_refiner_blocks(self) -> int: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def patch(self) -> tuple[int, int]: ...
    @property
    def rope_axes(self) -> tuple[int, int, int]: ...
    @property
    def rope_theta(self) -> float: ...
    @property
    def rms_norm_eps(self) -> float: ...


class Krea2RMSNorm(ResidencyRouted, torch.nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention
    (scale stored zero-centered), computed in float32."""

    def __init__(self, features: int, *, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.empty(features))

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return torch.float32

    def _normalize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        weight = scale + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._normalize(x, cast_weight(self.scale, dtype=torch.float32, device=x.device))
        with binding.lease() as lease:
            return self._normalize(x, lease.get("scale", dtype=torch.float32))


class Krea2QKNorm(torch.nn.Module):
    def __init__(self, head_dim: int, *, eps: float) -> None:
        super().__init__()
        self.qnorm = Krea2RMSNorm(head_dim, eps=eps)
        self.knorm = Krea2RMSNorm(head_dim, eps=eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qnorm(q), self.knorm(k)


class Krea2Attention(torch.nn.Module):
    """GQA attention with per-head QK RMSNorm and a sigmoid output
    gate: ``wo(attention * sigmoid(gate(x)))``."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        *,
        eps: float,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.wq = operations.linear(width, width, bias=False)
        self.wk = operations.linear(width, self.kv_heads * self.head_dim, bias=False)
        self.wv = operations.linear(width, self.kv_heads * self.head_dim, bias=False)
        self.gate = operations.linear(width, width, bias=False)
        self.qknorm = Krea2QKNorm(self.head_dim, eps=eps)
        self.wo = operations.linear(width, width, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        gate = self.gate(x)
        q = self.wq(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.qknorm(q, k)
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        if self.kv_heads != self.heads:
            groups = self.heads // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        output = self._attention_kernel(q, k, v)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.wo(output * torch.sigmoid(gate))


class Krea2SwiGLU(torch.nn.Module):
    def __init__(self, width: int, mlp_width: int, *, operations: Operations) -> None:
        super().__init__()
        self.gate = operations.linear(width, mlp_width, bias=False)
        self.up = operations.linear(width, mlp_width, bias=False)
        self.down = operations.linear(mlp_width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Krea2SharedModulation(ResidencyRouted, torch.nn.Module):
    """Per-block additive bias over the shared six-way timestep
    projection; the sum chunks into the six modulation legs."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = torch.nn.Parameter(torch.empty(6 * width))

    def forward(
        self, vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            lin = cast_weight(self.lin, dtype=vec.dtype, device=vec.device)
            prescale, preshift, pregate, postscale, postshift, postgate = (vec + lin).chunk(
                6, dim=-1
            )
        else:
            with binding.lease() as lease:
                lin = lease.get("lin", dtype=vec.dtype)
                prescale, preshift, pregate, postscale, postshift, postgate = (vec + lin).chunk(
                    6, dim=-1
                )
        return prescale, preshift, pregate, postscale, postshift, postgate


class Krea2FinalModulation(ResidencyRouted, torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = torch.nn.Parameter(torch.empty(2, width))

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            lin = cast_weight(self.lin, dtype=vec.dtype, device=vec.device)
            scale, shift = (vec + lin.unsqueeze(0)).chunk(2, dim=1)
        else:
            with binding.lease() as lease:
                lin = lease.get("lin", dtype=vec.dtype)
                scale, shift = (vec + lin.unsqueeze(0)).chunk(2, dim=1)
        return scale, shift


class Krea2FusionBlock(torch.nn.Module):
    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        text = config.text_width
        eps = config.rms_norm_eps
        self.prenorm = Krea2RMSNorm(text, eps=eps)
        self.postnorm = Krea2RMSNorm(text, eps=eps)
        self.attn = Krea2Attention(
            text,
            config.text_heads,
            config.text_kv_heads,
            eps=eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = Krea2SwiGLU(text, config.text_mlp_width, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class Krea2TextFusion(torch.nn.Module):
    """Collapse stacked per-layer text taps into one sequence: two
    blocks attend across the layer axis per token, a learned projector
    mixes the layers down to one, two blocks refine over the
    sequence."""

    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            Krea2FusionBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.text_fusion_layerwise_blocks)
        )
        self.projector = operations.linear(config.text_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            Krea2FusionBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.text_fusion_refiner_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, layers, width = x.shape
        x = x.reshape(batch * length, layers, width)
        prefetch = make_prefetch_queue(self.layerwise_blocks)
        try:
            for block in self.layerwise_blocks:
                prefetch_queue_pop(prefetch, block)
                x = block(x.contiguous())
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        x = x.view(batch, length, layers, width).permute(0, 1, 3, 2)
        x = self.projector(x).squeeze(-1)
        prefetch = make_prefetch_queue(self.refiner_blocks)
        try:
            for block in self.refiner_blocks:
                prefetch_queue_pop(prefetch, block)
                x = block(x)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        return x


class Krea2Block(torch.nn.Module):
    def __init__(
        self,
        config: _Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        features = config.features
        eps = config.rms_norm_eps
        self.mod = Krea2SharedModulation(features)
        self.prenorm = Krea2RMSNorm(features, eps=eps)
        self.postnorm = Krea2RMSNorm(features, eps=eps)
        self.attn = Krea2Attention(
            features,
            config.attention_heads,
            config.kv_heads,
            eps=eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = Krea2SwiGLU(features, config.mlp_width, operations=operations)

    def forward(
        self,
        x: torch.Tensor,
        tvec: torch.Tensor,
        rope: torch.Tensor,
        timestep_zero_index: int | None = None,
    ) -> torch.Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(tvec)
        if timestep_zero_index is None:
            x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope)
            return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        # Reference tokens after timestep_zero_index modulate with the
        # zero-timestep half of the doubled tvec batch.
        batch = x.shape[0]
        pre = self.prenorm(x)
        pre[:, :timestep_zero_index].mul_(1 + prescale[:batch]).add_(preshift[:batch])
        pre[:, timestep_zero_index:].mul_(1 + prescale[batch:]).add_(preshift[batch:])
        attended = self.attn(pre, rope)
        attended[:, :timestep_zero_index].mul_(pregate[:batch])
        attended[:, timestep_zero_index:].mul_(pregate[batch:])
        x = x + attended
        post = self.postnorm(x)
        post[:, :timestep_zero_index].mul_(1 + postscale[:batch]).add_(postshift[:batch])
        post[:, timestep_zero_index:].mul_(1 + postscale[batch:]).add_(postshift[batch:])
        update = self.mlp(post)
        update[:, :timestep_zero_index].mul_(postgate[:batch])
        update[:, timestep_zero_index:].mul_(postgate[batch:])
        return x + update


class Krea2LastLayer(torch.nn.Module):
    def __init__(self, config: _Config, *, operations: Operations) -> None:
        super().__init__()
        features = config.features
        patch_values = config.latent_channels * config.patch[0] * config.patch[1]
        self.norm = Krea2RMSNorm(features, eps=config.rms_norm_eps)
        self.linear = operations.linear(features, patch_values)
        self.modulation = Krea2FinalModulation(features)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(t)
        return self.linear((1 + scale) * self.norm(x) + shift)


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    """comfy/utils.py repeat_to_batch_size @ b78cec87 over dim 0."""
    if tensor.shape[0] > batch:
        return tensor.narrow(0, 0, batch)
    if tensor.shape[0] < batch:
        repeats = -(-batch // tensor.shape[0])
        return tensor.repeat(repeats, *(1,) * (tensor.ndim - 1)).narrow(0, 0, batch)
    return tensor


class Krea2DiT(torch.nn.Module):
    """``forward(x, timesteps, context, ...)``: x is the latent
    [batch, channels, height, width] (or [batch, channels, frames,
    height, width]; frames fold into the batch, and the conditioning
    batch must already be batch * frames), timesteps the 0..1 flow
    timestep per (folded) batch element, context the fused text-fusion
    stack [batch, tokens, text_layers * text_width]. Reference latents
    join the image sequence as extra tokens at RoPE index 1..n;
    ``ref_latents_method`` selects whether they modulate at the
    denoising timestep ("index") or at timestep zero
    ("index_timestep_zero")."""

    def __init__(
        self,
        config: _Config = KREA2_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        cfg: _Config = config
        self.config = cfg
        features = cfg.features
        patch_values = cfg.latent_channels * cfg.patch[0] * cfg.patch[1]
        head_dim = features // cfg.attention_heads
        self.rope_embedder = EmbedND(head_dim, int(cfg.rope_theta), cfg.rope_axes)
        self.first = operations.linear(patch_values, features)
        self.blocks = torch.nn.ModuleList(
            Krea2Block(cfg, operations=operations, attention_kernel=attention_kernel)
            for _ in range(cfg.transformer_blocks)
        )
        self.tmlp = torch.nn.Sequential(
            operations.linear(cfg.time_width, features),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(features, features),
        )
        self.txtfusion = Krea2TextFusion(
            cfg, operations=operations, attention_kernel=attention_kernel
        )
        self.txtmlp = torch.nn.Sequential(
            Krea2RMSNorm(cfg.text_width, eps=cfg.rms_norm_eps),
            operations.linear(cfg.text_width, features),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(features, features),
        )
        self.last = Krea2LastLayer(cfg, operations=operations)
        self.tproj = torch.nn.Sequential(
            torch.nn.GELU(approximate="tanh"),
            operations.linear(features, 6 * features),
        )

    def _process_img(
        self, x: torch.Tensor, index: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        patch_h, patch_w = self.config.patch
        batch, channels, height, width = x.shape
        pad_h, pad_w = (-height) % patch_h, (-width) % patch_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        rows, columns = x.shape[-2] // patch_h, x.shape[-1] // patch_w
        img = x.view(batch, channels, rows, patch_h, columns, patch_w)
        img = img.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, rows * columns, channels * patch_h * patch_w
        )
        ids = torch.zeros(rows, columns, 3, device=x.device, dtype=torch.float32)
        ids[..., 0] = index
        ids[..., 1] = torch.arange(rows, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(columns, device=x.device, dtype=torch.float32)[None, :]
        return img, ids.reshape(1, rows * columns, 3).repeat(batch, 1, 1), rows, columns

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        ref_latents: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        ref_latents_method: str | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        temporal: tuple[int, int] | None = None
        if x.ndim == 5:
            frames_batch, channels, frames, frames_height, frames_width = x.shape
            temporal = (frames_batch, frames)
            x = x.reshape(frames_batch * frames, channels, frames_height, frames_width)
        elif x.ndim != 4:
            raise ValueError(
                f"Krea 2 latent must be [batch, channels, height, width]"
                f" or [batch, channels, frames, height, width], got {tuple(x.shape)}"
            )
        batch, channels, height, width = x.shape
        if channels != cfg.latent_channels or min(batch, height, width) < 1:
            raise ValueError(
                f"Krea 2 latent must have {cfg.latent_channels} channels and positive extents"
            )
        if timesteps.shape != (batch,) or timesteps.device != x.device:
            raise ValueError(f"timesteps must be [batch] = ({batch},) on the latent device")
        fused = cfg.text_layers * cfg.text_width
        if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != fused:
            raise ValueError(
                f"context must be [batch, tokens, {cfg.text_layers} x {cfg.text_width}"
                f" = {fused}] (the stacked text-fusion taps), got {tuple(context.shape)}"
            )
        if context.shape[1] < 1 or context.device != x.device:
            raise ValueError("context must contain tokens on the latent device")
        if ref_latents_method not in (None, "index", "index_timestep_zero"):
            raise ValueError(
                f"unknown reference-latent method {ref_latents_method!r};"
                " expected 'index' or 'index_timestep_zero'"
            )
        stacked = context.reshape(batch, context.shape[1], cfg.text_layers, cfg.text_width)

        img, img_ids, rows, columns = self._process_img(x)
        img_tokens = img.shape[1]
        timestep_zero_index = None
        if ref_latents_method is not None and ref_latents:
            ref_parts = [img]
            ref_id_parts = [img_ids]
            for index, ref in enumerate(ref_latents, 1):
                if ref.ndim == 5:
                    ref = ref.reshape(-1, *ref.shape[-3:])
                ref = _repeat_to_batch(ref, batch)
                ref_img, ref_ids, _, _ = self._process_img(ref, index=index)
                ref_parts.append(ref_img)
                ref_id_parts.append(ref_ids)
            img = torch.cat(ref_parts, dim=1)
            img_ids = torch.cat(ref_id_parts, dim=1)
            if ref_latents_method == "index_timestep_zero":
                timestep_zero_index = img_tokens

        img = self.first(img)
        t = self.tmlp(flux_timestep_embedding(timesteps, cfg.time_width).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)
        if timestep_zero_index is not None:
            t0 = self.tmlp(
                flux_timestep_embedding(torch.zeros_like(timesteps), cfg.time_width)
                .unsqueeze(1)
                .to(img.dtype)
            )
            tvec = torch.cat((tvec, self.tproj(t0)), dim=0)

        text = self.txtmlp(self.txtfusion(stacked))
        txt_len = text.shape[1]
        txt_ids = torch.zeros(batch, txt_len, 3, device=x.device, dtype=torch.float32)
        combined = torch.cat((text, img), dim=1)
        if timestep_zero_index is not None:
            timestep_zero_index += txt_len
        rope = self.rope_embedder(torch.cat((txt_ids, img_ids), dim=1))

        prefetch = make_prefetch_queue(self.blocks)
        try:
            for block in self.blocks:
                prefetch_queue_pop(prefetch, block)
                combined = block(combined, tvec, rope, timestep_zero_index)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)

        final = self.last(combined, t)
        output = final[:, txt_len : txt_len + img_tokens]
        patch_h, patch_w = cfg.patch
        output = output.view(batch, rows, columns, channels, patch_h, patch_w)
        output = output.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, rows * patch_h, columns * patch_w
        )
        output = output[:, :, :height, :width]
        if temporal is not None:
            frames_batch, frames = temporal
            output = output.reshape(frames_batch, frames, channels, height, width).movedim(1, 2)
        return output
