"""SD1/SDXL diffusion UNet: the native torch architecture.

Faithful port of the reference 2D image UNet
(comfy/ldm/modules/diffusionmodules/openaimodel.py UNetModel /
ResBlock / Upsample / Downsample / timestep_embedding and
comfy/ldm/modules/attention.py SpatialTransformer /
BasicTransformerBlock / CrossAttention / FeedForward / GEGLU
@ b78cec87), constructed from a torch-free
:class:`~dinkster_inference.unet.UNetConfig` through the typed
:class:`~dinkster_inference_torch.operations.Operations` seam.
State-dict keys are IDENTICAL to the reference: time_embed.*,
label_emb.0.*, input_blocks.N.M.*, middle_block.M.*,
output_blocks.N.M.*, out.*.

Scope pins:

- Attention runs through PyTorch SDPA with the reference's
  head reshape and SDP_BATCH_LIMIT batch chunking
  (comfy/ldm/modules/attention.py attention_pytorch), never
  masked - no supported SD1/SDXL UNet passes a mask. The
  fp32-attention pin (SD2.x ``attn_precision``) is not ported.
- Classic SD1.5 ControlNet residuals apply to the saved input-block
  skips and middle activation. ``transformer_options`` patch/wrapper
  seams are not supported.
- 2D only: temporal/video modes (VideoResBlock,
  SpatialVideoTransformer), ``resblock_updown``,
  ``use_scale_shift_norm``, codebook prediction, and int/continuous
  class embeddings have no supported consumer.
- ``use_checkpoint`` (gradient checkpointing) is a training-memory
  policy and belongs to the training program, not the architecture.

Deliberate preservation: no ``inference_mode``/``no_grad`` anywhere
(training program, docs/native-inference-plan.md 3.1); residual adds
are out-of-place so autograd and torch.compile see plain dataflow.
The reference's trailing ``h.type(x.dtype)`` is a no-op without the
unported fp32-attention mode and is dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from dinkster_inference import AttentionGuidanceDescriptor
from dinkster_inference.unet import UNetConfig

from .attention import AttentionKernel, select_attention
from .ipadapter import SD15AttentionExecutionContext
from .model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from .operations import INITLESS, Operations

if TYPE_CHECKING:
    from .controlnet import SDControlResiduals

__all__ = [
    "AttentionGuidanceContext",
    "BasicTransformerBlock",
    "CrossAttention",
    "Downsample",
    "FeedForward",
    "Geglu",
    "ResBlock",
    "SpatialTransformer",
    "UNetModel",
    "Upsample",
    "timestep_embedding",
]

_DEFAULT_UNET_ATTENTION = select_attention("unet").kernel


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding, float32, [N x dim] - including
    the reference's zero pad for odd dim
    (comfy/ldm/modules/diffusionmodules/util.py @ b78cec87)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Upsample(torch.nn.Module):
    """2x nearest upsample plus 3x3 conv; ``output_shape`` overrides
    the doubled spatial size so odd input extents round-trip."""

    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = operations.conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if output_shape is None:
            size = (x.shape[2] * 2, x.shape[3] * 2)
        else:
            size = (output_shape[2], output_shape[3])
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


class Downsample(torch.nn.Module):
    """Strided 3x3 conv downsample (``op`` name per the reference)."""

    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.op = operations.conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(torch.nn.Module):
    """Timestep-conditioned residual block, the non-updown
    non-scale-shift path every supported UNet uses."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.in_layers = torch.nn.Sequential(
            operations.group_norm(channels, eps=1e-5),
            torch.nn.SiLU(),
            operations.conv2d(channels, out_channels, 3, padding=1),
        )
        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.linear(emb_channels, out_channels),
        )
        self.out_layers = torch.nn.Sequential(
            operations.group_norm(out_channels, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            operations.conv2d(out_channels, out_channels, 3, padding=1),
        )
        if out_channels == channels:
            self.skip_connection: torch.nn.Module = torch.nn.Identity()
        else:
            self.skip_connection = operations.conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None]
        h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h


class Geglu(torch.nn.Module):
    """Gated GELU projection (attention.py GEGLU @ b78cec87)."""

    def __init__(self, dim_in: int, dim_out: int, *, operations: Operations) -> None:
        super().__init__()
        self.proj = operations.linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(torch.nn.Module):
    """Transformer MLP; always the gated (GEGLU) variant - every
    supported UNet block passes glu=True."""

    def __init__(
        self,
        dim: int,
        *,
        dropout: float = 0.0,
        operations: Operations,
    ) -> None:
        super().__init__()
        inner = dim * 4
        self.net = torch.nn.Sequential(
            Geglu(dim, inner, operations=operations),
            torch.nn.Dropout(p=dropout),
            operations.linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(torch.nn.Module):
    """Multi-head attention over flattened spatial tokens; self- when
    ``context`` is None. SDPA with the reference head reshape."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        heads: int,
        dim_head: int,
        *,
        dropout: float = 0.0,
        operations: Operations,
        attention_kernel: AttentionKernel = _DEFAULT_UNET_ATTENTION,
        site_id: str | None = None,
    ) -> None:
        super().__init__()
        inner = heads * dim_head
        source = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.site_id = site_id
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.to_q = operations.linear(query_dim, inner, bias=False)
        self.to_k = operations.linear(source, inner, bias=False)
        self.to_v = operations.linear(source, inner, bias=False)
        self.to_out = torch.nn.Sequential(
            operations.linear(inner, query_dim),
            torch.nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        ipadapter: SD15AttentionExecutionContext | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        source = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(source)
        v = self.to_v(source)
        batch = q.shape[0]

        def heads_first(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = heads_first(q), heads_first(k), heads_first(v)
        out = self._attention_kernel(q, k, v)
        if ipadapter is not None:
            if self.site_id is None or spatial_shape is None:
                raise ValueError("IP-Adapter requires a canonical attn2 site and spatial shape")
            out = ipadapter.apply(
                self.site_id,
                q,
                out,
                height=spatial_shape[0],
                width=spatial_shape[1],
                kernel=self._attention_kernel,
            )
        out = out.transpose(1, 2).reshape(batch, -1, self.heads * self.dim_head)
        return self.to_out(out)


@dataclass(frozen=True)
class AttentionGuidanceContext:
    """Row-mapped attention rewrites for one fused batched forward.

    ``positive`` and ``negative`` are half-open batch-row ranges of the
    conditional and unconditional streams. ``apply`` rewrites the
    conditional rows of a fresh self-attention output in place, one
    descriptor at a time in contribution order, each observing the
    previous rewrite - matching the reference's chained attn1 output
    patches (comfy/samplers.py calc_cond_batch @ b78cec87)."""

    transforms: tuple[AttentionGuidanceDescriptor[torch.Tensor], ...]
    positive: tuple[int, int]
    negative: tuple[int, int]

    def apply(self, out: torch.Tensor) -> torch.Tensor:
        pos = slice(*self.positive)
        neg = slice(*self.negative)
        for descriptor in self.transforms:
            out[pos] = descriptor.transform(out[pos], out[neg])
        return out


class BasicTransformerBlock(torch.nn.Module):
    """Pre-norm self-attention, cross-attention, gated MLP - the
    ``disable_self_attn=False`` block every supported UNet builds."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        context_dim: int,
        *,
        dropout: float = 0.0,
        operations: Operations,
        attention_kernel: AttentionKernel = _DEFAULT_UNET_ATTENTION,
        site_id: str | None = None,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            dim,
            None,
            heads,
            dim_head,
            dropout=dropout,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.attn2 = CrossAttention(
            dim,
            context_dim,
            heads,
            dim_head,
            dropout=dropout,
            operations=operations,
            attention_kernel=attention_kernel,
            site_id=site_id,
        )
        self.ff = FeedForward(dim, dropout=dropout, operations=operations)
        self.norm1 = operations.layer_norm(dim)
        self.norm2 = operations.layer_norm(dim)
        self.norm3 = operations.layer_norm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_guidance: AttentionGuidanceContext | None = None,
        ipadapter: SD15AttentionExecutionContext | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        n = self.attn1(self.norm1(x))
        if attention_guidance is not None:
            n = attention_guidance.apply(n)
        x = n + x
        x = (
            self.attn2(
                self.norm2(x),
                context=context,
                ipadapter=ipadapter,
                spatial_shape=spatial_shape,
            )
            + x
        )
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(torch.nn.Module):
    """Project image features to tokens, run transformer blocks, and
    project back with a residual. ``use_linear`` picks linear vs 1x1
    conv projections, applied in the reference's exact order (conv
    before flatten, linear after)."""

    def __init__(
        self,
        in_channels: int,
        heads: int,
        dim_head: int,
        depth: int,
        context_dim: int,
        use_linear: bool,
        *,
        dropout: float = 0.0,
        operations: Operations,
        attention_kernel: AttentionKernel = _DEFAULT_UNET_ATTENTION,
        site_prefix: str | None = None,
    ) -> None:
        super().__init__()
        inner = heads * dim_head
        self.use_linear = use_linear
        self.norm = operations.group_norm(in_channels, eps=1e-6)
        if use_linear:
            self.proj_in: torch.nn.Module = operations.linear(in_channels, inner)
            # The reference constructs proj_out as
            # Linear(in_channels, inner) - transposed relative to
            # proj_in, coincidentally square since inner ==
            # in_channels for every supported profile.
            self.proj_out: torch.nn.Module = operations.linear(in_channels, inner)
        else:
            self.proj_in = operations.conv2d(in_channels, inner, 1)
            self.proj_out = operations.conv2d(inner, in_channels, 1)
        self.transformer_blocks = torch.nn.ModuleList(
            BasicTransformerBlock(
                inner,
                heads,
                dim_head,
                context_dim,
                dropout=dropout,
                operations=operations,
                attention_kernel=attention_kernel,
                site_id=(
                    None
                    if site_prefix is None
                    else f"{site_prefix}.transformer_blocks.{index}.attn2"
                ),
            )
            for index in range(depth)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_guidance: AttentionGuidanceContext | None = None,
        ipadapter: SD15AttentionExecutionContext | None = None,
    ) -> torch.Tensor:
        _, _, height, width = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        prefetch = make_prefetch_queue(self.transformer_blocks)
        try:
            for block in self.transformer_blocks:
                prefetch_queue_pop(prefetch, block)
                x = block(
                    x,
                    context,
                    attention_guidance,
                    ipadapter,
                    (height, width),
                )
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], height, width, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class _Layers(torch.nn.Sequential):
    """The reference's TimestepEmbedSequential: one input/middle/
    output block whose children take their extra inputs by type."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
        output_shape: tuple[int, ...] | None = None,
        attention_guidance: AttentionGuidanceContext | None = None,
        ipadapter: SD15AttentionExecutionContext | None = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, attention_guidance, ipadapter)
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x


class UNetModel(torch.nn.Module):
    """The SD1/SDXL denoiser: input/middle/output blocks over a skip
    stack, timestep plus optional ADM ("sequential") conditioning."""

    def __init__(
        self,
        config: UNetConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_UNET_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        m = config.model_channels
        emb = config.time_embed_dim
        dropout = config.dropout

        self.time_embed = torch.nn.Sequential(
            operations.linear(m, emb),
            torch.nn.SiLU(),
            operations.linear(emb, emb),
        )
        if config.adm_in_channels is not None:
            self.label_emb = torch.nn.Sequential(
                torch.nn.Sequential(
                    operations.linear(config.adm_in_channels, emb),
                    torch.nn.SiLU(),
                    operations.linear(emb, emb),
                )
            )

        attention_sites: list[tuple[str, int]] = []

        def transformer(ch: int, depth: int, site_prefix: str) -> SpatialTransformer:
            heads, dim_head = config.heads_for(ch)
            attention_sites.extend(
                (f"{site_prefix}.transformer_blocks.{index}.attn2", ch) for index in range(depth)
            )
            return SpatialTransformer(
                ch,
                heads,
                dim_head,
                depth,
                config.context_dim,
                config.use_linear_in_transformer,
                dropout=dropout,
                operations=operations,
                attention_kernel=attention_kernel,
                site_prefix=site_prefix,
            )

        self.input_blocks = torch.nn.ModuleList(
            [_Layers(operations.conv2d(config.in_channels, m, 3, padding=1))]
        )
        depths = list(config.transformer_depth)
        input_chans = [m]
        ch = m
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks[level]):
                out_ch = mult * m
                layers: list[torch.nn.Module] = [
                    ResBlock(ch, emb, out_ch, dropout=dropout, operations=operations)
                ]
                ch = out_ch
                depth = depths.pop(0)
                if depth > 0:
                    index = len(self.input_blocks)
                    layers.append(transformer(ch, depth, f"input_blocks.{index}.1"))
                self.input_blocks.append(_Layers(*layers))
                input_chans.append(ch)
            if level != len(config.channel_mult) - 1:
                self.input_blocks.append(_Layers(Downsample(ch, operations=operations)))
                input_chans.append(ch)

        self.middle_block = _Layers(
            ResBlock(ch, emb, ch, dropout=dropout, operations=operations),
            transformer(ch, config.transformer_depth_middle, "middle_block.1"),
            ResBlock(ch, emb, ch, dropout=dropout, operations=operations),
        )

        self.output_blocks = torch.nn.ModuleList()
        out_depths = list(config.transformer_depth_output)
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            for block in range(config.num_res_blocks[level] + 1):
                ich = input_chans.pop()
                out_ch = mult * m
                layers = [
                    ResBlock(
                        ch + ich,
                        emb,
                        out_ch,
                        dropout=dropout,
                        operations=operations,
                    )
                ]
                ch = out_ch
                depth = out_depths.pop()
                if depth > 0:
                    index = len(self.output_blocks)
                    layers.append(transformer(ch, depth, f"output_blocks.{index}.1"))
                if level and block == config.num_res_blocks[level]:
                    layers.append(Upsample(ch, operations=operations))
                self.output_blocks.append(_Layers(*layers))

        self.out = torch.nn.Sequential(
            operations.group_norm(ch, eps=1e-5),
            torch.nn.SiLU(),
            operations.conv2d(m, config.out_channels, 3, padding=1),
        )
        self.attention_sites = tuple(attention_sites)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        control: SDControlResiduals | None = None,
        attention_guidance: AttentionGuidanceContext | None = None,
        ipadapter: SD15AttentionExecutionContext | None = None,
    ) -> torch.Tensor:
        if (y is None) == (self.config.adm_in_channels is not None):
            raise ValueError(
                "y must be provided if and only if the model is"
                " ADM-conditioned (adm_in_channels ="
                f" {self.config.adm_in_channels})"
            )
        if y is not None and y.shape[0] != x.shape[0]:
            # The reference asserts this (openaimodel.py forward);
            # a silent broadcast would condition every sample on
            # one embedding.
            raise ValueError(f"y batch {y.shape[0]} does not match x batch {x.shape[0]}")
        t_emb = timestep_embedding(timesteps, self.config.model_channels).to(x.dtype)
        emb = self.time_embed(t_emb)
        if y is not None:
            emb = emb + self.label_emb(y)

        h = x
        hs = []
        input_prefetch = make_prefetch_queue(self.input_blocks)
        try:
            for module in self.input_blocks:
                prefetch_queue_pop(input_prefetch, module)
                h = module(
                    h,
                    emb,
                    context,
                    attention_guidance=attention_guidance,
                    ipadapter=ipadapter,
                )
                hs.append(h)
            prefetch_queue_pop(input_prefetch, None)
        finally:
            close_prefetch_queue(input_prefetch)
        h = self.middle_block(
            h,
            emb,
            context,
            attention_guidance=attention_guidance,
            ipadapter=ipadapter,
        )
        if control is not None:
            if len(control.down) != len(hs):
                raise ValueError(
                    f"control has {len(control.down)} down residuals for {len(hs)} UNet skips"
                )
            h = h + control.middle.to(h.dtype)
        output_prefetch = make_prefetch_queue(self.output_blocks)
        try:
            for module in self.output_blocks:
                prefetch_queue_pop(output_prefetch, module)
                skip = hs.pop()
                if control is not None:
                    skip = skip + control.down[len(hs)].to(skip.dtype)
                h = torch.cat([h, skip], dim=1)
                output_shape = hs[-1].shape if hs else None
                h = module(h, emb, context, output_shape, attention_guidance, ipadapter)
            prefetch_queue_pop(output_prefetch, None)
        finally:
            close_prefetch_queue(output_prefetch)
        return self.out(h)
