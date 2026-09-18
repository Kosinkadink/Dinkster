"""SD/SDXL AutoencoderKL: the native torch architecture.

Faithful 2D port of the reference building blocks
(comfy/ldm/modules/diffusionmodules/model.py Encoder/Decoder/
ResnetBlock/AttnBlock/Upsample/Downsample and
comfy/ldm/models/autoencoder.py AutoencodingEngineLegacy/AutoencoderKL
@ b78cec87), constructed from a torch-free ``KLConfig``
(dinkster_inference.autoencoder_kl) through the typed
:class:`~dinkster_inference_torch.operations.Operations` seam instead of
the reference's ``conv_op`` class namespace. State-dict keys are
IDENTICAL to the reference for every supported checkpoint: encoder.*,
decoder.*, and (for the classic SD/SDXL layout) quant_conv.*,
post_quant_conv.*; the regularizer-only variant
(``KLConfig.quant_convs`` False, the classic Flux ae.safetensors)
builds neither conv, and the batch-norm-latent variant
(``KLConfig.batch_norm_latent``, the Flux2 VAE) adds the frozen
``bn.*`` running-statistic buffers.

Scope pins, matching what ``detect_kl_config`` accepts (everything
else refuses at detection, ledgered in ROADMAP "Native inference"):

- 2D only: the conv3d/carried video paths, ``time_compress``, and
  ``tanh_out`` are not ported.
- Mid-block attention only (``attn_resolutions`` is always empty for
  this family); per-resolution attention lists carry no parameters in
  the reference and are omitted here entirely.
- No timestep embedding (the reference's VAE always builds with
  ``temb_ch=0``, so ``temb_proj`` never exists in checkpoints).
- Resampling always convolves (``resamp_with_conv=True`` for every
  supported checkpoint); the avg-pool branch is not ported.
- Attention uses PyTorch SDPA (the reference's pytorch_attention
  reshape, single head over the channel axis). The xformers/sliced
  fallbacks are memory POLICY and stay with the executing layer (same
  ruling as tiling OOM fallback, codecs module docstring).

Deliberate divergence: no ``inference_mode``/``no_grad`` anywhere -
training program, docs/native-inference-plan.md 3.1. The reference's
in-place activations (SiLU, dropout) ARE kept; PyTorch's silu_
backward is output-based, so autograd survives (proven by the
gradient tests).

``encode()`` is the reference default: the DiagonalGaussianRegularizer
with ``sample=False`` takes the posterior MODE, deterministically.
Stochastic encodes take an explicit generator via
:meth:`DiagonalGaussian.sample` - never hidden global RNG.

``process_input``/``process_output`` are the VAE wrapper's content
transforms (comfy/sd.py @ b78cec87): image in [0,1] -> [-1,1] on the
way in; (x+1)/2 clamped to [0,1] on the way out, in place on the
codec-owned output. Their placement relative to tiling lives in
``CodecPlugin`` (content_in before the sweep, content_out after the
full average - clamp timing is observable across tile seams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from dinkster_inference.autoencoder_kl import (
    KL_BATCH_NORM_EPS,
    KL_LATENT_PATCH,
    KLConfig,
    KLMemoryEstimator,
    kl_descriptor,
)

from .attention import AttentionKernel, select_attention
from .codecs import CodecPlugin
from .operations import INITLESS, Operations, ResidencyRouted

__all__ = [
    "AttnBlock",
    "AutoencoderKL",
    "Decoder",
    "DiagonalGaussian",
    "Downsample",
    "Encoder",
    "LatentBatchNorm2d",
    "ResnetBlock",
    "Upsample",
    "crop_to_multiple",
    "kl_codec_plugin",
    "process_input",
    "process_output",
]

_DEFAULT_VAE_ATTENTION = select_attention("vae").kernel


class ResnetBlock(torch.nn.Module):
    """Reference ResnetBlock @ b78cec87, temb and conv_shortcut paths
    excluded (never present in supported checkpoints: temb_ch=0 and
    use_conv_shortcut=False)."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = operations.group_norm(in_channels)
        self.conv1 = operations.conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = operations.group_norm(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = operations.conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.nin_shortcut: torch.nn.Conv2d | None = None
        if in_channels != out_channels:
            self.nin_shortcut = operations.conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(torch.nn.Module):
    """Reference AttnBlock with the pytorch_attention path
    @ b78cec87: 1x1-conv q/k/v, single-head SDPA over flattened
    spatial positions, residual add."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        in_channels: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.norm = operations.group_norm(in_channels)
        self.q = operations.conv2d(in_channels, in_channels, 1)
        self.k = operations.conv2d(in_channels, in_channels, 1)
        self.v = operations.conv2d(in_channels, in_channels, 1)
        self.proj_out = operations.conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        batch, channels = q.shape[0], q.shape[1]
        shape = q.shape
        # the reference's reshape: (B, C, H, W) -> (B, 1, HW, C),
        # one head whose dimension is the channel axis
        q, k, v = (t.view(batch, 1, channels, -1).transpose(2, 3).contiguous() for t in (q, k, v))
        h = self._attention_kernel(q, k, v)
        h = h.transpose(2, 3).reshape(shape)
        h = self.proj_out(h)
        return x + h


class Upsample(torch.nn.Module):
    """Reference Upsample @ b78cec87, 2D with_conv path: nearest 2x
    then 3x3 conv."""

    def __init__(self, in_channels: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.conv = operations.conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(torch.nn.Module):
    """Reference Downsample @ b78cec87, 2D with_conv path: asymmetric
    zero pad (right/bottom) then 3x3 stride-2 conv."""

    def __init__(self, in_channels: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.conv = operations.conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class _DownLevel(torch.nn.Module):
    """One encoder resolution level: ``block`` list plus an optional
    ``downsample`` - the reference's anonymous nn.Module container,
    with identical child names."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        downsample: bool,
        dropout: float,
        operations: Operations,
    ) -> None:
        super().__init__()
        blocks: list[torch.nn.Module] = []
        width = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResnetBlock(
                    in_channels=width,
                    out_channels=out_channels,
                    dropout=dropout,
                    operations=operations,
                )
            )
            width = out_channels
        self.block = torch.nn.ModuleList(blocks)
        self.downsample: Downsample | None = None
        if downsample:
            self.downsample = Downsample(out_channels, operations=operations)


class _UpLevel(torch.nn.Module):
    """One decoder resolution level: ``block`` list plus an optional
    ``upsample``."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        upsample: bool,
        dropout: float,
        operations: Operations,
    ) -> None:
        super().__init__()
        blocks: list[torch.nn.Module] = []
        width = in_channels
        for _ in range(num_res_blocks + 1):
            blocks.append(
                ResnetBlock(
                    in_channels=width,
                    out_channels=out_channels,
                    dropout=dropout,
                    operations=operations,
                )
            )
            width = out_channels
        self.block = torch.nn.ModuleList(blocks)
        self.upsample: Upsample | None = None
        if upsample:
            self.upsample = Upsample(out_channels, operations=operations)


class _Mid(torch.nn.Module):
    """The shared middle stack: block_1, attn_1, block_2."""

    def __init__(
        self,
        channels: int,
        *,
        dropout: float,
        operations: Operations,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            dropout=dropout,
            operations=operations,
        )
        self.attn_1 = AttnBlock(channels, operations=operations, attention_kernel=attention_kernel)
        self.block_2 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            dropout=dropout,
            operations=operations,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.block_1(h)
        h = self.attn_1(h)
        return self.block_2(h)


class Encoder(torch.nn.Module):
    """Reference Encoder @ b78cec87, 2D, mid attention only. Emits
    2*z_channels moment channels (double_z)."""

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        dropout: float = 0.0,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        levels = len(ch_mult)
        self.conv_in = operations.conv2d(in_channels, ch, 3, stride=1, padding=1)
        in_ch_mult = (1, *ch_mult)
        down: list[torch.nn.Module] = []
        for level in range(levels):
            down.append(
                _DownLevel(
                    in_channels=ch * in_ch_mult[level],
                    out_channels=ch * ch_mult[level],
                    num_res_blocks=num_res_blocks,
                    downsample=level != levels - 1,
                    dropout=dropout,
                    operations=operations,
                )
            )
        self.down = torch.nn.ModuleList(down)
        block_in = ch * ch_mult[-1]
        self.mid = _Mid(
            block_in,
            dropout=dropout,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm_out = operations.group_norm(block_in)
        self.conv_out = operations.conv2d(block_in, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in cast("list[_DownLevel]", list(self.down)):
            for block in level.block:
                h = block(h)
            if level.downsample is not None:
                h = level.downsample(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.conv_out(h)


class Decoder(torch.nn.Module):
    """Reference Decoder @ b78cec87, 2D, mid attention only.
    ``up`` is indexed shallowest-first like the reference (levels are
    built deepest-first and prepended), so state-dict keys match."""

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        out_channels: int,
        z_channels: int,
        dropout: float = 0.0,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        levels = len(ch_mult)
        block_in = ch * ch_mult[-1]
        self.conv_in = operations.conv2d(z_channels, block_in, 3, stride=1, padding=1)
        self.mid = _Mid(
            block_in,
            dropout=dropout,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        up: list[torch.nn.Module] = []
        for level in reversed(range(levels)):
            block_out = ch * ch_mult[level]
            up.insert(
                0,
                _UpLevel(
                    in_channels=block_in,
                    out_channels=block_out,
                    num_res_blocks=num_res_blocks,
                    upsample=level != 0,
                    dropout=dropout,
                    operations=operations,
                ),
            )
            block_in = block_out
        self.up = torch.nn.ModuleList(up)
        self.norm_out = operations.group_norm(block_in)
        self.conv_out = operations.conv2d(block_in, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        for level in reversed(cast("list[_UpLevel]", list(self.up))):
            for block in level.block:
                h = block(h)
            if level.upsample is not None:
                h = level.upsample(h)
        h = self.norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.conv_out(h)


@dataclass(frozen=True)
class DiagonalGaussian:
    """The KL posterior (reference DiagonalGaussianDistribution
    @ b78cec87): mean and logvar chunked from the moment tensor (the
    quant_conv output, or encoder.conv_out directly for the
    regularizer-only variant), logvar clamped to [-30, 20]. Sampling
    takes an explicit generator; there is no hidden global RNG."""

    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def from_parameters(cls, parameters: torch.Tensor) -> DiagonalGaussian:
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        return cls(mean=mean, logvar=torch.clamp(logvar, -30.0, 20.0))

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise


class LatentBatchNorm2d(ResidencyRouted, torch.nn.BatchNorm2d):
    """The frozen latent-normalization statistics of the batch-norm
    latent VAEs (AutoencodingEngineLegacy @ 947c2749): running
    statistics only, no affine parameters, never training. Calling the
    module normalizes; :meth:`denormalize` inverts. Statistics are
    cast to the latent dtype at use, exactly like the reference's
    encode/decode ``cast_to`` calls, and leased through the residency
    binding when the component is offloaded."""

    def __init__(self, num_features: int) -> None:
        super().__init__(
            num_features,
            eps=KL_BATCH_NORM_EPS,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )
        self.eval()

    def _stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.running_mean, self.running_var
        assert mean is not None and var is not None
        return mean, var

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        # num_batches_tracked is int64 training bookkeeping above the
        # float32 materialization ceiling and unused in inference;
        # only the running statistics are ever materialized.
        binding = self._offloaded_residency()
        if binding is None:
            return None
        mean, var = self._stats()
        return binding.mechanism, (
            (binding.key("running_mean"), mean.dtype),
            (binding.key("running_var"), var.dtype),
        )

    def _normalized(
        self, latent: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.batch_norm(
            latent,
            mean.to(dtype=latent.dtype, device=latent.device),
            var.to(dtype=latent.dtype, device=latent.device),
            momentum=0.1,
            eps=self.eps,
        )

    def _denormalized(
        self, latent: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        scale = torch.sqrt(
            var.view(1, -1, 1, 1).to(dtype=latent.dtype, device=latent.device) + self.eps
        )
        center = mean.view(1, -1, 1, 1).to(dtype=latent.dtype, device=latent.device)
        return latent * scale + center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        binding = self._offloaded_residency()
        if binding is None:
            return self._normalized(input, mean, var)
        with binding.lease() as lease:
            return self._normalized(
                input,
                lease.get("running_mean", dtype=mean.dtype),
                lease.get("running_var", dtype=var.dtype),
            )

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        binding = self._offloaded_residency()
        if binding is None:
            return self._denormalized(latent, mean, var)
        with binding.lease() as lease:
            return self._denormalized(
                latent,
                lease.get("running_mean", dtype=mean.dtype),
                lease.get("running_var", dtype=var.dtype),
            )


class AutoencoderKL(torch.nn.Module):
    """Reference AutoencoderKL/AutoencodingEngineLegacy @ b78cec87:
    encoder -> quant_conv -> posterior; post_quant_conv -> decoder.
    With ``config.quant_convs`` False (regularizer-only
    AutoencodingEngine, the classic Flux ae.safetensors) both convs
    are omitted: encoder -> posterior; decoder(latent) directly.
    The reference's ``max_batch_size`` chunking is batch policy and
    stays with the executing layer.

    With ``config.batch_norm_latent`` (the Flux2 VAE) the external
    latent is the packed, normalized form: ``encode`` packs each 2x2
    latent patch into channels and normalizes with the frozen ``bn``
    running statistics; ``decode`` denormalizes and unpacks before
    post_quant_conv. ``encode_posterior`` stays in the unpacked
    embed_dim space (the reference regularizes before packing);
    stochastic draws from it go through :meth:`pack_latent` to reach
    the external contract."""

    def __init__(
        self,
        config: KLConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_VAE_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            in_channels=config.in_channels,
            z_channels=config.z_channels,
            dropout=config.dropout,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.decoder = Decoder(
            ch=config.decoder_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            out_channels=config.out_channels,
            z_channels=config.z_channels,
            dropout=config.dropout,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.quant_conv: torch.nn.Conv2d | None
        self.post_quant_conv: torch.nn.Conv2d | None
        if config.quant_convs:
            self.quant_conv = operations.conv2d(2 * config.z_channels, 2 * config.embed_dim, 1)
            self.post_quant_conv = operations.conv2d(config.embed_dim, config.z_channels, 1)
        else:
            # Regularizer-only AutoencodingEngine (the classic Flux
            # ae.safetensors; comfy/sd.py @ b78cec87): the posterior
            # consumes encoder.conv_out's moments directly and the
            # decoder consumes the latent directly.
            self.quant_conv = None
            self.post_quant_conv = None
        self.bn: LatentBatchNorm2d | None = None
        if config.batch_norm_latent:
            # The reference construction (AutoencodingEngineLegacy
            # @ b78cec87): frozen running statistics only, no affine.
            # Buffer keys bn.running_mean / bn.running_var /
            # bn.num_batches_tracked match both real Flux2 layouts.
            self.bn = LatentBatchNorm2d(KL_LATENT_PATCH * KL_LATENT_PATCH * config.z_channels)

    def encode_posterior(self, content: torch.Tensor) -> DiagonalGaussian:
        """The full posterior, for stochastic encodes and training."""
        moments = self.encoder(content)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        return DiagonalGaussian.from_parameters(moments)

    def pack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """The batch-norm-latent packing (AutoencodingEngineLegacy
        encode @ b78cec87): each 2x2 spatial patch moves into
        channels (channel-major, then row, then column - the
        reference's ``c (i pi) (j pj) -> (c pi pj) i j``), then the
        frozen running statistics normalize."""
        bn = self.bn
        assert bn is not None
        batch, channels, height, width = latent.shape
        p = KL_LATENT_PATCH
        latent = latent.reshape(batch, channels, height // p, p, width // p, p)
        latent = latent.permute(0, 1, 3, 5, 2, 4)
        latent = latent.reshape(batch, channels * p * p, height // p, width // p)
        return bn(latent)

    def unpack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """The inverse of :meth:`pack_latent` (AutoencodingEngineLegacy
        decode @ b78cec87): denormalize with the frozen running
        statistics, then spread the packed channels back over 2x2
        spatial patches."""
        bn = self.bn
        assert bn is not None
        latent = bn.denormalize(latent)
        batch, packed, height, width = latent.shape
        p = KL_LATENT_PATCH
        latent = latent.reshape(batch, packed // (p * p), p, p, height, width)
        latent = latent.permute(0, 1, 4, 2, 5, 3)
        return latent.reshape(batch, packed // (p * p), height * p, width * p)

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        """Deterministic encode: posterior mode (the reference's
        DiagonalGaussianRegularizer default, sample=False), packed
        and normalized when batch_norm_latent."""
        latent = self.encode_posterior(content).mode()
        if self.bn is not None:
            latent = self.pack_latent(latent)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.bn is not None:
            latent = self.unpack_latent(latent)
        if self.post_quant_conv is not None:
            latent = self.post_quant_conv(latent)
        return self.decoder(latent)


def crop_to_multiple(content: torch.Tensor, multiple: int) -> torch.Tensor:
    """comfy/sd.py VAE.vae_encode_crop_pixels @ b78cec87 (spatial
    branch): center-crop every spatial dimension of (N, C, *spatial)
    down to a multiple of the codec's downscale, so encode geometry
    is exact instead of floor-truncating inside the downsamplers.
    Returns views (narrow), never copies. Content smaller than one
    downscale step refuses instead of emitting an empty tensor."""
    for dim in range(2, content.dim()):
        size = content.shape[dim]
        cropped = (size // multiple) * multiple
        if cropped == 0:
            raise ValueError(
                f"content dimension {dim} has extent {size}, smaller"
                f" than one {multiple}x downscale step"
            )
        if cropped != size:
            content = content.narrow(dim, (size % multiple) // 2, cropped)
    return content


def process_input(content: torch.Tensor) -> torch.Tensor:
    """comfy/sd.py VAE.process_input @ b78cec87: [0,1] -> [-1,1].
    Out of place; the caller's content is never mutated."""
    return content * 2.0 - 1.0


def process_output(content: torch.Tensor) -> torch.Tensor:
    """comfy/sd.py VAE.process_output @ b78cec87: [-1,1] -> [0,1],
    clamped. IN PLACE on the codec-owned decode output, exactly like
    the reference."""
    return content.add_(1.0).div_(2.0).clamp_(0.0, 1.0)


def kl_codec_plugin(model: AutoencoderKL) -> CodecPlugin:
    """Bind a constructed KL autoencoder into a registrable codec
    plugin: descriptor from the torch-free layer, the model as both
    encoder and decoder, the SD content transforms (crop to the
    downscale grid, then [0,1] <-> [-1,1]) at the wrapper boundary,
    and the reference memory formulas (decode x4 for
    batch-norm-latent, comfy/sd.py @ b78cec87)."""
    downscale = model.config.spatial_downscale

    def crop(content: torch.Tensor) -> torch.Tensor:
        return crop_to_multiple(content, downscale)

    return CodecPlugin(
        descriptor=kl_descriptor(model.config),
        encoder=model,
        decoder=model,
        memory=KLMemoryEstimator(decode_multiplier=4.0 if model.config.batch_norm_latent else 1.0),
        content_crop=crop,
        content_in=process_input,
        content_out=lambda value: process_output(value.float()),
    )
